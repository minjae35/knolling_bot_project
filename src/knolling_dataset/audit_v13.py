"""Policy-aware audit; human approval is never synthesized by automation."""
from dataclasses import replace
import json
from pathlib import Path
from .schema import scene_from_dict,instruction_from_dict,digest
from .policies import BALANCED,policy_metadata,require_single_policy
from .teacher import solve
from .quality import target_hash,target_changed,layout_metrics,counterfactual_audit
from .instructions import clarify_instruction


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_v13(root,records,provenance, *, invariance=True):
    from .builder import file_hash
    root=Path(root); errors=[]
    teachers=read_jsonl(root/"teacher_audit.jsonl")
    try:
        require_single_policy([provenance["teacher_policy"]]+[r["teacher_policy"] for r in teachers])
    except ValueError:
        errors.append("mixed_teacher_policy")
    expected=policy_metadata(BALANCED)
    if provenance.get("teacher_config_hash")!=expected["teacher_config_hash"]:
        errors.append("teacher_config_hash")
    by_id={r["sample_id"]:r for r in teachers}
    if len(by_id)!=len(teachers) or set(by_id)!={r["sample_id"] for r in records}:
        errors.append("teacher_audit_coverage")
    counts={"policy_reproduction":0,"array_order":0,"language":0,"asset_alias":0,"reapplied_noop":0}
    for record in records:
        sid=record["sample_id"]; source=scene_from_dict(record["source"]); ins=instruction_from_dict(record["instruction"])
        expected_hash=target_hash(scene_from_dict(record["target"]))
        audit=by_id.get(sid,{})
        if audit.get("target_hash")!=expected_hash: errors.append(sid+":teacher_target_hash")
        result=solve(source,ins,policy=BALANCED)
        if result.status!="ok" or target_hash(result.target)!=expected_hash:
            errors.append(sid+":policy_reproduction"); continue
        counts["policy_reproduction"]+=1
        if any(digest(audit.get(k))!=digest(v) for k,v in result.audit.items()):
            errors.append(sid+":teacher_scores_or_configuration")
        if invariance:
            variants=[("array_order",replace(source,objects=tuple(reversed(source.objects))),replace(ins,selected_ids=tuple(reversed(ins.selected_ids)),locked_ids=tuple(reversed(ins.locked_ids)))),
                      ("language",source,replace(ins,language="en" if ins.language=="ko" else "ko",text="Equivalent semantic paraphrase / 같은 의미의 문장")),
                      ("asset_alias",replace(source,seed=source.seed+1,objects=tuple(replace(o,asset=replace(o.asset,asset_id="alias_"+o.object_id)) for o in source.objects)),ins)]
            for name,scene,instruction in variants:
                other=solve(scene,instruction,policy=BALANCED)
                if other.status!="ok" or target_hash(other.target)!=expected_hash: errors.append(sid+":"+name)
                else: counts[name]+=1
            current=replace(source,objects=tuple(replace(o,source_pose=o.target_pose,target_pose=None,locked=False,move_label=False) for o in result.target.objects))
            repeat=solve(current,ins,policy=BALANCED)
            if repeat.status!="ok" or repeat.mode!="no_op" or repeat.rows or repeat.relations or any(o.target_pose!=o.source_pose for o in repeat.target.objects):
                errors.append(sid+":reapplied_noop")
            else: counts["reapplied_noop"]+=1
    parent_hash=provenance.get("parent_dataset_manifest_sha256")
    if parent_hash:
        if file_hash(root/"parent_manifest.jsonl")!=parent_hash: errors.append("parent_manifest_hash")
        parent_manifest={r["sample_id"]:r for r in read_jsonl(root/"parent_manifest.jsonl")}
        migrations=read_jsonl(root/"migration.jsonl"); migration_by_id={r["sample_id"]:r for r in migrations}
        if len(migrations)!=len(records) or set(migration_by_id)!=set(parent_manifest) or set(migration_by_id)!=set(by_id):
            errors.append("migration_coverage")
        if len(records)!=provenance["requested_sample_count"]: errors.append("incomplete_migration")
        snapshot=json.loads((root/"parent_snapshot.json").read_text())
        for name in ("generation.json","asset_registry.json"):
            if file_hash(root/name)!=snapshot[name]: errors.append("parent_"+name)
        source_hashes=json.loads((root/"source_file_hashes.json").read_text())
        expected_source_files={str(Path(r["source_render_dir"])/name) for r in records for name in ("rgb.png","instance_mask.png","render.json")}
        if set(source_hashes)!=expected_source_files: errors.append("source_hash_coverage")
        for relative,sha in source_hashes.items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha!=snapshot[relative] or file_hash(path)!=sha:
                errors.append("source_file_changed:"+relative)
        for new in records:
            sid=new["sample_id"]; path=root/f"parent_annotations/{sid}.json"
            if file_hash(path)!=parent_manifest[sid]["annotation_sha256"]: errors.append(sid+":parent_annotation_hash")
            old=json.loads(path.read_text()); row=migration_by_id[sid]
            if digest(old["source"])!=digest(new["source"]) or any(old[k]!=new[k] for k in ("scene_family_id","split","role","mode","no_op_type")):
                errors.append(sid+":source_or_partition_changed")
            if clarify_instruction(instruction_from_dict(old["instruction"]))!=instruction_from_dict(new["instruction"]):
                errors.append(sid+":instruction_contract")
            a,b=scene_from_dict(old["target"]),scene_from_dict(new["target"])
            changed=target_changed(a,b)
            if row["target_changed"]!=changed or row["target_render_action"]!=("render" if changed else "reuse"):
                errors.append(sid+":render_reuse_decision")
            if row["old_target_hash"]!=target_hash(a) or row["new_target_hash"]!=target_hash(b): errors.append(sid+":migration_target_hash")
            if not changed:
                for name in ("rgb.png","instance_mask.png","render.json"):
                    if file_hash(root/new["target_render_dir"]/name)!=old["file_hashes"][str(Path(old["target_render_dir"])/name)]:
                        errors.append(sid+":unchanged_target_rerendered")
            source=scene_from_dict(old["source"]); ins=instruction_from_dict(new["instruction"])
            before,after=layout_metrics(source,a,ins),layout_metrics(source,b,ins)
            if row["old_metrics"]!=before or row["new_metrics"]!=after: errors.append(sid+":migration_scores")
            if old["mode"]=="partial" and after["J_move"]>before["J_move"]+1e-12: errors.append(sid+":partial_movement_increased")
    pairs=counterfactual_audit(records)
    if digest(read_jsonl(root/"counterfactual_audit.jsonl"))!=digest(pairs): errors.append("counterfactual_audit_mismatch")
    reviews=read_jsonl(root/"human_review.jsonl")
    if len(reviews)!=len(records) or {r["sample_id"] for r in reviews}!=set(by_id): errors.append("human_review_coverage")
    approved=[]
    for r in reviews:
        if r.get("acceptable") is True:
            if not r.get("reviewer") or not r.get("reviewed_at") or r.get("status")!="reviewed":
                errors.append(r["sample_id"]+":unattributed_human_approval")
            else: approved.append(r["sample_id"])
    # This phase does not export a training set or automatically approve reviews.
    if read_jsonl(root/"training_approved.jsonl"):
        errors.append("training_export_not_enabled_in_dataset_review_phase")
    return {"valid":not errors,"errors":errors,"teacher_policy":BALANCED,"invariance_counts":counts,
            "counterfactual_pairs":len(pairs),"diagnostic_counterfactual_pairs":sum(p["diagnostic_pair"] for p in pairs),
            "human_reviewed":sum(r.get("status")=="reviewed" for r in reviews),"human_approved_samples":approved,
            "human_review_complete":len(approved)==len(records),"training_ready":False}
