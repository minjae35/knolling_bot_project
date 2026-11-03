"""Fixed-input migration. Never resample a source and never write to the parent."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
from .schema import scene_from_dict,instruction_from_dict,scene_dict,canonical_json,digest
from .policies import LEGACY,BALANCED,LEGACY_MANIFEST,policy_metadata,resolve_dataset_policy
from .instructions import clarify_instruction
from .teacher import solve
from .builder import write_json,file_hash,_record,_object_annotations
from .rendering import make_job,render_batch,validate_render
from .quality import target_hash,target_changed,layout_metrics,partial_quality,counterfactual_audit


def read_records(root):
    root=Path(root)
    manifest=[json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    records=[]
    for entry in manifest:
        path=(root/entry["annotation"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path)!=entry["annotation_sha256"]:
            raise ValueError("invalid parent annotation path/hash")
        records.append(json.loads(path.read_text()))
    return manifest,records


def tree_hashes(root):
    root=Path(root)
    return {str(p.relative_to(root)):file_hash(p) for p in root.rglob("*") if p.is_file()}


def migrate_record(old):
    source=scene_from_dict(old["source"])
    ins=clarify_instruction(instruction_from_dict(old["instruction"]))
    result=solve(source,ins,policy=BALANCED)
    audit=dict(result.audit,sample_id=old["sample_id"],status=result.status,reason=result.reason)
    row={"sample_id":old["sample_id"],"scene_family_id":old["scene_family_id"],"split":old["split"],
         "status":result.status,"reason":result.reason,"old_target_hash":target_hash(scene_from_dict(old["target"])),
         "old_raw_target_hash":target_hash(scene_from_dict(old["target"]),canonical=False)}
    if result.status!="ok":
        row["change_kind"]="generation_failure"
        return None,audit,row
    new=_record(source,ins,result,old["sample_id"],old["role"],old["no_op_type"])
    new["source"]=deepcopy(old["source"])  # Preserve serialized source exactly.
    changed=target_changed(scene_from_dict(old["target"]),result.target)
    text_changed=old["instruction"]["text"]!=ins.text
    row.update(target_changed=changed,text_changed=text_changed,
               change_kind="target_changed" if changed else "text_only" if text_changed else "unchanged",
               new_target_hash=target_hash(result.target),new_raw_target_hash=target_hash(result.target,canonical=False),
               old_metrics=layout_metrics(source,scene_from_dict(old["target"]),ins),
               new_metrics=layout_metrics(source,result.target,ins),old_rows=old["rows"],new_rows=result.rows)
    audit["target_hash"]=row["new_target_hash"]
    return new,audit,row


def write_jsonl(path,rows):
    Path(path).write_text("".join(canonical_json(r)+"\n" for r in rows))


def pending_reviews(records):
    return [{"sample_id":r["sample_id"],"reviewer":None,"reviewed_at":None,"instruction_satisfaction":None,
             "tidiness":None,"partial_movement_naturalness":None,"legibility":None,"overall_score":None,
             "acceptable":None,"status":"pending","notes":""} for r in records]


def initialize_balanced_sidecars(root,records):
    """New v1.3 builds, kept separate from fixed-source migration history."""
    root=Path(root); teachers=[]
    for r in records:
        result=solve(scene_from_dict(r["source"]),instruction_from_dict(r["instruction"]),policy=BALANCED)
        if result.status!="ok" or target_hash(result.target)!=target_hash(scene_from_dict(r["target"])):
            raise RuntimeError("policy changed while building dataset")
        teachers.append(dict(result.audit,sample_id=r["sample_id"],status=result.status,reason=result.reason,target_hash=target_hash(result.target)))
    write_jsonl(root/"teacher_audit.jsonl",teachers)
    write_jsonl(root/"counterfactual_audit.jsonl",counterfactual_audit(records))
    write_jsonl(root/"human_review.jsonl",pending_reviews(records))
    write_jsonl(root/"training_approved.jsonl",[])


def _copy_verified(parent,root,relative,destination=None):
    source=(parent/relative).resolve()
    dest=(root/(destination or relative)).resolve()
    if not source.is_relative_to(parent.resolve()) or not dest.is_relative_to(root.resolve()):
        raise ValueError("copy path escapes dataset")
    expected=file_hash(source)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if file_hash(dest)!=expected: raise ValueError("conflicting destination reuse")
    else:
        shutil.copy2(source,dest)  # Independent file, never hard-link.
    if file_hash(dest)!=expected or (source.stat().st_dev,source.stat().st_ino)==(dest.stat().st_dev,dest.stat().st_ino):
        raise RuntimeError("copy integrity/independence failure")
    return expected


def migrate_dataset(parent,output, *, registered_legacy_hash=LEGACY_MANIFEST):
    from .splits import audit_dataset
    parent,root=Path(parent).resolve(),Path(output).resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("migration destination must be separate from parent")
    if root.exists() and any(root.iterdir()):
        raise ValueError("migration destination must be empty")
    if resolve_dataset_policy(parent,registered_legacy_hash=registered_legacy_hash)!=LEGACY:
        raise ValueError("this migration requires a registered legacy parent")
    baseline=audit_dataset(parent,read_only=True)
    if not baseline["valid"]: raise ValueError(f"parent audit failed: {baseline['errors']}")
    snapshot=tree_hashes(parent)
    parent_manifest,old_records=read_records(parent)
    if len({r["scene_family_id"] for r in old_records})>100:
        raise ValueError("migration is capped at 100 source families")
    planned=[migrate_record(old) for old in old_records]
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"parent_snapshot.json",snapshot)
    for name in ("manifest.jsonl","provenance.json"):
        _copy_verified(parent,root,name,"parent_"+name)
    for name in ("generation.json","asset_registry.json"):
        _copy_verified(parent,root,name)
    for entry in parent_manifest:
        _copy_verified(parent,root,entry["annotation"],f"parent_annotations/{entry['sample_id']}.json")
    failures=[row for new,audit,row in planned if new is None]
    if failures:
        write_jsonl(root/"migration.jsonl",[row for new,audit,row in planned])
        write_jsonl(root/"teacher_audit.jsonl",[audit for new,audit,row in planned])
        write_json(root/"FAILED.json",{"migration_complete":False,"requested":len(old_records),"failures":failures})
        raise RuntimeError(f"{len(failures)} fixed-input migrations failed; parent untouched")
    jobs=[]; source_hashes={}; records=[]; migrations=[]; teachers=[]
    for old,(new,audit,row) in zip(old_records,planned):
        source_dir=old["source_render_dir"]
        for name in ("rgb.png","instance_mask.png","render.json"):
            relative=str(Path(source_dir)/name)
            source_hashes[relative]=_copy_verified(parent,root,relative)
        target_dir=source_dir if new["mode"]=="no_op" else f"samples/{new['sample_id']}/target"
        new.update(source_render_dir=source_dir,target_render_dir=target_dir)
        legacy_dir=f"legacy_targets/{new['sample_id']}" if row["target_changed"] else target_dir
        for name in ("rgb.png","instance_mask.png","render.json"):
            _copy_verified(parent,root,str(Path(old["target_render_dir"])/name),str(Path(legacy_dir)/name))
        if row["target_changed"]:
            jobs.append(make_job(scene_from_dict(new["target"]),"target_pose",root/target_dir,new["sample_id"]))
        row.update(legacy_target_render_dir=legacy_dir,target_render_dir=target_dir,
                   target_render_action="render" if row["target_changed"] else "reuse")
        records.append(new); migrations.append(row); teachers.append(audit)
    write_json(root/"source_file_hashes.json",source_hashes)
    print(f"Fixed sources: {len(source_hashes)//3}; samples: {len(records)}; changed target renders: {len(jobs)}",flush=True)
    render_batch(jobs,root/"work")
    manifest=[]; source_reports={}
    for new in records:
        source,target=scene_from_dict(new["source"]),scene_from_dict(new["target"])
        directory=new["source_render_dir"]
        if directory not in source_reports:
            source_reports[directory]=validate_render(source,"source_pose",root/directory)
        src=source_reports[directory]; dst=validate_render(target,"target_pose",root/new["target_render_dir"])
        if not src["valid"] or not dst["valid"]:
            write_json(root/"FAILED.json",{"sample_id":new["sample_id"],"source":src,"target":dst})
            raise RuntimeError("migration render validation failed")
        new["render_validation"]={"source":src,"target":dst}
        new["objects"]=_object_annotations(new,src,dst)
        new["file_hashes"]={str(Path(new[role+"_render_dir"])/name):file_hash(root/new[role+"_render_dir"]/name)
                           for role in ("source","target") for name in ("rgb.png","instance_mask.png","render.json")}
        path=f"samples/{new['sample_id']}/annotation.json"
        write_json(root/path,new)
        manifest.append({**{k:new[k] for k in ("sample_id","scene_family_id","split","mode","role")},
                         "annotation":path,"annotation_sha256":file_hash(root/path)})
    write_jsonl(root/"manifest.jsonl",manifest)
    write_jsonl(root/"migration.jsonl",migrations)
    write_jsonl(root/"teacher_audit.jsonl",teachers)
    write_jsonl(root/"human_review.jsonl",pending_reviews(records))
    write_jsonl(root/"training_approved.jsonl",[])
    pairs=counterfactual_audit(records)
    write_jsonl(root/"counterfactual_audit.jsonl",pairs)
    partials,partial_summary=partial_quality(old_records,records)
    write_jsonl(root/"partial_quality.jsonl",partials)
    write_json(root/"partial_quality_summary.json",partial_summary)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    provenance=json.loads((parent/"provenance.json").read_text())
    source_render_revision=provenance.pop("render_code_revision",provenance["code_revision"])
    provenance.pop("annotation_correction",None)  # Retained in immutable parent_provenance.json.
    provenance.update(spec_version="1.3",instruction_contract="row_edges_v1",**policy_metadata(BALANCED),
                      source_render_revision=source_render_revision,changed_target_render_revision=revision,
                      parent_dataset_manifest_sha256=file_hash(parent/"manifest.jsonl"),teacher_code_revision=revision,
                      code_revision=revision,rendered_views=len(jobs),source_views_rendered=0,
                      requested_sample_count=len(old_records),sample_count=len(records),
                      legacy_generation_history={"file":"generation.json","teacher_policy":LEGACY},
                      parent_provenance_file="parent_provenance.json",source_directory=str(parent))
    write_json(root/"provenance.json",provenance)
    summary={"requested_samples":len(old_records),"successful_samples":len(records),"failed_samples":len(failures),
             "target_changed":len(jobs),"target_reused":len(records)-len(jobs),"source_rendered":0,
             "text_only":sum(r["change_kind"]=="text_only" for r in migrations),
             "unchanged":sum(r["change_kind"]=="unchanged" for r in migrations),
             "source_files_verified":len(source_hashes),"counterfactual_pairs":len(pairs),
             "diagnostic_counterfactual_pairs":sum(r["diagnostic_pair"] for r in pairs),
             "human_review_complete":False,"training_ready":False}
    write_json(root/"migration_summary.json",summary)
    if tree_hashes(parent)!=snapshot: raise RuntimeError("parent dataset was modified")
    from .qc import write_qc
    write_qc(root,records,json.loads((root/"generation.json").read_text()))
    audit=audit_dataset(root)
    if not audit["valid"]: raise RuntimeError(f"migration audit failed: {audit['errors']}")
    return summary
