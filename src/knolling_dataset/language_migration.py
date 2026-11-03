"""Text-only v1.3 migration. The parent is immutable and no renderer is called."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from .builder import file_hash,write_json,write_split_manifests
from .migration import read_records,tree_hashes,write_jsonl,pending_reviews
from .schema import scene_from_dict,instruction_from_dict,digest
from .english import english_instruction,semantic_fields,LANGUAGE_POLICY,INSTRUCTION_CONTRACT


def convert_record(old):
    new=deepcopy(old)
    ins=english_instruction(scene_from_dict(old["source"]),instruction_from_dict(old["instruction"]))
    # Preserve JSON field representations and every key except the surface form.
    new["instruction"]=dict(old["instruction"],language=ins.language,text=ins.text)
    return new


def comparison(old,new):
    a,b=instruction_from_dict(old["instruction"]),instruction_from_dict(new["instruction"])
    return {"sample_id":old["sample_id"],"split":old["split"],"old_language":a.language,"new_language":b.language,
            "old_text_hash":digest(a.text),"new_text_hash":digest(b.text),"text_changed":a.text!=b.text,
            "old_semantics_hash":digest(semantic_fields(a)),"new_semantics_hash":digest(semantic_fields(b)),
            "semantics_preserved":semantic_fields(a)==semantic_fields(b),"english_word_count":len(b.text.split())}


def audit_language_migration(root,records,provenance):
    root=Path(root); errors=[]
    if provenance.get("language_policy")!=LANGUAGE_POLICY or provenance.get("instruction_contract")!=INSTRUCTION_CONTRACT:
        errors.append("english_policy_metadata")
    path=root/"language_parent_manifest.jsonl"
    if file_hash(path)!=provenance["language_parent_dataset_manifest_sha256"]:
        errors.append("language_parent_manifest_hash")
    parent={r["sample_id"]:r for r in (json.loads(l) for l in path.read_text().splitlines())}
    rows=[json.loads(l) for l in (root/"language_migration.jsonl").read_text().splitlines()]
    changes={r["sample_id"]:r for r in rows}
    ids={r["sample_id"] for r in records}
    if ids!=set(parent) or ids!=set(changes) or len(rows)!=len(records):
        errors.append("language_migration_coverage")
    image_files=set()
    for new in records:
        sid=new["sample_id"]
        if sid not in parent or sid not in changes: continue
        path=root/"language_parent_annotations"/f"{sid}.json"
        if file_hash(path)!=parent[sid]["annotation_sha256"]: errors.append(sid+":language_parent_annotation_hash")
        old=json.loads(path.read_text())
        expected=convert_record(old)
        if digest(expected)!=digest(new): errors.append(sid+":non_language_change_or_wrong_translation")
        if digest(changes[sid])!=digest(comparison(old,new)): errors.append(sid+":language_migration_scores")
        for relative,sha in old["file_hashes"].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path)!=sha:
                errors.append(sid+":changed_render_file")
            image_files.add(relative)
    return {"valid":not errors,"errors":errors,"samples":len(records),"unchanged_render_files":len(image_files),
            "target_poses_changed":0 if not errors else None,"human_review_complete":False}


def migrate_english(parent,output):
    from .splits import audit_dataset
    parent,root=Path(parent).resolve(),Path(output).resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("English output must be separate from its parent")
    if root.exists():
        raise ValueError("English output must be a new directory")
    provenance=json.loads((parent/"provenance.json").read_text())
    if provenance.get("teacher_policy")!="balanced_rows_v1.3" or provenance.get("language_policy")==LANGUAGE_POLICY:
        raise ValueError("English migration requires an archived, non-English-policy v1.3 dataset")
    before=tree_hashes(parent)
    original_audit=audit_dataset(parent,read_only=True,require_english=False)
    if not original_audit["valid"]: raise ValueError("parent dataset audit failed")
    manifest,original=read_records(parent)
    records=[convert_record(r) for r in original]  # Fail before copying if grounding cannot be preserved.
    if len({r["scene_family_id"] for r in records})>100:
        raise ValueError("English migration is limited to 100 source families")
    shutil.copytree(parent,root,copy_function=shutil.copy2)
    if tree_hashes(root)!=before: raise RuntimeError("parent copy did not preserve all files")
    for name in ("manifest.jsonl","provenance.json"):
        shutil.copy2(parent/name,root/("language_parent_"+name))
    write_json(root/"language_parent_snapshot.json",before)
    archive=root/"language_parent_annotations"; archive.mkdir()
    new_manifest=[]
    for entry,new in zip(manifest,records):
        shutil.copy2(parent/entry["annotation"],archive/f"{entry['sample_id']}.json")
        write_json(root/entry["annotation"],new)
        new_manifest.append(dict(entry,annotation_sha256=file_hash(root/entry["annotation"])))
    write_jsonl(root/"manifest.jsonl",new_manifest)
    write_split_manifests(root,new_manifest)
    rows=[comparison(a,b) for a,b in zip(original,records)]
    write_jsonl(root/"language_migration.jsonl",rows)
    write_jsonl(root/"human_review.jsonl",pending_reviews(records))
    write_jsonl(root/"training_approved.jsonl",[])
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    provenance.update(language_policy=LANGUAGE_POLICY,instruction_contract=INSTRUCTION_CONTRACT,
                      supported_languages=["en"],dataset_role="active_english_v1",code_revision=revision,
                      language_code_revision=revision,language_parent_dataset_manifest_sha256=file_hash(parent/"manifest.jsonl"),
                      language_parent_directory=str(parent),rendered_views=0,source_views_rendered=0,
                      archival_annotations_excluded_from_training=["parent_annotations/","language_parent_annotations/"],
                      active_manifest="manifest.jsonl",historical_geometry_provenance="language_parent_provenance.json")
    write_json(root/"provenance.json",provenance)
    summary={"samples":len(records),"scene_families":len({r["scene_family_id"] for r in records}),
             "original_language_counts":dict(Counter(r["instruction"]["language"] for r in original)),
             "active_language_counts":dict(Counter(r["instruction"]["language"] for r in records)),
             "split_sample_counts":dict(Counter(r["split"] for r in records)),
             "text_changed":sum(r["text_changed"] for r in rows),"semantic_inconsistencies":sum(not r["semantics_preserved"] for r in rows),
             "source_views_rendered":0,"target_views_rendered":0,"target_poses_changed":0,
             "english_word_count_range":[min(r["english_word_count"] for r in rows),max(r["english_word_count"] for r in rows)],
             "human_review_complete":False,"training_ready":False}
    write_json(root/"language_migration_summary.json",summary)
    from .qc_english import write_english_qc
    write_english_qc(root,records,json.loads((root/"generation.json").read_text()))
    result=audit_dataset(root)
    if not result["valid"]: raise RuntimeError(f"English dataset audit failed: {result['errors']}")
    if tree_hashes(parent)!=before: raise RuntimeError("original v1.3 dataset changed")
    return summary
