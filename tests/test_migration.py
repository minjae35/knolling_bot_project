from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import pytest
from knolling_dataset.builder import prepare_annotations,build_dataset,file_hash,write_json
from knolling_dataset.migration import migrate_record,migrate_dataset,tree_hashes,initialize_balanced_sidecars
from knolling_dataset.schema import scene_from_dict,digest
from knolling_dataset.quality import target_hash,target_changed,partial_quality,counterfactual_audit
from knolling_dataset.policies import LEGACY,BALANCED,policy_metadata
from knolling_dataset.audit_v13 import check_v13


def test_fixed_inputs_targets_and_quality_no_resampling(monkeypatch):
    old=prepare_annotations(10,1202,policy=LEGACY)[1]
    monkeypatch.setattr("knolling_dataset.builder.prepare_annotations",lambda *a,**k:(_ for _ in ()).throw(AssertionError("resampling forbidden")))
    snapshot=digest(old)
    new=[]
    for record in old:
        result,audit,migration=migrate_record(record)
        assert result is not None
        assert result["source"]==record["source"]
        assert all(result[k]==record[k] for k in ("sample_id","split","scene_family_id","role","mode"))
        assert audit["teacher_policy"]==BALANCED
        assert migration["target_changed"]==target_changed(scene_from_dict(record["target"]),scene_from_dict(result["target"]))
        if record["mode"]=="partial":
            assert migration["new_metrics"]["J_move"]<=migration["old_metrics"]["J_move"]+1e-12
        new.append(result)
    assert digest(old)==snapshot
    partials,summary=partial_quality(old,new)
    assert len(partials)==3 and not summary["distribution_modified"]
    assert all(p["diagnostic_pair"] for p in counterfactual_audit(new))


def test_pi_equivalent_targets_reuse_images():
    records=prepare_annotations(10)[1]
    scene=scene_from_dict(records[0]["target"])
    import math
    rotated=replace(scene,objects=tuple(replace(o,target_pose=replace(o.target_pose,theta_rad=o.target_pose.theta_rad+math.pi)) for o in scene.objects))
    assert not target_changed(scene,rotated)
    assert target_hash(scene)==target_hash(rotated)


def test_policy_sidecar_tampering(tmp_path):
    records=prepare_annotations(10,1202,policy=BALANCED)[1]
    initialize_balanced_sidecars(tmp_path,records)
    provenance=policy_metadata(BALANCED)
    assert check_v13(tmp_path,records,provenance,invariance=False)["valid"]
    p=tmp_path/"teacher_audit.jsonl"
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    rows[0]["teacher_policy"]=LEGACY
    p.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    assert "mixed_teacher_policy" in check_v13(tmp_path,records,provenance,invariance=False)["errors"]


@pytest.mark.skipif(os.environ.get("RUN_MIGRATION_TESTS")!="1",reason="enable real rendered migration integration")
def test_real_migration_preserves_parent_and_only_renders_changes(tmp_path,monkeypatch):
    parent=tmp_path/"legacy"; output=tmp_path/"new"
    build_dataset(parent,10,1203,policy=LEGACY)
    before=tree_hashes(parent)
    from knolling_dataset import migration
    actual=migration.render_batch; seen=[]
    def spy(jobs,path):
        seen.extend(jobs); return actual(jobs,path)
    monkeypatch.setattr(migration,"render_batch",spy)
    summary=migrate_dataset(parent,output,registered_legacy_hash=file_hash(parent/"manifest.jsonl"))
    assert summary["successful_samples"]==13 and summary["failed_samples"]==0
    assert len(seen)==summary["target_changed"] and all(j["pose_field"]=="target_pose" for j in seen)
    assert before==tree_hashes(parent)
    assert (output/"AUTOMATED_VALIDATION.json").exists() and not (output/"COMPLETE.json").exists()
    assert (output/"training_approved.jsonl").read_text()==""
    for relative in json.loads((output/"source_file_hashes.json").read_text()):
        assert file_hash(parent/relative)==file_hash(output/relative)
        assert (parent/relative).stat().st_ino!=(output/relative).stat().st_ino
