from collections import Counter
from knolling_dataset.builder import prepare_annotations
from knolling_dataset.schema import scene_from_dict,instruction_from_dict,digest
from knolling_dataset.validator import validate_layout
import os
import json
import pytest
from knolling_dataset.builder import build_dataset,file_hash


def test_annotations_deterministic_complete_and_balanced():
    registry,records,stats = prepare_annotations(100,1202)
    assert digest(records) == digest(prepare_annotations(100,1202)[1])
    assert len(records) == 130
    assert stats["primary_mode_counts"] == {"full":50,"partial":30,"no_op":20}
    assert stats["no_op_types"] == {"repeat":14,"keep_all":6}
    families = {}
    for r in records:
        assert validate_layout(scene_from_dict(r["source"]),scene_from_dict(r["target"]),instruction_from_dict(r["instruction"]),r["mode"]).valid
        families.setdefault(r["scene_family_id"],[]).append(r)
    assert len(families) == 100
    assert sum(len(v) == 2 for v in families.values()) == 30
    for members in families.values():
        if len(members) == 2:
            a,b = members
            assert a["source"] == b["source"]
            assert a["instruction"]["text"] != b["instruction"]["text"]
            assert a["target"] != b["target"]


@pytest.mark.skipif(os.environ.get("RUN_BUILDER_TESTS") != "1",reason="set RUN_BUILDER_TESTS=1 for 10-scene rendered integration")
@pytest.mark.parametrize("policy",["legacy_v1_2","balanced_rows_v1.3"])
def test_complete_builder_with_actual_renders(tmp_path,policy):
    root = tmp_path/"dataset"
    records,stats = build_dataset(root,10,1203,policy=policy)
    assert len(records) == 13
    rows = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 13
    for row in rows:
        assert file_hash(root/row["annotation"]) == row["annotation_sha256"]
    assert (root/"qc/index.html").exists() and (root/"qc/contact_sheet.png").exists()
    assert json.loads((root/"qc/summary.json").read_text())["all_render_checks_valid"]
    from knolling_dataset.splits import audit_dataset
    audit = audit_dataset(root,require_english=policy!="legacy_v1_2")
    assert audit["valid"],audit
    marker="COMPLETE.json" if policy=="legacy_v1_2" else "AUTOMATED_VALIDATION.json"
    assert (root/marker).exists()
    if policy=="balanced_rows_v1.3":
        assert not (root/"COMPLETE.json").exists()
        assert audit["policy_audit"]["invariance_counts"]["policy_reproduction"]==13
    # A dropped manifest entry is not a smaller, silently valid dataset.
    manifest_path = root/"manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_path.read_text().splitlines()[1:])+"\n")
    broken = audit_dataset(root,verify_renders=False,require_english=policy!="legacy_v1_2")
    assert not broken["valid"] and "manifest_count" in broken["errors"]
    assert not (root/marker).exists()
