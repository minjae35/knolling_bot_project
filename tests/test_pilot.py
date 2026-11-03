"""Opt-in acceptance test of the actual on-disk 100-scene deliverable."""
import json
import os
from pathlib import Path
import pytest
from knolling_dataset.builder import prepare_annotations,file_hash
from knolling_dataset.schema import digest
from knolling_dataset.splits import audit_dataset


@pytest.mark.skipif(not os.environ.get("PILOT_DATASET"),reason="set PILOT_DATASET to an existing completed 100-scene pilot")
def test_actual_pilot_acceptance_and_annotation_reproduction():
    root = Path(os.environ["PILOT_DATASET"])
    provenance = json.loads((root/"provenance.json").read_text())
    assert provenance["scene_count"] == 100 and provenance["sample_count"] == 130
    audit = audit_dataset(root,verify_renders=True,read_only=True)
    assert audit["valid"],audit
    assert audit["renders_revalidated"]
    manifest = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    existing = {m["sample_id"]:json.loads((root/m["annotation"]).read_text()) for m in manifest}
    # Rebuild only annotations, NOT another rendered dataset. Compare all labels.
    _,regenerated,stats = prepare_annotations(100,provenance["seed"],policy="legacy_v1_2")
    assert stats["primary_mode_counts"] == {"full":50,"partial":30,"no_op":20}
    assert stats["no_op_types"] == {"repeat":14,"keep_all":6}
    assert len({r["scene_family_id"] for r in regenerated}) == 100
    for expected in regenerated:
        actual = existing[expected["sample_id"]]
        assert digest(expected) == digest({k:actual[k] for k in expected})
        if expected["mode"] == "no_op":
            assert expected["rows"] == [] and expected["relations"] == {}
            for obj in actual["target"]["objects"]:
                assert obj["source_pose"] == obj["target_pose"]
                assert obj["locked"] is True and obj["move_label"] is False
    complete = json.loads((root/"COMPLETE.json").read_text())
    assert complete["manifest_sha256"] == file_hash(root/"manifest.jsonl")
    assert complete["human_review"] == "pending"
