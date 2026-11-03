from copy import deepcopy
import pytest
from knolling_dataset.builder import prepare_annotations
from knolling_dataset.splits import audit_records,_safe_path


@pytest.fixture
def records():
    return prepare_annotations(10,1202)[1]


def test_clean_splits(records):
    report = audit_records(records)
    assert report["valid"],report
    assert report["scene_families"] == 10


def test_family_and_asset_leakage(records):
    changed = deepcopy(records)
    r = next(r for r in changed if r["role"] == "counterfactual")
    r["split"] = "test" if r["split"] != "test" else "train"
    r["source"]["split"] = r["target"]["split"] = r["split"]
    errors = audit_records(changed)["errors"]
    assert any("scene_family_leakage" in e for e in errors)
    assert any("asset_family_leakage" in e for e in errors)
    assert any("geometry_alias_leakage" in e for e in errors)


def test_noop_relation_and_identity_corruption(records):
    changed = deepcopy(records)
    r = next(r for r in changed if r["mode"] == "no_op")
    r["relations"] = {"bad":"label"}
    r["target"]["objects"][0]["color"] = "green" if r["target"]["objects"][0]["color"] != "green" else "blue"
    errors = audit_records(changed)["errors"]
    assert any("noop_relations" in e for e in errors)
    assert any("source_target_identity" in e for e in errors)


def test_paths_cannot_escape_dataset(tmp_path):
    with pytest.raises(ValueError):
        _safe_path(tmp_path,"../outside.json")
