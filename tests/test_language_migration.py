from copy import deepcopy
import json
import os
from pathlib import Path
import pytest
from knolling_dataset.builder import prepare_annotations,file_hash,write_json,write_split_manifests
from knolling_dataset.migration import read_records,tree_hashes,write_jsonl
from knolling_dataset.language_migration import convert_record,migrate_english
from knolling_dataset.english import audit_english_records
from knolling_dataset.splits import audit_dataset
from knolling_dataset.policies import LEGACY


def test_conversion_only_changes_language_and_text():
    for old in prepare_annotations(10,policy=LEGACY)[1]:
        snapshot=deepcopy(old); new=convert_record(old)
        assert old==snapshot
        assert audit_english_records([new])["valid"]
        restored=deepcopy(new); restored["instruction"]=old["instruction"]
        assert restored==old
        assert {k:v for k,v in new["instruction"].items() if k not in ("text","language")}=={
            k:v for k,v in old["instruction"].items() if k not in ("text","language")}


def assert_english_qc(root,count):
    from test_qc_v13 import ReportParser
    qc=root/"qc"; parser=ReportParser(); text=(qc/"index.html").read_text(); parser.feed(text)
    assert len(parser.cards)==len(set(parser.cards))==count
    assert len(parser.raw)==2*count and len(parser.images)==3*count
    assert text.count('lang="en"')==count+1
    assert not any('\uac00'<=c<='\ud7a3' for c in text)
    for p in parser.links+parser.images+parser.raw: assert (qc/p).is_file(),p


@pytest.mark.skipif(os.environ.get("RUN_LANGUAGE_MIGRATION_TESTS")!="1" or not os.environ.get("PILOT_V13_DATASET"),
                    reason="enable text-only migration acceptance against the archived v1.3 dataset")
def test_real_language_migration_never_renders_and_preserves_everything(tmp_path,monkeypatch):
    parent=Path(os.environ["PILOT_V13_DATASET"]).resolve()
    snapshot=tree_hashes(parent)
    def forbidden(*a,**k): raise AssertionError("language migration must not render or regenerate scenes")
    for name in ("rendering.render_batch","builder.render_batch","migration.render_batch",
                 "builder.prepare_annotations","builder.solve","scenes.generate_scene"):
        monkeypatch.setattr("knolling_dataset."+name,forbidden)
    root=tmp_path/"english"
    summary=migrate_english(parent,root)
    assert summary["active_language_counts"]=={"en":130}
    assert summary["original_language_counts"]=={"ko":73,"en":57}
    assert summary["semantic_inconsistencies"]==summary["target_views_rendered"]==summary["source_views_rendered"]==0
    assert tree_hashes(parent)==snapshot
    assert_english_qc(root,130)
    _,old=read_records(parent); manifest,new=read_records(root)
    for a,b in zip(old,new):
        assert b==convert_record(a)
        for p,sha in a["file_hashes"].items():
            assert file_hash(root/p)==sha==file_hash(parent/p)
            assert (root/p).stat().st_ino!=(parent/p).stat().st_ino
    # Reject a non-English sample even if its hash and all split indexes are repaired,
    # and even if the caller requests a legacy audit of this active artifact.
    new[0]["instruction"]["language"]="ko"
    write_json(root/manifest[0]["annotation"],new[0])
    manifest[0]["annotation_sha256"]=file_hash(root/manifest[0]["annotation"])
    write_jsonl(root/"manifest.jsonl",manifest); write_split_manifests(root,manifest)
    bad=audit_dataset(root,verify_renders=False,read_only=True,require_english=False)
    assert not bad["valid"] and any("non_english_instruction" in e for e in bad["errors"])
    assert tree_hashes(parent)==snapshot


@pytest.mark.skipif(not os.environ.get("PILOT_EN_DATASET"),reason="set PILOT_EN_DATASET for delivered English acceptance")
def test_actual_english_pilot():
    root=Path(os.environ["PILOT_EN_DATASET"])
    result=audit_dataset(root,read_only=True)
    assert result["valid"],result
    assert result["samples"]==130 and result["scene_families"]==100
    assert result["language_migration_audit"]["unchanged_render_files"]==630
    assert_english_qc(root,130)
    for split in ("train","validation","development","test"):
        rows=[json.loads(l) for l in (root/f"{split}.jsonl").read_text().splitlines()]
        assert rows
        assert audit_english_records([json.loads((root/r["annotation"]).read_text()) for r in rows])["valid"]
    assert (root/"training_approved.jsonl").read_text()==""


def test_language_migration_refuses_overwrite_and_parent(tmp_path):
    parent=tmp_path/"parent"; parent.mkdir()
    for output in (parent,parent/"nested",tmp_path):
        with pytest.raises(ValueError,match="separate"): migrate_english(parent,output)
    other=tmp_path/"other"; other.mkdir()
    with pytest.raises(ValueError,match="new directory"): migrate_english(parent,other)
