import csv
import json
import os
from pathlib import Path
from html.parser import HTMLParser
import pytest
from knolling_dataset.splits import audit_dataset


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.cards=[]; self.links=[]; self.raw=[]; self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=="article": self.cards.append(a["id"])
        if tag=="a": self.links.append(a["href"])
        if tag=="img":
            self.images.append(a["src"])
            if "data-raw" in a: self.raw.append(a["data-raw"])


def assert_complete_report(root,count):
    qc=Path(root)/"qc"; parser=ReportParser()
    parser.feed((qc/"index.html").read_text())
    assert len(parser.cards)==len(set(parser.cards))==count
    assert len(parser.raw)==3*count and len(parser.images)==4*count
    for relative in parser.links+parser.images+parser.raw:
        assert (qc/relative).is_file(),relative
    with (qc/"target_comparison.csv").open() as f: rows=list(csv.DictReader(f))
    assert {r["sample_id"] for r in rows}==set(parser.cards)
    with (qc/"object_movements.csv").open() as f: movements=list(csv.DictReader(f))
    assert {r["sample_id"] for r in movements}==set(parser.cards)
    for row in movements:
        if row["locked"]=="True":
            assert float(row["translation_mm"])==float(row["rotation_deg"])==0
    reviews=[json.loads(line) for line in (Path(root)/"human_review.jsonl").read_text().splitlines()]
    assert len(reviews)==count and all(r["status"]=="pending" and r["acceptable"] is None for r in reviews)


@pytest.mark.skipif(not os.environ.get("PILOT_V13_DATASET"),reason="set PILOT_V13_DATASET for full migrated deliverable acceptance")
def test_actual_v13_all_samples_and_qc():
    root=Path(os.environ["PILOT_V13_DATASET"])
    result=audit_dataset(root,read_only=True,require_english=False)
    assert result["valid"],result
    assert result["samples"]==130 and result["scene_families"]==100
    assert set(result["policy_audit"]["invariance_counts"].values())=={130}
    assert result["policy_audit"]["diagnostic_counterfactual_pairs"]==30
    assert not result["policy_audit"]["human_review_complete"]
    assert_complete_report(root,130)
