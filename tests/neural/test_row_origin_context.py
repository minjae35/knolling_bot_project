"""The 112 same-row scenes must show the other row's height in the origin input."""
from pathlib import Path

from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.origin_audit import cross_row_height_reaches_origin
from knolling_model.experiments.row_layout import NeuralRowPlacement

DATA = Path(__file__).resolve().parents[2]/'artifacts/generalization_20000/dataset'


def _multi_row(record):
    instruction = record['instruction']
    count = len(record['source']['objects'])
    return (record['mode'] == 'full' and not instruction['keep_all'] and not instruction['locked_ids']
            and count in (2, 4, 6, 8) and len(record['supervision_v3']['rows']) >= 2
            and instruction['group_by'] == 'none' and instruction['region'] == 'full'
            and instruction['axis_alignment'] == 'horizontal')


def test_known_height_conflicts_reach_the_second_row_input():
    records = [record for split in ('train', 'validation')
               for record in ExpandedDataset(DATA, split).records if _multi_row(record)]
    report = cross_row_height_reaches_origin(NeuralRowPlacement(), records)
    assert report['reached'], report
