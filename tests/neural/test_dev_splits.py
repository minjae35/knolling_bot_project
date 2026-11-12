"""The v2 seal stays off every slice that v1 already used."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments' / 'end_to_end'))

from knolling_model.experiments.expanded_data import ExpandedDataset
import dev_splits


def _record(family):
    return dict(
        scene_family_id=family, mode='full',
        instruction=dict(keep_all=False, locked_ids=[], selected_ids=['a', 'b']),
        source=dict(objects=[{'object_id': 'a'}, {'object_id': 'b'}]))


def test_the_v2_seal_is_disjoint_from_slices_already_used():
    records = [_record(f'f{index:04d}') for index in range(1600)]
    sealed = dev_splits.v2_sealed_families(records)
    _grouped, trainable, reserved, blocked = dev_splits.families(records)
    assert len(sealed) == dev_splits.V2_SEAL_COUNT
    assert set(sealed) <= set(trainable)
    assert not set(sealed) & set(reserved)
    assert not set(sealed) & set(blocked)


def test_the_real_train_seal_misses_validation_and_test():
    root = 'artifacts/generalization_20000/dataset'
    train = ExpandedDataset(root, 'train').records
    validation = {record['scene_family_id'] for record in ExpandedDataset(root, 'validation').records}
    test = {record['scene_family_id'] for record in ExpandedDataset(root, 'test').records}
    sealed = set(dev_splits.v2_sealed_families(train))
    assert len(sealed) == dev_splits.V2_SEAL_COUNT
    assert not sealed & validation
    assert not sealed & test
