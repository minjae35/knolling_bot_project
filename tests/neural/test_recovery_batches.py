import pytest

from knolling_model.arrangement.batches import stage_batches
from knolling_model.arrangement.data import RecoveryDataset


def test_perception_and_joint_replay_cover_unique_sources(s0_dataset):
    dataset = RecoveryDataset(s0_dataset)
    # Record-only collator; no token download or optimization in this unit test.
    collate = lambda samples: [s['metadata']['record'] for s in samples]
    batches = list(stage_batches(dataset, collate, 'perception', seed=12, epoch=0))
    families = [r['scene_family_id'] for batch in batches for r in batch.batch]
    assert len(families) == len(set(families)) == 25
    joint = list(stage_batches(dataset, collate, 'joint', seed=12, epoch=0))
    replay = [r['scene_family_id'] for batch in joint for r in batch.replay]
    samples = [r['sample_id'] for batch in joint for r in batch.batch]
    assert set(replay) == set(families)
    assert set(samples) == {r['sample_id'] for r in dataset.records}
    assert len(set(replay[:25])) == 25  # No duplicate until a complete source pass.
    with pytest.raises(ValueError, match='stage'):
        list(stage_batches(dataset, collate, 'unknown', seed=12, epoch=0))
