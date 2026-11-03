from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from knolling_model.coordinates import to_meters, to_normalized, to_pixels, angle_delta
from knolling_model.data import supervision, Collator, FamilySampler, S0Dataset


class TokenizerFixture:
    def __call__(self, texts, **kwargs):
        assert kwargs['truncation'] is False
        n = max(len(t.split()) for t in texts) + 2
        return {'input_ids': torch.ones(len(texts), n, dtype=torch.long),
                'attention_mask': torch.ones(len(texts), n, dtype=torch.long)}


def test_coordinate_units_pixels_and_gradients():
    uv = torch.tensor([[0., 0.], [1., 1.], [.5, .5]], requires_grad=True)
    xy = to_meters(uv)
    assert torch.allclose(xy, torch.tensor([[0., 0.], [.6, .4], [.3, .2]]))
    assert torch.allclose(to_normalized(xy), uv)
    assert torch.allclose(to_pixels(uv), torch.tensor([[-.5, 383.5], [575.5, -.5], [287.5, 191.5]]))
    xy.sum().backward()
    assert torch.allclose(uv.grad, torch.tensor([[.6, .4]]).expand(3, 2))
    assert torch.allclose(angle_delta(torch.deg2rad(torch.tensor(179.)), torch.deg2rad(torch.tensor(1.))).abs(), torch.deg2rad(torch.tensor(2.)), atol=1e-6)


def test_selection_integrity_and_inputs(s0_dataset):
    ds = s0_dataset
    assert len(ds) == 32
    assert {r['split'] for r in ds.records} == {'train'}
    assert all(2 <= len(r['source']['objects']) <= 4 for r in ds.records)
    batch = Collator(TokenizerFixture())([ds[0], ds[1]])
    assert set(batch.inputs) == {'images', 'input_ids', 'attention_mask'}
    assert batch.inputs['images'].shape == (2, 3, 384, 576)
    assert batch.inputs['images'].min() >= 0 and batch.inputs['images'].max() <= 1
    assert batch.inputs['attention_mask'].dtype == torch.bool
    assert len(batch.supervision) == 2
    assert set(batch.supervision[0]) == {'source_xy','source_angle','target_xy','target_angle','size','color','shape'}


def test_target_join_is_by_identity_and_gt_cannot_change_inputs(s0_dataset):
    ds = s0_dataset
    before = ds[0]
    record = deepcopy(ds.records[0])
    ids, labels = supervision(record)
    record['target']['objects'].reverse()
    record['source']['objects'].reverse()
    new_ids, reordered = supervision(record)
    assert ids == new_ids
    for key in labels:
        assert torch.equal(labels[key], reordered[key])
    for o in record['target']['objects']:
        o['target_pose']['x_m'] += .02
    record['instruction']['selected_ids'] = []
    record['rows'] = [['hidden_gt']]
    ds.records[0] = record
    after = ds[0]
    assert torch.equal(before['image'], after['image']) and before['text'] == after['text']
    assert not torch.equal(before['supervision']['target_xy'], after['supervision']['target_xy'])
    broken = deepcopy(record)
    broken['target']['objects'][0]['object_id'] = 'missing'
    with pytest.raises(ValueError, match='identity'):
        supervision(broken)


def test_adapter_rejects_changed_manifest_and_held_out_ids(s0_dataset):
    selection = deepcopy(s0_dataset.selection)
    selection['manifest_sha256'] = 'bad'
    with pytest.raises(ValueError, match='manifest hash'):
        S0Dataset(s0_dataset.root, selection)
    selection = deepcopy(s0_dataset.selection)
    selection['sample_ids'][0] = 'scene_0003_primary'
    with pytest.raises(ValueError, match='eligible'):
        S0Dataset(s0_dataset.root, selection)


def test_no_silent_truncation(s0_dataset):
    sample = s0_dataset[0]
    sample['text'] = 'word ' * 511
    with pytest.raises(ValueError, match='truncation'):
        Collator(TokenizerFixture())([sample])


def test_family_sampler_balanced_cyclic_and_resumable(s0_dataset):
    sampler = FamilySampler(s0_dataset.records, 123)
    seen = set()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        visits = list(sampler)
        assert len(visits) == len(sampler) == 25
        assert len({s0_dataset.records[i]['scene_family_id'] for i in visits}) == 25
        assert list(sampler) == visits
        seen.update(visits)
    assert seen == set(range(32))
    resumed = FamilySampler(s0_dataset.records, 123)
    resumed.set_epoch(1)
    assert list(resumed) == list(sampler)
