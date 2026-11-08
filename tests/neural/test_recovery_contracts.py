from dataclasses import asdict

import pytest
import torch

from knolling_model.recovery.contracts import detector_labels, oracle_objects, pack_sources
from knolling_model.recovery.data import (assert_family_disjoint, balanced_instruction_indices,
    counterfactual_pair, recovery_supervision, unique_source_indices)


def test_oracle_labels_units_and_source_only_boxes(s0_dataset):
    _, truth = recovery_supervision(s0_dataset.records[0])
    objects = oracle_objects([truth, truth])
    assert objects.features().shape == (2, len(truth['size']), 12)
    labels = detector_labels([truth])[0]
    torch.testing.assert_close(labels['boxes'][:, 1], 1-truth['source_xy'][:, 1]/.4)
    changed = {**truth, 'target_xy': truth['target_xy']+100, 'target_angle': truth['target_angle']+1}
    torch.testing.assert_close(labels['boxes'], detector_labels([changed])[0]['boxes'])
    rotated = {**truth, 'source_angle': torch.full_like(truth['source_angle'], torch.pi/2)}
    torch.testing.assert_close(detector_labels([rotated])[0]['boxes'][:, 2], truth['size'][:, 1]/.6)


def test_gather_preserves_identity_gradient_and_empty_scenes():
    uv = torch.rand(2, 4, 2, requires_grad=True)
    source = dict(uv=uv, size=torch.ones_like(uv)*.05, angle=torch.zeros(2, 4),
                  color_logits=torch.zeros(2, 4, 3), shape_logits=torch.zeros(2, 4, 3))
    obj = pack_sources(source, [torch.tensor([1, 3]), torch.tensor([], dtype=torch.long)])
    assert obj.query_index.tolist() == [[1, 3], [-1, -1]]
    assert not obj.valid[1].any()
    obj.features().sum().backward()
    assert uv.grad[0, [1, 3]].abs().sum() > 0
    assert uv.grad[0, [0, 2]].abs().sum() == 0
    assert uv.grad[1].abs().sum() == 0
    with pytest.raises(ValueError, match='duplicate'):
        pack_sources(source, [torch.tensor([1, 1]), torch.tensor([], dtype=torch.long)])


def test_read_only_s0_exposure_and_counterfactuals(s0_dataset):
    from knolling_dataset.schema import scene_from_dict, instruction_from_dict
    records = s0_dataset.records
    assert len(unique_source_indices(records)) == 25
    order = balanced_instruction_indices(records, 1203, 0)
    assert set(order) == set(range(32))
    assert order == balanced_instruction_indices(records, 1203, 0)
    assert order != balanced_instruction_indices(records, 1203, 1)
    counts = {'full': 0, 'partial': 0, 'no_op': 0}
    for i in order:
        r = records[i]
        ins = r['instruction']
        key = 'no_op' if r.get('mode') == 'no_op' or ins['keep_all'] else ('partial' if ins['locked_ids'] else 'full')
        counts[key] += 1
    assert len(set(counts.values())) == 1
    r = next(r for r in records if not r['instruction']['keep_all'])
    scene, ins = scene_from_dict(r['source']), instruction_from_dict(r['instruction'])
    alternative = 'vertical' if ins.axis_alignment == 'horizontal' else 'horizontal'
    a, b = counterfactual_pair(scene, ins, 'axis_alignment', alternative)
    differences = {k for k in asdict(a) if asdict(a)[k] != asdict(b)[k]}
    assert differences == {'text', 'axis_alignment'}
    assert 'vertical' in (a.text if a.axis_alignment == 'vertical' else b.text)
    assert_family_disjoint({'train': records, 'test': []})
    with pytest.raises(ValueError, match='leaked'):
        assert_family_disjoint({'train': records, 'test': [records[0]]})
