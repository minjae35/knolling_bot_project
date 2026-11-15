from dataclasses import replace

import pytest
import torch

from knolling_model.coordinates import angle_delta
from knolling_model.arrangement.contracts import oracle_objects
from knolling_model.arrangement.layout import AutoregressiveLayout, PeriodicBins, layout_loss
from test_recovery_perception import truth


def fixture():
    torch.manual_seed(17)
    model = AutoregressiveLayout(dimension=32, heads=4, layers=1, bins=18)
    objects = oracle_objects([truth(), truth()])
    text = torch.randn(2, 4, 384, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return model, objects, text, mask


def test_periodic_angle_bins_at_boundaries_and_no_mode_average():
    bins = PeriodicBins(18)
    angles = torch.tensor([-torch.pi/2, -torch.pi/2+1e-5, -.001, 0., .001, torch.pi/2-1e-5, torch.pi/2])
    index, offset = bins.encode(angles)
    logits = torch.full((len(angles), 18), -10.)
    logits.scatter_(1, index[:, None], 10.)
    residual = torch.zeros_like(logits)
    residual.scatter_(1, index[:, None], torch.atanh(offset.clamp(-.999999, .999999))[:, None])
    decoded = bins.decode(logits, residual)
    assert angle_delta(decoded, angles).abs().max() < 1e-5
    _, axis_residual = bins.encode(torch.tensor([0., -torch.pi/2]))
    torch.testing.assert_close(axis_residual, torch.zeros(2), atol=1e-6, rtol=0)
    scores = torch.zeros(1, 18)
    scores[0, 0], scores[0, 9] = 10., 9.9
    assert abs(float(bins.decode(scores, torch.zeros_like(scores)))) > 1.4


def test_teacher_forcing_causality_and_free_rollout_prefix_equivalence():
    model, objects, text, mask = fixture()
    xy, angle = objects.xy+.02, objects.angle+.1
    a = model.teacher_forced(objects, text, mask, xy, angle)
    changed_xy, changed_angle = xy.clone(), angle.clone()
    changed_xy[:, 1] += .2
    changed_angle[:, 1] += 1
    b = model.teacher_forced(objects, text, mask, changed_xy, changed_angle)
    torch.testing.assert_close(a['candidate_xy'], b['candidate_xy'])
    changed_xy[:, 0] += .1
    c = model.teacher_forced(objects, text, mask, changed_xy, changed_angle)
    torch.testing.assert_close(a['candidate_xy'][:, 0], c['candidate_xy'][:, 0])
    assert not torch.allclose(a['candidate_xy'][:, 1], c['candidate_xy'][:, 1])
    rollout = model(objects, text, mask)
    replay = model.teacher_forced(objects, text, mask, rollout['xy'], rollout['angle'])
    for key in ('candidate_xy', 'angle_logits', 'move_logits'):
        torch.testing.assert_close(rollout[key], replay[key], atol=1e-6, rtol=1e-5)
    assert rollout['rollout'] == 'free_rollout'
    model.eval()
    with pytest.raises(ValueError, match='evaluation'):
        model.teacher_forced(objects, text, mask, xy, angle)


def test_geometry_text_gradients_padding_and_hard_copy():
    model, objects, text, mask = fixture()
    objects.xy.requires_grad_()
    objects.size.requires_grad_()
    objects.angle.requires_grad_()
    objects.color.requires_grad_()
    output = model(objects, text, mask)
    target = dict(xy=objects.xy.detach()+.01, angle=objects.angle.detach()+.1,
                  move=torch.ones(2, 2, dtype=torch.bool), selected=torch.ones(2, 2, dtype=torch.bool),
                  axis=torch.tensor([0, 1]), region=torch.tensor([0, 2]))
    losses = layout_loss(output, target, model.orientation)
    sum(losses.values()).backward()
    for x in (objects.xy, objects.size, objects.angle, objects.color, text):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert text.grad[~mask].abs().sum() == 0
    # A neural keep decision copies its associated source; no command parser.
    with torch.no_grad():
        model.move.weight.zero_()
        model.move.bias.fill_(-100)
    keep = model(objects, text, mask)
    torch.testing.assert_close(keep['xy'], objects.xy)
    torch.testing.assert_close(keep['angle'], objects.angle)
    torch.testing.assert_close(keep['query_index'], objects.query_index)
    empty = replace(objects, valid=torch.zeros_like(objects.valid), query_index=torch.full_like(objects.query_index, -1))
    result = model(empty, text, mask)
    assert torch.isfinite(result['xy']).all() and not result['valid'].any()
    target.update(move=torch.zeros(2, 2, dtype=torch.bool), axis=torch.tensor([-100, -100]), region=torch.tensor([-100, -100]))
    assert all(torch.isfinite(x) for x in layout_loss(result, target, model.orientation).values())


def test_padded_object_values_do_not_change_valid_predictions():
    model, objects, text, mask = fixture()
    valid = torch.tensor([[True, False], [True, True]])
    objects = replace(objects, valid=valid, query_index=objects.query_index.masked_fill(~valid, -1))
    a = model(objects, text, mask)
    xy = objects.xy.clone()
    xy[0, 1] = 50
    b = model(replace(objects, xy=xy), text, mask)
    torch.testing.assert_close(a['candidate_xy'][valid], b['candidate_xy'][valid])
