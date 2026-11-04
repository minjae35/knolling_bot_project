import math
import pytest
import torch
from knolling_model.coordinates import to_normalized
from knolling_model.losses import (S0Criterion, match_source, position_loss, size_loss,
                                    orientation_loss, objectness_loss)


def test_balanced_objectness_invariant_to_repeated_empty_queries_and_gradient_mass():
    # Increasing capacity alone must not dilute the foreground objective.
    for negatives in (10, 298, 598):
        logits = torch.zeros(2 + negatives, requires_grad=True)
        labels = torch.arange(len(logits)) < 2
        loss = objectness_loss(logits, labels, 'balanced_groups')
        loss.backward()
        assert loss.item() == pytest.approx(math.log(2))
        assert logits.grad[:2].sum().item() == pytest.approx(-.25)
        assert logits.grad[2:].sum().item() == pytest.approx(.25)
        assert (logits.grad[:2] < 0).all() and (logits.grad[2:] > 0).all()


def test_balanced_objectness_all_positive_and_empty_positive_rejection():
    x = torch.zeros(2, requires_grad=True)
    loss = objectness_loss(x, torch.ones(2, dtype=torch.bool), 'balanced_groups')
    loss.backward()
    assert x.grad.tolist() == pytest.approx([-.25, -.25])
    with pytest.raises(ValueError):
        objectness_loss(x, torch.zeros(2, dtype=torch.bool), 'balanced_groups')
    with pytest.raises(ValueError):
        S0Criterion('unknown')


def test_balanced_criterion_keeps_shared_assignment_and_geometry_losses():
    output, truth = fixture()
    legacy, m1 = S0Criterion()(output, [truth])
    balanced, m2 = S0Criterion('balanced_groups')(output, [truth])
    assert torch.equal(m1[0][0], m2[0][0])
    for key in ['source_position', 'source_size', 'source_angle', 'target']:
        assert torch.equal(legacy[key], balanced[key])


def fixture(n=2):
    xy = torch.tensor([[.10 + .07*i, .2] for i in range(n)])
    truth = {'source_xy': xy, 'source_angle': torch.zeros(n),
             'target_xy': xy.flip(0), 'target_angle': torch.zeros(n),
             'size': torch.tensor([[.060, .035]]).repeat(n, 1),
             'color': torch.zeros(n, dtype=torch.long), 'shape': torch.zeros(n, dtype=torch.long)}
    source = {'uv': to_normalized(torch.cat((xy, torch.tensor([[.55, .35]])))).unsqueeze(0).requires_grad_(),
              'size': torch.tensor([.060, .035]).expand(1, n+1, 2).clone().requires_grad_(),
              'angle': torch.zeros(1, n+1, requires_grad=True), 'objectness_logits': torch.zeros(1, n+1, requires_grad=True),
              'color_logits': torch.zeros(1, n+1, 3, requires_grad=True), 'shape_logits': torch.zeros(1, n+1, 3, requires_grad=True)}
    target = {'uv': to_normalized(torch.cat((truth['target_xy'], torch.tensor([[.55, .35]])))).unsqueeze(0).requires_grad_(),
              'angle': torch.zeros(1, n+1, requires_grad=True)}
    return {'source': source, 'targets': [target, target, target]}, truth


def test_loss_scales_exact_values():
    assert position_loss(torch.tensor([[.01, 0.]]), torch.zeros(1, 2)).item() == pytest.approx(.25)
    assert size_loss(torch.tensor([[.005, 0.]]), torch.zeros(1, 2)).item() == pytest.approx(.25)
    assert orientation_loss(torch.tensor([math.pi/18]), torch.zeros(1)).item() == pytest.approx(.5)
    output, truth = fixture()
    terms, _ = S0Criterion()(output, [truth])
    assert terms['objectness'].item() == pytest.approx(math.log(2))
    assert terms['target'].item() == pytest.approx(0., abs=1e-10)
    assert terms['loss'].item() == pytest.approx(math.log(2) + .2*math.log(3), abs=1e-6)


def test_source_assignment_reused_for_crossing_targets_and_classes_excluded():
    output, truth = fixture()
    queries, gt = match_source(output['source'], [truth])[0]
    assert queries.tolist() == [0, 1] and gt.tolist() == [0, 1]
    wrong_target = dict(truth, target_xy=truth['source_xy'], color=1-truth['color'])
    assert torch.equal(match_source(output['source'], [wrong_target])[0][0], queries)
    correct, _ = S0Criterion()(output, [truth])
    wrong, _ = S0Criterion()(output, [wrong_target])
    assert correct['target'] < 1e-8 and wrong['target'] > 1
    # Reversing ALL GT arrays must preserve losses and the ID-based target join.
    permuted = {k: v.flip(0) for k, v in truth.items()}
    changed, _ = S0Criterion()(output, [permuted])
    assert torch.allclose(correct['loss'], changed['loss'])


def test_target_stage_weights_and_unmatched_objectness_normalization():
    output, truth = fixture()
    output['targets'] = [{k: v.clone() for k, v in stage.items()} for stage in output['targets']]
    for k, stage in enumerate(output['targets']):
        stage['uv'] = stage['uv'] + torch.tensor([.01/.6 * (k+1), 0.])
    terms, _ = S0Criterion()(output, [truth])
    expected = .25*.25 + .25*.75 + .5*1.25
    assert terms['target'].item() == pytest.approx(expected, abs=1e-5)
    terms['loss'].backward()
    # Positive BCE gradient -0.5/2.1; negative +0.1*0.5/2.1.
    assert output['source']['objectness_logits'].grad.tolist()[0] == pytest.approx([-.5/2.1, -.5/2.1, .05/2.1])


@pytest.mark.parametrize('angle', [0., math.pi/2, -math.pi/2, math.pi, math.radians(179)])
def test_periodic_orientation_finite_at_90_degree_boundary(angle):
    prediction = torch.tensor([angle], requires_grad=True)
    value = orientation_loss(prediction, torch.zeros(1)).sum()
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(prediction.grad).all()
    if abs(angle) == math.pi:
        assert value.item() < 1e-10


def test_scene_reduction_variable_object_counts_and_low_objectness_targets():
    a, ta = fixture(2)
    b, tb = fixture(3)
    # Pad the smaller prediction's unmatched queries; recompute its own objectness baseline.
    for key in a['source']:
        a['source'][key] = torch.cat((a['source'][key], a['source'][key][:, -1:]), 1)
    for stage in [a['targets'][0]]:
        for key in stage:
            stage[key] = torch.cat((stage[key], stage[key][:, -1:]), 1)
    a['source']['objectness_logits'] = torch.full((1, 4), -20., requires_grad=True)
    a['targets'][0]['uv'] = a['targets'][0]['uv'] + .01
    a['targets'] = [a['targets'][0]]*3
    combined = {'source': {k: torch.cat((a['source'][k], b['source'][k])) for k in a['source']},
                'targets': [{k: torch.cat((x[k], y[k])) for k in x} for x,y in zip(a['targets'],b['targets'])]}
    criterion = S0Criterion()
    result, _ = criterion(combined, [ta, tb])
    ra, _ = criterion(a, [ta]); rb, _ = criterion(b, [tb])
    assert torch.allclose(result['loss'], (ra['loss']+rb['loss'])/2)
    assert ra['target'] > 0  # Supervision never disappears at low objectness.


def test_one_hungarian_call_per_scene(monkeypatch):
    import knolling_model.losses as losses
    original = losses.linear_sum_assignment
    calls = []
    def counted(cost):
        calls.append(cost)
        return original(cost)
    monkeypatch.setattr(losses, 'linear_sum_assignment', counted)
    output, truth = fixture()
    S0Criterion()(output, [truth])
    assert len(calls) == 1 and calls[0].shape == (3, 2)
