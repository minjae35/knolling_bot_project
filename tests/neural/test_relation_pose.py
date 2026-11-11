"""The relation head owns neighbor displacement and does not share it."""
import torch
from knolling_model.experiments.relation_pose import NeuralRelationPose, relation_loss


def batch():
    torch.manual_seed(0)
    features = torch.randn(2, 3, 12)
    valid = torch.ones(2, 3, dtype=torch.bool)
    goal = torch.zeros(2, 13)
    goal[:, 6] = 1
    teacher = torch.tensor([[0, 1, 2], [2, 0, 1]])
    target = torch.tensor([[[.20, .20], [.30, .21], [.42, .19]],
                           [[.25, .18], [.36, .22], [.48, .20]]])
    angle = torch.zeros(2, 3)
    return features, valid, goal, teacher, target, angle


def test_later_position_is_only_the_relation_delta():
    features, valid, goal, teacher, target, angle = batch()
    head = NeuralRelationPose(12)
    out = head(features, valid, goal, teacher)
    rows = torch.arange(2)
    for step in (1, 2):
        prev, curr = teacher[:, step-1], teacher[:, step]
        spacing = out['xy'][rows, curr]-out['xy'][rows, prev]
        torch.testing.assert_close(spacing, out['deltas'][rows, curr])
    own = {id(p) for p in head.relation.parameters()}
    assert own.isdisjoint(id(p) for p in head.anchor.parameters())


def test_row_transition_preserves_the_old_delta_until_a_break_is_chosen():
    features, valid, goal, teacher, target, angle = batch()
    torch.manual_seed(1)
    old = NeuralRelationPose(12)
    new = NeuralRelationPose(12)
    new.load_state_dict(old.state_dict())
    new.enable_row_transition()
    left = old(features, valid, goal, teacher)
    right = new(features, valid, goal, teacher)
    torch.testing.assert_close(left['xy'], right['xy'])
    torch.testing.assert_close(left['deltas'], right['deltas'])
    assert torch.all(right['transition_logits'] <= 0)
    assert torch.any(right['transition_logits'] < 0)
    bits = torch.zeros(2, 3)
    bits[torch.arange(2), teacher[:, 1]] = 1
    supervised = new(features, valid, goal, teacher, bits)
    relation_loss(supervised, target, angle, teacher)['loss'].backward()
    assert new.relation[0].weight.grad[:, -1].abs().sum() > 0
    new.zero_grad(set_to_none=True)
    again = new(features, valid, goal, teacher, bits)
    again['transition_logits'].sum().backward()
    assert new.transition[-1].weight.grad.abs().sum() > 0


def test_relation_and_anchor_both_receive_the_pose_loss():
    features, valid, goal, teacher, target, angle = batch()
    head = NeuralRelationPose(12)
    out = head(features, valid, goal, teacher)
    loss = relation_loss(out, target, angle, teacher)
    loss['loss'].backward()
    assert head.relation[-1].weight.grad.abs().sum() > 0
    assert head.anchor[-1].weight.grad.abs().sum() > 0
    assert head.angle[-1].weight.grad.abs().sum() > 0
    free = head(features, valid, goal)
    assert free['rollout'] == 'free_rollout'
    assert set(free['predicted_order'][0].tolist()) == {0, 1, 2}
