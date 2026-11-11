"""A row has one Y. Later X steps see the two widths and the goal gap."""
import torch

from torch.nn import functional as F

from knolling_model.experiments.relation_pose import NeuralRelationPose
from knolling_model.experiments.row_layout import NeuralRowPlacement, origin_loss, placement_loss, step_placement_loss


def _scene():
    torch.manual_seed(4)
    head = NeuralRelationPose(12)
    head.requires_grad_(False)
    model = NeuralRowPlacement()
    features = torch.randn(2, 4, 12)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    goal = torch.zeros(2, 13)
    goal[:, 6] = 1
    row = torch.tensor([[0, 0, 1, -1], [0, 1, 1, 0]])
    slot = torch.tensor([[0, 1, 0, -1], [0, 0, 1, 1]])
    widths = torch.tensor([[.04, .06, .05, 0.], [.03, .04, .05, .06]])
    heights = torch.tensor([[.02, .03, .04, 0.], [.02, .05, .03, .04]])
    hidden = head.hidden(features)
    return head, model, hidden, valid, goal, widths, heights, row, slot


def test_one_row_shares_y_and_training_does_not_touch_the_relation_head():
    head, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    output = model(hidden, valid, goal, widths, heights, row, slot)
    origin = output['origins'].gather(1, row.clamp_min(0)[..., None].expand(-1, -1, 2))
    first = valid & (slot == 0)
    torch.testing.assert_close(output['xy'][first], origin[first])
    for local in range(2):
        for row_id in row[local].unique().tolist():
            if row_id < 0:
                continue
            members = valid[local] & (row[local] == row_id)
            column = output['xy'][local, members, 1]
            torch.testing.assert_close(column, column[:1].expand_as(column))
    target = torch.rand_like(output['xy'])
    placement_loss(output, target, torch.full(valid.shape, 0.4), valid)['loss'].backward()
    assert model.row_layout[-1].weight.grad.abs().sum() > 0
    assert model.step[-1].weight.grad.abs().sum() > 0
    assert model.angle[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in head.parameters())


def test_current_width_moves_later_x_without_moving_the_shared_y():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    with torch.no_grad():
        model.step[0].weight.zero_()
        model.step[0].bias.zero_()
        model.step[0].weight[0, 1] = 1.
        model.step[-1].weight.zero_()
        model.step[-1].weight[0, 0] = 1.
        model.step[-1].bias.zero_()
    before = model(hidden, valid, goal, widths, heights, row, slot)
    wider = widths.clone()
    wider[0, 1] = .20
    after = model(hidden, valid, goal, wider, heights, row, slot)
    torch.testing.assert_close(before['xy'][..., 1], after['xy'][..., 1])
    assert float(after['xy'][0, 1, 0].detach()) > float(before['xy'][0, 1, 0].detach())


def test_predicted_slots_are_a_permutation_and_teacher_slots_keep_the_coordinates():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    free = model(hidden, valid, goal, widths, heights, row, None, predict_order=True)
    for local in range(2):
        for row_id in row[local].unique().tolist():
            if row_id < 0:
                continue
            members = (valid[local] & (row[local] == row_id)).nonzero().flatten()
            found = free['slot'][local, members].tolist()
            assert sorted(found) == list(range(len(members)))
    forced = model(hidden, valid, goal, widths, heights, row, slot, predict_order=True)
    plain = model(hidden, valid, goal, widths, heights, row, slot, predict_order=False)
    torch.testing.assert_close(forced['xy'], plain['xy'])
    assert forced['order_logits'] is not None


def test_step_dx_is_not_the_rejected_absolute_offset():
    xy = torch.tensor([[[.30, .20], [.40, .20]]], requires_grad=True)
    delta = torch.tensor([[0., .05]], requires_grad=True)
    target = torch.tensor([[[.10, .20], [.19, .20]]])
    valid = torch.tensor([[1, 1]], dtype=torch.bool)
    row = torch.tensor([[0, 0]])
    slot = torch.tensor([[0, 1]])
    angle = torch.zeros(1, 2)
    losses = step_placement_loss(dict(xy=xy, delta_x=delta, angle=angle), target, angle, valid, row, slot)
    absolute = F.smooth_l1_loss(xy[0, 1, 0], target[0, 1, 0], beta=.010)
    previous = target[0, 0, 0].detach()
    cancelled = F.smooth_l1_loss(xy[0, 1, 0]-previous, target[0, 1, 0]-previous, beta=.010)
    torch.testing.assert_close(cancelled, absolute)
    assert not torch.allclose(losses['dx'], absolute)
    losses['loss'].backward()
    assert xy.grad[0, 1, 0] == 0
    assert xy.grad[0, 0, 0] != 0
    assert delta.grad[0, 1] != 0
    assert delta.grad[0, 0] == 0


def test_origin_loss_leaves_the_step_alone_and_other_row_height_is_an_input():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    output = model(hidden, valid, goal, widths, heights, row, slot)
    origin_loss(output, torch.rand_like(output['xy']), valid, slot).backward()
    assert model.row_layout[-1].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(model.step[-1].weight.grad) == 0
    taller = heights.clone()
    taller[0, 1] += .04
    before = model.origin_inputs(hidden, valid, goal, row, widths, heights, slot)
    after = model.origin_inputs(hidden, valid, goal, row, widths, taller, slot)
    channel = model.above_height_channel
    torch.testing.assert_close(after[0, 1, channel]-before[0, 1, channel], taller.new_tensor(.04))
    torch.testing.assert_close(after[0, 1, model.own_height_channel], before[0, 1, model.own_height_channel])
