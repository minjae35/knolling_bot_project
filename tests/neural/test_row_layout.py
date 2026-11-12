"""A row has one Y. Later X steps see the two widths and the goal gap."""
import torch

from torch.nn import functional as F

from knolling_model.experiments.relation_pose import NeuralRelationPose
from knolling_model.experiments.row_layout import (
    NeuralRowPlacement, assignment_loss, chain_centers, color_separation_loss, order_loss, origin_loss,
    placement_loss, step_placement_loss)
from knolling_model.experiments.set_context import ObjectSetContext, SetArrangement


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


def test_zero_context_matches_the_row_model_and_its_gradient_skips_the_step():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    context = ObjectSetContext()
    with torch.no_grad():
        mixed = context(hidden, valid, goal, widths, heights)
        torch.testing.assert_close(mixed, hidden)
        plain = model(hidden, valid, goal, widths, heights, None, None, True)
        held = model(hidden, valid, goal, widths, heights, None, None, True, context=mixed)
    torch.testing.assert_close(plain['xy'], held['xy'])
    torch.testing.assert_close(plain['row_logits'], held['row_logits'])
    model.requires_grad_(False)
    for parameter in model.assign.parameters():
        parameter.requires_grad_(True)
    output = model(hidden, valid, goal, widths, heights, None, None, True,
                   context=context(hidden, valid, goal, widths, heights))
    assignment_loss(output, row, valid).backward()
    assert context.attn.out_proj.weight.grad.abs().sum() > 0
    assert model.step[-1].weight.grad is None
    assert model.angle[-1].weight.grad is None


def test_color_grouping_penalizes_a_shared_row_and_ignores_other_instructions():
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 0] = 20
    logits[0, 1, 0] = 20
    logits[0, 2, 0] = 20
    color = torch.eye(3)[None]
    valid = torch.ones(1, 3, dtype=torch.bool)
    mixed = color_separation_loss(logits, color, valid, torch.tensor([True]))
    separated = logits.clone()
    separated[0, 1] = 0
    separated[0, 1, 1] = 20
    separated[0, 2] = 0
    separated[0, 2, 2] = 20
    pure = color_separation_loss(separated, color, valid, torch.tensor([True]))
    assert float(mixed) > .5
    assert float(pure) < 1e-4
    assert float(color_separation_loss(logits, color, valid, torch.tensor([False]))) == 0
    monochrome = color.clone()
    monochrome[:] = color[:, :1]
    assert float(color_separation_loss(logits, monochrome, valid, torch.tensor([True]))) == 0


def test_zero_row_gap_matches_the_frozen_origin_and_a_gap_moves_only_later_rows():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    output = model(hidden, valid, goal, widths, heights, row, slot, False)
    base = model.row_layout(model.origin_inputs(hidden, valid, goal, row, widths, heights, slot))
    torch.testing.assert_close(output['origins'], base)
    shifted = NeuralRowPlacement()
    shifted.load_state_dict(model.state_dict())
    with torch.no_grad():
        shifted.row_gap[-1].bias.fill_(.01)
    moved = shifted(hidden, valid, goal, widths, heights, row, slot, False)
    torch.testing.assert_close(moved['origins'][:, 0], output['origins'][:, 0])
    torch.testing.assert_close(moved['xy'][..., 0], output['xy'][..., 0])
    assert torch.all(moved['origins'][:, 1, 1] < output['origins'][:, 1, 1]-.009)


def test_chain_keeps_the_gap_when_a_row_grows():
    heights = torch.tensor([.04, .06, .05])
    gaps = torch.tensor([.02, .03])
    before = chain_centers(torch.tensor(.30), heights, gaps)
    taller = heights.clone()
    taller[0] = .08
    after = chain_centers(torch.tensor(.30), taller, gaps)
    torch.testing.assert_close(after[0], before[0])
    torch.testing.assert_close(before[0]-before[1], heights[0]/2+gaps[0]+heights[1]/2)
    torch.testing.assert_close(after[1], before[1]-.02)
    torch.testing.assert_close(after[2], before[2]-.02)


def test_old_row_checkpoint_keeps_zero_gap_heads():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    saved = {key: value for key, value in model.state_dict().items()
             if not key.startswith('row_anchor') and not key.startswith('row_gap')}
    restored = NeuralRowPlacement()
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.row_anchor.weight, torch.zeros_like(restored.row_anchor.weight))
    torch.testing.assert_close(restored.row_gap[-1].weight, torch.zeros_like(restored.row_gap[-1].weight))
    torch.testing.assert_close(
        restored(hidden, valid, goal, widths, heights, row, slot, False)['xy'],
        model(hidden, valid, goal, widths, heights, row, slot, False)['xy'])


def test_selector_can_change_the_row_without_moving_a_fixed_rows_pose():
    _, model, hidden, valid, goal, widths, heights, row, slot = _scene()
    arrangement = SetArrangement(model)
    with torch.no_grad():
        arrangement.context.attn.out_proj.weight.normal_()
        plain = model(hidden, valid, goal, widths, heights, row, slot, False)
        arranged = arrangement(hidden, valid, goal, widths, heights, row, slot, False)
    torch.testing.assert_close(arranged['xy'], plain['xy'])
    torch.testing.assert_close(arranged['origins'], plain['origins'])
    assert not torch.allclose(arranged['row_logits'], plain['row_logits'])
    model.requires_grad_(False)
    for parameter in list(model.assign.parameters())+list(model.order_query.parameters())+[model.order_begin]:
        parameter.requires_grad_(True)
    output = arrangement(hidden, valid, goal, widths, heights, row, slot, True)
    (assignment_loss(output, row, valid)+order_loss(output)).backward()
    assert arrangement.context.attn.out_proj.weight.grad.abs().sum() > 0
    assert model.row_layout[-1].weight.grad is None
    assert model.step[-1].weight.grad is None
    model.zero_grad(set_to_none=True)
    arrangement.context.zero_grad(set_to_none=True)
    for parameter in list(model.row_layout.parameters())+list(model.step.parameters()):
        parameter.requires_grad_(True)
    posed = arrangement(hidden, valid, goal, widths, heights, row, slot, False)
    F.smooth_l1_loss(posed['xy'][valid], torch.zeros_like(posed['xy'][valid])).backward()
    assert model.row_layout[-1].weight.grad.abs().sum() > 0
    assert arrangement.context.attn.out_proj.weight.grad is None
