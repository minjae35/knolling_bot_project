"""Row assignment reads the object set and leaves coordinates fixed."""
import torch
from torch.nn import functional as F

from knolling_model.experiments.row_layout import NeuralRowPlacement, assignment_loss, load_row_weights


def _batch():
    hidden = torch.randn(1, 3, 128)
    valid = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    goal = torch.zeros(1, 13)
    goal[0, 6] = 1
    widths = torch.tensor([[.04, .05, 0.]])
    heights = torch.tensor([[.02, .03, 0.]])
    row = torch.tensor([[0, 1, -1]])
    slot = torch.tensor([[0, 0, -1]])
    return hidden, valid, goal, widths, heights, row, slot


def test_padding_is_outside_the_assignment_loss_and_coordinates_get_no_gradient():
    torch.manual_seed(0)
    model = NeuralRowPlacement()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('assign'))
    hidden, valid, goal, widths, heights, row, slot = _batch()
    output = model(hidden, valid, goal, widths, heights, row, slot, False)
    loss = assignment_loss(output, row, valid)
    changed = output['row_logits'].detach().clone()
    changed[0, 2] += 3
    other = assignment_loss(dict(row_logits=changed), row, valid)
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert model.assign[-1].weight.grad.abs().sum() > 0
    assert model.assign_set.value.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith('assign'))


def test_row_prediction_follows_the_object_when_the_input_order_changes():
    torch.manual_seed(2)
    model = NeuralRowPlacement()
    torch.nn.init.xavier_uniform_(model.assign_set.value.weight)
    hidden = torch.randn(1, 4, 128)
    valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    goal = torch.zeros(1, 13)
    goal[0, 6] = 1
    widths = torch.tensor([[.04, .07, .05, 0.]])
    heights = torch.tensor([[.02, .03, .06, 0.]])
    order = torch.tensor([2, 0, 1, 3])

    def rows_by_id(permutation):
        output = model(hidden[:, permutation], valid[:, permutation], goal, widths[:, permutation],
                       heights[:, permutation], None, None, True)
        return {int(permutation[index]): int(output['row'][0, index])
                for index in range(permutation.shape[0]) if bool(valid[0, permutation[index]])}

    assert rows_by_id(torch.arange(4)) == rows_by_id(order)


def test_a_pointwise_checkpoint_keeps_its_logits_until_the_context_is_trained():
    torch.manual_seed(3)
    model = NeuralRowPlacement()
    state = model.state_dict()
    old = torch.randn(128, 141)
    state['assign.0.weight'] = old
    load_row_weights(model, state)
    hidden, valid, goal, widths, heights, row, slot = _batch()
    output = model(hidden, valid, goal, widths, heights, row, slot, False)
    manual = torch.cat((hidden, hidden.new_zeros(hidden.shape), goal[:, None].expand(-1, hidden.shape[1], -1)), -1)
    expected = F.linear(F.gelu(F.linear(manual, model.assign[0].weight, model.assign[0].bias)),
                        model.assign[2].weight, model.assign[2].bias)
    torch.testing.assert_close(output['row_logits'], expected)
    assert torch.count_nonzero(model.assign[0].weight[:, 128:-13]) == 0
    torch.nn.init.xavier_uniform_(model.assign_set.value.weight)
    held = model(hidden, valid, goal, widths, heights, row, slot, False)
    torch.testing.assert_close(held['row_logits'], expected)
