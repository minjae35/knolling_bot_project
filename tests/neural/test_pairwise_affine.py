"""CPU checks for the learnable pairwise X path. No dataset and no solver."""
import torch
from torch import nn
from knolling_model.arrangement.contracts import Objects
from knolling_model.arrangement.goal_joint_layout import GoalSetBlock
from knolling_model.arrangement.layout import PeriodicBins
from knolling_model.experiments.pairwise_affine import (
    PairwiseAffineResidual, SequentialPose, position_loss)


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(13, 16)
        self.blocks = nn.ModuleList([GoalSetBlock(16, 4) for _ in range(2)])
        self.orientation = PeriodicBins(18)

    def forward(self, objects, goal, selected):
        move = (selected >= .5) & objects.valid
        return dict(xy=objects.xy, angle=objects.angle, proposal_angle=objects.angle,
                    move_mask=move, valid=objects.valid, query_index=objects.query_index)


def example():
    torch.manual_seed(1203)
    n = 3
    valid = torch.ones(1, n, dtype=torch.bool)
    size = torch.tensor([[[0.060, 0.030], [0.050, 0.028], [0.070, 0.032]]])
    obj = Objects(torch.tensor([[[0.12, 0.18], [0.20, 0.16], [0.28, 0.19]]]), size,
                  torch.tensor([[0.1, -0.2, 0.4]]),
                  torch.ones(1, n, 3)/3, torch.ones(1, n, 3)/3, valid,
                  torch.arange(n)[None])
    goal = torch.zeros(1, 13)
    goal[:, 6:9] = torch.tensor([0.2, 0.5, 0.3])
    return obj, goal, valid.float()


def test_affine_coefficients_are_learnable_and_size_moves_x():
    layer = PairwiseAffineResidual().double().eval()
    prev = torch.tensor([[0.05, 0.03]], dtype=torch.double)
    curr = torch.tensor([[0.06, 0.04]], dtype=torch.double)
    gap = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.double)
    angle = torch.zeros(1, dtype=torch.double)
    step = 1e-3
    plus = layer(prev, curr+torch.tensor([[step, 0.]]), gap, angle, angle)
    minus = layer(prev, curr-torch.tensor([[step, 0.]]), gap, angle, angle)
    gain = float(((plus-minus)/(2*step))[0].detach())
    assert abs(gain-0.25) < 1e-6
    curr = curr.clone().requires_grad_(True)
    value = layer(prev, curr, gap, angle, angle).sum()
    value.backward()
    torch.testing.assert_close(curr.grad[0, 0], torch.tensor(0.25, dtype=torch.double))
    assert layer.w_curr.grad[0].item() != 0
    assert layer.w_gap.grad.abs().sum() > 0
    assert layer.w_pair.grad.abs().sum() > 0
    assert layer.w_gap_prev.grad.abs().sum() > 0
    assert layer.w_gap_curr.grad.abs().sum() > 0
    assert layer.residual[-1].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(layer.residual[-1].weight) == 0


def test_pose_x_follows_size_until_the_affine_path_is_removed():
    torch.manual_seed(1203)
    model = SequentialPose(Reference()).eval()
    obj, goal, selected = example()
    order = torch.tensor([[0, 1, 2]])

    def placed_x(size_delta):
        changed = Objects(obj.xy, obj.size.clone(), obj.angle, obj.color, obj.shape,
                          obj.valid, obj.query_index)
        changed.size[:, 1, 0] += size_delta
        return model.diagnose(changed, goal, selected, order)['xy'][0, 1, 0]

    millimetres = float(placed_x(0.001)-placed_x(0.0))*1000
    assert millimetres > 0.15
    with torch.no_grad():
        model.decoder.offset.bias.zero_()
        model.decoder.offset.w_prev.zero_()
        model.decoder.offset.w_curr.zero_()
        model.decoder.offset.w_gap.zero_()
        model.decoder.offset.w_pair.zero_()
        model.decoder.offset.w_gap_prev.zero_()
        model.decoder.offset.w_gap_curr.zero_()
    suppressed = abs(float(placed_x(0.001)-placed_x(0.0))*1000)
    assert suppressed < 0.05
    out = model(obj, goal, selected)
    position_loss(out, out['xy'].detach()+torch.tensor([0.004, -0.002])).backward()
    assert model.decoder.offset.w_curr.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.reference.parameters())
