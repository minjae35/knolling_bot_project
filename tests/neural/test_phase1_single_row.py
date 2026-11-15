"""CPU stand-in for the single-row pose module those sequential tests build on."""
import torch
from torch import nn

from knolling_model.arrangement.contracts import Objects
from knolling_model.arrangement.goal_joint_layout import GoalSetBlock
from knolling_model.arrangement.layout import PeriodicBins


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


def example(device='cpu', extra=0):
    device = torch.device(device)
    torch.manual_seed(1203)
    base = 3
    n = base + int(extra)
    xy = torch.tensor([[[0.12, 0.18], [0.20, 0.16], [0.28, 0.19]]], device=device)
    size = torch.tensor([[[0.060, 0.030], [0.050, 0.028], [0.070, 0.032]]], device=device)
    angle = torch.tensor([[0.1, -0.2, 0.4]], device=device)
    if extra:
        xy = torch.cat((xy, torch.zeros(1, extra, 2, device=device)), 1)
        size = torch.cat((size, torch.tensor([0.060, 0.030], device=device).expand(1, extra, 2).contiguous()), 1)
        angle = torch.cat((angle, torch.zeros(1, extra, device=device)), 1)
    valid = torch.zeros(1, n, dtype=torch.bool, device=device)
    valid[:, :base] = True
    color = torch.zeros(1, n, 3, device=device)
    color[..., 0] = 1
    shape = torch.zeros(1, n, 3, device=device)
    shape[..., 0] = 1
    query = torch.full((1, n), -1, dtype=torch.long, device=device)
    query[:, :base] = torch.arange(base, device=device)
    goal = torch.zeros(1, 13, device=device)
    goal[:, 6:9] = torch.tensor([0.2, 0.5, 0.3], device=device)
    return Objects(xy, size, angle, color, shape, valid, query), goal, valid.float()
