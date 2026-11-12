"""One object-level attention in front of row assignment, order, and origin.

The residual projection starts at zero, so an existing row checkpoint keeps
its outputs until this block is trained. Detector, language, and size features
stay outside this module. Teacher rows are not an input.
"""
import torch
from torch import nn


class ObjectSetContext(nn.Module):
    def __init__(self, hidden=128, heads=4):
        super().__init__()
        if hidden < 1 or hidden % heads:
            raise ValueError('object context width is invalid')
        self.norm = nn.LayerNorm(hidden)
        self.goal = nn.Linear(13, hidden)
        self.size = nn.Linear(2, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.zeros_(self.goal.weight)
        nn.init.zeros_(self.goal.bias)
        nn.init.zeros_(self.size.weight)
        nn.init.zeros_(self.size.bias)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, hidden, valid, goal, widths, heights):
        if hidden.shape[:2] != valid.shape or widths.shape != valid.shape or heights.shape != valid.shape:
            raise ValueError('object context batch shape mismatch')
        if goal.ndim != 2 or goal.shape != (len(hidden), 13):
            raise ValueError('object context expects the 13-d goal')
        size = torch.stack((widths, heights), -1)
        tokens = hidden+self.goal(goal)[:, None]+self.size(size)
        query = self.norm(tokens)
        keys = torch.cat((query, self.null.expand(len(hidden), -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros(len(hidden), 1, dtype=torch.bool, device=valid.device)), 1)
        mixed = self.attn(query, keys, keys, key_padding_mask=padding, need_weights=False)[0]
        mixed = mixed.masked_fill(~valid[..., None], 0)
        return hidden+mixed


class SetArrangement(nn.Module):
    """Set encoder for row and order. Pose reads the original object features.

    A change in the set encoder moves coordinates only when the discrete row
    or the within-row order changes. Teacher rows are not inference inputs.
    """

    def __init__(self, placement):
        super().__init__()
        self.placement = placement
        self.context = ObjectSetContext(placement.hidden_dim)

    def forward(self, hidden, valid, goal, widths, heights, teacher_row=None, teacher_slot=None, predict_order=False, colors=None):
        mixed = self.context(hidden, valid, goal, widths, heights)
        return self.placement(hidden, valid, goal, widths, heights, teacher_row, teacher_slot, predict_order,
                              selector=mixed, colors=colors)


class ContextualRows(nn.Module):
    """Same call signature as NeuralRowPlacement. Angle still reads the original hidden."""

    def __init__(self, placement, context):
        super().__init__()
        self.placement = placement
        self.context = context

    def forward(self, hidden, valid, goal, widths, heights, teacher_row=None, teacher_slot=None, predict_order=False):
        mixed = self.context(hidden, valid, goal, widths, heights)
        return self.placement(hidden, valid, goal, widths, heights, teacher_row, teacher_slot, predict_order,
                              context=mixed)
