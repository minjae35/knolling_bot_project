"""Neighbor relation pose. One displacement path, no placement solver.

The anchor head predicts only the first selected object's absolute position.
The relation head predicts only the displacement to the next object. Later
positions are that sum. Nothing else emits an X or Y offset, and inference
never reads a target pose or a clearance formula.
"""
import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.coordinates import angle_delta, wrap_pi


class NeuralRelationPose(nn.Module):
    def __init__(self, feature_dim, goal_dim=13, hidden=128):
        super().__init__()
        if feature_dim < 1 or goal_dim != 13:
            raise ValueError('relation pose expects a 13-d goal vector')
        self.feature_dim = feature_dim
        self.hidden = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.begin = nn.Parameter(torch.zeros(hidden))
        self.query = nn.Sequential(nn.Linear(hidden*2+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.anchor = nn.Sequential(nn.Linear(hidden+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.relation = nn.Sequential(nn.Linear(hidden*2+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.angle = nn.Sequential(nn.Linear(hidden+goal_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.anchor[-1].bias)
        self.anchor[-1].bias.data.copy_(torch.tensor([0., 0.]))

    def _hidden(self, features):
        return self.hidden(features)

    def _order(self, hidden, valid, goal, teacher):
        batch, count, width = hidden.shape
        previous = self.begin.expand(batch, -1)
        available = valid.clone()
        rows = torch.arange(batch, device=hidden.device)
        logits, order = [], []
        for _ in range(count):
            active = available.any(-1)
            context = (hidden*available[..., None]).sum(1)/available.sum(-1, keepdim=True).clamp_min(1)
            query = self.query(torch.cat((context, previous, goal), -1))
            scores = (hidden*query[:, None]).sum(-1)/(width**.5)
            scores = scores.masked_fill(~available, -1e9)
            chosen = teacher[:, len(order)].clamp_min(0) if teacher is not None else scores.argmax(-1)
            if teacher is not None and not bool((available[rows, chosen] | ~active).all()):
                raise ValueError('teacher order selects an unavailable object')
            logits.append(scores)
            order.append(torch.where(active, chosen, torch.full_like(chosen, -1)))
            previous = torch.where(active[:, None], hidden[rows, chosen], previous)
            available = available & ~(F.one_hot(chosen, count).bool() & active[:, None])
        return torch.stack(order, 1), torch.stack(logits, 1)

    def forward(self, features, valid, goal, teacher_order=None):
        if features.shape[-1] != self.feature_dim or goal.shape[-1] != 13:
            raise ValueError('feature or goal width does not match this head')
        if features.shape[:2] != valid.shape or len(features) != len(goal):
            raise ValueError('relation batch shape mismatch')
        hidden = self._hidden(features)
        order, logits = self._order(hidden, valid, goal, teacher_order)
        batch, count, _ = hidden.shape
        rows = torch.arange(batch, device=hidden.device)
        xy = features.new_zeros(batch, count, 2)
        raw_angle = self.angle(torch.cat((hidden, goal[:, None].expand(-1, count, -1)), -1))
        angle = wrap_pi(.5*torch.atan2(raw_angle[..., 1], raw_angle[..., 0])).masked_fill(~valid, 0)
        deltas = features.new_zeros(batch, count, 2)
        carried = None
        for step in range(count):
            active = order[:, step] >= 0
            chosen = order[:, step].clamp_min(0)
            current = hidden[rows, chosen]
            if carried is None:
                predicted = torch.sigmoid(self.anchor(torch.cat((current, goal), -1)))*current.new_tensor([.6, .4])
            else:
                delta = self.relation(torch.cat((carried, current, goal), -1))
                base = xy[rows, order[:, step-1].clamp_min(0)]
                predicted = base+delta
                placed = active & (order[:, step-1] >= 0)
                deltas = deltas+F.one_hot(chosen, count).to(deltas.dtype)[..., None]*delta[:, None]*placed[:, None, None]
            where = F.one_hot(chosen, count).bool() & active[:, None]
            xy = torch.where(where[..., None], predicted[:, None], xy)
            fallback = current if carried is None else carried
            carried = torch.where(active[:, None], current, fallback)
        return dict(xy=xy.masked_fill(~valid[..., None], 0), angle=angle, predicted_order=order,
                    order_logits=logits, deltas=deltas, rollout='free_rollout' if teacher_order is None else 'teacher_order')


def relation_loss(output, target_xy, target_angle, teacher_order):
    """Supervise the anchor and the neighbor displacement, not both at once."""
    if output['rollout'] != 'teacher_order':
        raise ValueError('relation supervision requires the teacher order forward')
    order = output['predicted_order']
    rows = torch.arange(len(order), device=order.device)
    first = order[:, 0].clamp_min(0)
    anchor = F.smooth_l1_loss(output['xy'][rows, first]/.010, target_xy[rows, first]/.010)
    active = (order[:, 1:] >= 0) & (order[:, :-1] >= 0)
    current = order[:, 1:].clamp_min(0)
    previous = order[:, :-1].clamp_min(0)
    batch = rows[:, None].expand_as(current)
    relative = target_xy[batch, current]-target_xy[batch, previous]
    got = output['deltas'][batch, current]
    delta = F.smooth_l1_loss(got/.010, relative/.010, reduction='none').sum(-1)
    delta = delta.masked_fill(~active, 0).sum()/active.sum().clamp_min(1)
    valid = teacher_order >= 0
    # Angle is not an X path. Compare every selected object once.
    selected = order.clamp_min(0)
    picked = rows[:, None].expand_as(selected)
    angle = angle_delta(output['angle'][picked, selected], target_angle[picked, selected]).abs()
    angle = angle.masked_fill(~valid, 0).sum()/valid.sum().clamp_min(1)
    labels = teacher_order.masked_fill(teacher_order < 0, -100)
    ce = F.cross_entropy(output['order_logits'].flatten(0, 1), labels.flatten(), ignore_index=-100)
    return dict(loss=anchor+delta+angle+ce, anchor=anchor, delta=delta, angle=angle, order_ce=ce)
