"""A frozen relation head plus a row-break residual.

The relation weights are never updated. A new row adds a learned displacement
on top of the frozen delta. A zero residual reproduces the modular head, and
inference uses the predicted break decision rather than the label.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RowResidual(nn.Module):
    def __init__(self, hidden=128, goal_dim=13):
        super().__init__()
        width = hidden*2+goal_dim
        self.classifier = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.residual = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, -4.)

    def forward(self, head, features, valid, goal, teacher_order=None, teacher_break=None):
        with torch.no_grad():
            base = head(features, valid, goal, teacher_order)
            hidden = head._hidden(features).detach()
        order = base['predicted_order'] if teacher_order is None else teacher_order
        batch, count, _ = hidden.shape
        rows = torch.arange(batch, device=hidden.device)
        logits = features.new_zeros(batch, count)
        applied = features.new_zeros(batch, count)
        xy = features.new_zeros(batch, count, 2)
        deltas = base['deltas'].detach().clone()
        carried_h = carried_xy = None
        for step in range(count):
            active = order[:, step] >= 0
            chosen = order[:, step].clamp_min(0)
            current_h = hidden[rows, chosen]
            current_xy = base['xy'].detach()[rows, chosen]
            current_delta = base['deltas'].detach()[rows, chosen]
            if carried_h is None:
                predicted = current_xy
            else:
                pair = torch.cat((carried_h, current_h, goal), -1)
                logit = self.classifier(pair).squeeze(-1)
                correction = self.residual(pair)
                if teacher_break is None:
                    bit = (logit.detach() > 0).to(pair.dtype)
                else:
                    bit = teacher_break[rows, chosen].to(pair.dtype)
                placed = active & (order[:, step-1] >= 0)
                write = F.one_hot(chosen, count).to(pair.dtype)*placed[:, None]
                logits = logits+write*logit[:, None]
                applied = applied+write*bit[:, None]
                extra = bit[:, None]*correction
                deltas = deltas+write[..., None]*extra[:, None]
                predicted = carried_xy+current_delta+extra
            where = F.one_hot(chosen, count).bool() & active[:, None]
            xy = torch.where(where[..., None], predicted[:, None], xy)
            fallback_h = current_h if carried_h is None else carried_h
            fallback_xy = predicted if carried_xy is None else carried_xy
            carried_h = torch.where(active[:, None], current_h, fallback_h)
            carried_xy = torch.where(active[:, None], predicted, fallback_xy)
        return dict(xy=xy.masked_fill(~valid[..., None], 0), angle=base['angle'],
                    predicted_order=order, deltas=deltas, transition_logits=logits, break_bit=applied,
                    rollout='teacher_order' if teacher_order is not None else 'free_rollout')


class FrozenRowModel(nn.Module):
    """Inference signature matches the relation head: features, mask, goal."""

    def __init__(self, head, residual):
        super().__init__()
        self.head = head
        self.row = residual

    def forward(self, features, valid, goal, teacher_order=None, teacher_break=None):
        return self.row(self.head, features, valid, goal, teacher_order, teacher_break)


def residual_loss(output, target_xy, teacher_order, teacher_break):
    if output['rollout'] != 'teacher_order':
        raise ValueError('the residual is supervised along the teacher order')
    current = teacher_order[:, 1:]
    previous = teacher_order[:, :-1]
    active = (current >= 0) & (previous >= 0)
    rows = torch.arange(len(teacher_order), device=teacher_order.device)[:, None]
    batch = rows.expand_as(current)
    relative = target_xy[batch, current.clamp_min(0)]-target_xy[batch, previous.clamp_min(0)]
    got = output['deltas'][batch, current.clamp_min(0)]
    delta = F.smooth_l1_loss(got/.010, relative/.010, reduction='none').sum(-1)
    delta = delta.masked_fill(~active, 0).sum()/active.sum().clamp_min(1)
    logits = output['transition_logits'][batch, current.clamp_min(0)]
    target = teacher_break[batch, current.clamp_min(0)]
    decision = F.binary_cross_entropy_with_logits(logits[active], target[active])
    return dict(loss=delta+decision, delta=delta, transition=decision)
