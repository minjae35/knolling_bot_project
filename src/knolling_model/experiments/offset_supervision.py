"""Pose loss only: one-step X offset under a fixed Teacher prefix.

Inference never imports this target. The previous pose is the constant workspace
anchor at the first step and the detached GT pose afterwards. Y is not a target.
"""
import torch
from torch.nn import functional as F

SCALE = .010  # Same Huber scale as the existing absolute XY loss.
ANCHOR_X = .3


def x_offset_loss(xy, order, target_xy):
    """Huber on (predicted x - fixed previous x) for each Teacher step.

    `xy` must be the decoder output from that same Teacher order with
    `teacher_xy` set to `target_xy`. The fixed previous x is detached, so the
    gradient stays in the current step's offset.
    """
    if xy.ndim != 3 or xy.shape[-1] != 2 or order.shape != xy.shape[:2] or target_xy.shape != xy.shape:
        raise ValueError('X-offset tensors must be aligned BxNx2 predictions and a BxN order')
    if order.dtype != torch.long:
        raise ValueError('teacher order must be integer labels')
    if not torch.isfinite(xy).all() or not torch.isfinite(target_xy).all():
        raise ValueError('nonfinite X offset')
    scenes = []
    for b in range(xy.shape[0]):
        previous = xy.new_tensor(ANCHOR_X)
        terms = []
        for step in range(order.shape[1]):
            chosen = int(order[b, step])
            if chosen < 0:
                continue
            if chosen >= xy.shape[1]:
                raise ValueError('teacher order leaves the object axis')
            predicted = xy[b, chosen, 0] - previous
            target = target_xy[b, chosen, 0] - previous.detach()
            terms.append(F.smooth_l1_loss(predicted / SCALE, target / SCALE))
            previous = target_xy[b, chosen, 0].detach()
        if not terms:
            raise ValueError('teacher order placed no object')
        scenes.append(torch.stack(terms).mean())
    return torch.stack(scenes).mean()
