"""v1.3 sections 11.1–11.4. One source assignment, reused at every stage."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .coordinates import to_meters, angle_delta

POSITION_SCALE = .010
SIZE_SCALE = .005
ANGLE_SCALE = math.pi / 18
NEGATIVE_WEIGHT = .1
STAGE_WEIGHTS = (.25, .25, .50)


def rho(error):
    return F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none', beta=1.)


def position_loss(predicted_xy, truth_xy):
    return rho((predicted_xy - truth_xy) / POSITION_SCALE).mean(-1)


def size_loss(predicted, truth):
    return rho((predicted - truth) / SIZE_SCALE).mean(-1)


def orientation_loss(predicted, truth):
    return rho(angle_delta(predicted, truth) / ANGLE_SCALE)


@torch.no_grad()
def match_source(source, supervision):
    """No targets, classes or selection/lock labels enter assignment."""
    matches = []
    for b, truth in enumerate(supervision):
        n, q = len(truth['source_xy']), source['uv'].shape[1]
        if not 0 < n <= q:
            raise ValueError('positive S0 sample requires 1 <= N <= Q')
        cost = (-source['objectness_logits'][b].sigmoid()[:, None]
                + torch.cdist(to_meters(source['uv'][b]), truth['source_xy'], p=1) / (2 * POSITION_SCALE)
                + torch.cdist(source['size'][b], truth['size'], p=1) / (2 * SIZE_SCALE)
                + angle_delta(source['angle'][b, :, None], truth['source_angle'][None, :]).abs() / ANGLE_SCALE)
        if not torch.isfinite(cost).all():
            raise ValueError('nonfinite source matching cost')
        query, gt = linear_sum_assignment(cost.cpu().numpy())
        matches.append((torch.as_tensor(query, device=cost.device), torch.as_tensor(gt, device=cost.device)))
    return matches


class S0Criterion(nn.Module):
    def forward(self, output, supervision):
        source, targets = output['source'], output['targets']
        if len(targets) != 3 or source['uv'].shape[0] != len(supervision):
            raise ValueError('expected three target stages and one label bundle per scene')
        matches = match_source(source, supervision)
        scene_losses = []
        for b, (truth, (query, gt)) in enumerate(zip(supervision, matches)):
            logits = source['objectness_logits'][b]
            positive = torch.zeros_like(logits)
            positive[query] = 1
            weights = torch.where(positive.bool(), 1., NEGATIVE_WEIGHT)
            terms = {'objectness': (F.binary_cross_entropy_with_logits(logits, positive, reduction='none') * weights).sum() / weights.sum(),
                     'source_position': position_loss(to_meters(source['uv'][b, query]), truth['source_xy'][gt]).mean(),
                     'source_size': size_loss(source['size'][b, query], truth['size'][gt]).mean(),
                     'source_angle': orientation_loss(source['angle'][b, query], truth['source_angle'][gt]).mean(),
                     'color': F.cross_entropy(source['color_logits'][b, query], truth['color'][gt]),
                     'shape': F.cross_entropy(source['shape_logits'][b, query], truth['shape'][gt])}
            terms['detection'] = (terms['objectness'] + terms['source_position'] + terms['source_size'] + terms['source_angle']
                                  + .1 * terms['color'] + .1 * terms['shape'])
            for k, stage in enumerate(targets):
                terms[f'target_stage_{k+1}'] = (position_loss(to_meters(stage['uv'][b, query]), truth['target_xy'][gt])
                                               + orientation_loss(stage['angle'][b, query], truth['target_angle'][gt])).mean()
            terms['target'] = sum(w * terms[f'target_stage_{k+1}'] for k, w in enumerate(STAGE_WEIGHTS))
            terms['loss'] = terms['detection'] + terms['target']
            scene_losses.append(terms)
        # Per-scene object means THEN batch mean; N never weights a scene implicitly.
        result = {key: torch.stack([s[key] for s in scene_losses]).mean() for key in scene_losses[0]}
        return result, matches
