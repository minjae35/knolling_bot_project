"""Give the sequential offset head the previous and current object sizes.

The set encoder, selector, attention, and the original head inputs stay in
place. Four new input columns are zero, so the widened head matches the
original decoder until those columns change. Sizes use the same 0.6 m / 0.4 m
scale as object features. The head still emits a learned 2D delta. Inference
does not receive a teacher pose, and no edge or centre formula is applied.
"""
import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.experiments.neural_sequential import ScenePoseDecoder

EXTRA = 4
FEATURE_SCALE = (.6, .4)


class SizeConditionedDecoder(ScenePoseDecoder):
    def forward(self, objects, goal, selected, move, order, teacher_xy=None):
        h = self.features(objects, goal, selected)
        b, n, d = h.shape
        rows = torch.arange(b, device=h.device)
        valid = objects.valid
        scale = h.new_tensor(FEATURE_SCALE)
        previous_xy = h.new_tensor([.3, .2]).expand(b, -1)
        previous_h = h.new_zeros(b, d)
        previous_size = h.new_zeros(b, 2)
        scene_xy = objects.xy
        output_xy = objects.xy
        placed = valid & ~move
        for step in range(n):
            active = order[:, step] >= 0
            chosen = order[:, step].clamp_min(0)
            current = h[rows, chosen]
            state = torch.cat((scene_xy / scale, placed[..., None], move[..., None]), -1)
            keys = torch.cat((h + self.state(state), self.null.expand(b, -1, -1)), 1)
            padding = torch.cat((~valid, torch.zeros(b, 1, device=h.device, dtype=torch.bool)), 1)
            context = self.attention((current + self.goal_query(goal))[:, None], keys, keys,
                                     key_padding_mask=padding, need_weights=False)[0][:, 0]
            current_size = torch.where(active[:, None], objects.size[rows, chosen], previous_size.new_zeros(b, 2))
            sizes = torch.cat((previous_size / scale, current_size / scale), -1)
            delta = .1 * self.output(torch.cat((current, context, previous_h, goal, previous_xy / scale, sizes), -1))
            predicted = previous_xy + delta
            where = F.one_hot(chosen, n).bool() & active[:, None]
            output_xy = torch.where(where[..., None], predicted[:, None], output_xy)
            feedback = predicted if teacher_xy is None else teacher_xy[rows, chosen]
            scene_xy = torch.where(where[..., None], feedback[:, None], scene_xy)
            previous_xy = torch.where(active[:, None], feedback, previous_xy)
            previous_h = torch.where(active[:, None], current, previous_h)
            previous_size = torch.where(active[:, None], current_size, previous_size)
            placed = placed | where
        return output_xy.masked_fill(~valid[..., None], 0)


def widen_decoder(decoder):
    """Keep every original column and append four zero size columns."""
    if decoder.output[0].in_features == _wide_width(decoder):
        if not isinstance(decoder, SizeConditionedDecoder):
            decoder.__class__ = SizeConditionedDecoder
        return decoder
    if not isinstance(decoder, ScenePoseDecoder):
        raise TypeError('only the sequential pose decoder can be widened')
    old = decoder.output[0]
    layer = nn.Linear(old.in_features + EXTRA, old.out_features)
    layer = layer.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(old.bias)
        layer.weight[:, :old.in_features].copy_(old.weight)
    decoder.output[0] = layer
    decoder.__class__ = SizeConditionedDecoder
    return decoder


def _wide_width(decoder):
    return 3 * decoder.features.input.out_features + 15 + EXTRA
