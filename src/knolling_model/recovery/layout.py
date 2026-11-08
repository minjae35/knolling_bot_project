"""Object-centric language Transformer with autoregressive planar pose decoding."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.coordinates import angle_delta, wrap_pi


class PeriodicBins(nn.Module):
    """Non-overlapping pi-periodic bins plus a bounded within-bin residual.

    Hard bin choice avoids averaging horizontal/vertical modes. Cross-entropy
    supervises the discrete choice; residual gradients use the true bin only.
    """
    def __init__(self, bins=18):
        super().__init__()
        if bins < 2:
            raise ValueError('at least two orientation bins required')
        self.bins = bins
        self.width = math.pi/bins
        self.register_buffer('centers', (-math.pi/2 + torch.arange(bins, dtype=torch.float64)*self.width).float())

    def encode(self, angle):
        wrapped = wrap_pi(angle)
        index = ((wrapped+math.pi/2)/self.width+.5).floor().long().remainder(self.bins)
        residual = angle_delta(wrapped, self.centers[index])/(self.width/2)
        return index, residual

    def decode(self, logits, residual):
        index = logits.argmax(-1)
        offset = residual.gather(-1, index[..., None]).squeeze(-1).tanh()*self.width/2
        return wrap_pi(self.centers[index]+offset)

    def loss(self, logits, residual, angle):
        index, offset = self.encode(angle)
        classification = F.cross_entropy(logits.reshape(-1, self.bins), index.reshape(-1), reduction='none')
        predicted = residual.gather(-1, index[..., None]).squeeze(-1).tanh()
        regression = F.smooth_l1_loss(predicted, offset, reduction='none')
        return classification.reshape_as(angle) + regression


def pose_features(xy, angle):
    return torch.cat((xy/xy.new_tensor([.6, .4]),
                      torch.stack(((2*angle).cos(), (2*angle).sin()), -1)), -1)


class AutoregressiveLayout(nn.Module):
    def __init__(self, dimension=128, heads=4, layers=2, bins=18):
        super().__init__()
        self.object_projection = nn.Linear(12, dimension)
        self.text_projection = nn.Linear(384, dimension)
        self.null_memory = nn.Parameter(torch.randn(1, 1, dimension)*.02)
        self.begin = nn.Parameter(torch.randn(1, 1, dimension)*.02)
        self.previous_pose = nn.Linear(4, dimension)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            dimension, heads, dimension*4, dropout=0., batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            dimension, heads, dimension*4, dropout=0., batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(dimension)
        self.xy = nn.Linear(dimension, 2)
        self.orientation = PeriodicBins(bins)
        self.bin_logits = nn.Linear(dimension, bins)
        self.bin_residual = nn.Linear(dimension, bins)
        self.move = nn.Linear(dimension, 1)
        self.selection = nn.Linear(dimension, 1)
        self.axis = nn.Linear(dimension, 2)
        self.region = nn.Linear(dimension, 3)

    def _memory(self, objects, text, text_mask):
        features = objects.features()
        b, n, _ = features.shape
        if text.ndim != 3 or text.shape[:2] != text_mask.shape or text.shape[0] != b:
            raise ValueError('invalid text memory shape')
        if text_mask.dtype != torch.bool or not text_mask.any(-1).all():
            raise ValueError('each instruction requires at least one text token')
        tokens = self.object_projection(features)
        padding = torch.cat((~objects.valid, ~text_mask,
                             torch.zeros(b, 1, dtype=torch.bool, device=text.device)), 1)
        memory = torch.cat((tokens, self.text_projection(text), self.null_memory.expand(b, -1, -1)), 1)
        memory = self.encoder(memory, src_key_padding_mask=padding)
        text_states = memory[:, n:n+text.shape[1]]
        pooled = (text_states*text_mask[..., None]).sum(1)/text_mask.sum(1, keepdim=True)
        return memory, padding, memory[:, :n], pooled

    def _decode(self, object_states, previous, memory, padding, valid):
        n = object_states.shape[1]
        # BOS is always unmasked, including an image with zero predicted objects.
        prefix_valid = torch.cat((torch.ones_like(valid[:, :1]), valid[:, :-1]), 1)
        causal = torch.ones(n, n, dtype=torch.bool, device=memory.device).triu(1)
        hidden = self.decoder(object_states+previous, memory, tgt_mask=causal,
                              tgt_key_padding_mask=~prefix_valid, memory_key_padding_mask=padding)
        return self.norm(hidden)

    def _heads(self, hidden, objects, upto):
        candidate_xy = self.xy(hidden).sigmoid()*hidden.new_tensor([.6, .4])
        logits, residual = self.bin_logits(hidden), self.bin_residual(hidden)
        candidate_angle = self.orientation.decode(logits, residual)
        move_logits = self.move(hidden).squeeze(-1)
        move = (move_logits >= 0) & objects.valid[:, :upto]
        return dict(candidate_xy=candidate_xy, candidate_angle=candidate_angle,
                    angle_logits=logits, angle_residual=residual, move_logits=move_logits,
                    selected_logits=self.selection(hidden).squeeze(-1),
                    xy=torch.where(move[..., None], candidate_xy, objects.xy[:, :upto]),
                    angle=torch.where(move, candidate_angle, objects.angle[:, :upto]))

    def forward(self, objects, text, text_mask):
        """Free rollout only: no supervision, teacher or target-prefix argument."""
        memory, padding, states, pooled = self._memory(objects, text, text_mask)
        b, n = objects.valid.shape
        previous = self.begin.expand(b, 1, -1)
        results = []
        for k in range(n):
            hidden = self._decode(states[:, :k+1], previous, memory, padding, objects.valid[:, :k+1])
            current = self._heads(hidden, objects, k+1)
            # Preserve already emitted poses; never revise a prefix using future targets.
            results.append({key: value[:, -1:] for key, value in current.items()})
            feedback = self.previous_pose(pose_features(current['xy'][:, -1:], current['angle'][:, -1:]))
            previous = torch.cat((previous, feedback), 1)
        output = {key: torch.cat([r[key] for r in results], 1) for key in results[0]}
        return self._finish(output, objects, pooled, 'free_rollout')

    def teacher_forced(self, objects, text, text_mask, target_xy, target_angle):
        """Explicit training-only diagnostic; future target labels are causally masked."""
        if not self.training:
            raise ValueError('teacher forcing is forbidden in evaluation')
        if target_xy.shape != objects.xy.shape or target_angle.shape != objects.angle.shape:
            raise ValueError('target prefix must use the source object order')
        memory, padding, states, pooled = self._memory(objects, text, text_mask)
        previous = torch.cat((self.begin.expand(len(text), 1, -1),
                              self.previous_pose(pose_features(target_xy[:, :-1], target_angle[:, :-1]))), 1)
        hidden = self._decode(states, previous, memory, padding, objects.valid)
        return self._finish(self._heads(hidden, objects, objects.valid.shape[1]), objects, pooled, 'teacher_forced')

    def _finish(self, output, objects, pooled, mode):
        return dict(**output, valid=objects.valid, query_index=objects.query_index,
                    axis_logits=self.axis(pooled), region_logits=self.region(pooled), rollout=mode)


def layout_loss(output, targets, orientation):
    """Label-aligned loss; padded rows and undefined no-op command fields ignored."""
    valid = output['valid']
    count = valid.sum().clamp_min(1)
    moved = valid & targets['move']
    def masked(values, mask):
        return values.masked_fill(~mask, 0).sum()/mask.sum().clamp_min(1)
    position = F.smooth_l1_loss(output['candidate_xy']/.01, targets['xy']/.01, reduction='none').sum(-1)
    angles = orientation.loss(output['angle_logits'], output['angle_residual'], targets['angle'])
    losses = dict(position=masked(position, moved), orientation=masked(angles, moved))
    for head, key in (('move_logits', 'move'), ('selected_logits', 'selected')):
        values = F.binary_cross_entropy_with_logits(output[head], targets[key].float(), reduction='none')
        losses[key] = values.masked_fill(~valid, 0).sum()/count
    for key in ('axis', 'region'):
        labels = targets[key]
        defined = labels != -100
        # reduction=mean on an all-ignored batch would produce NaN.
        values = F.cross_entropy(output[key+'_logits'], labels, ignore_index=-100, reduction='none')
        losses[key] = .1*masked(values, defined)
    return losses
