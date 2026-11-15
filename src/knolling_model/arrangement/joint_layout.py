"""Permutation-equivariant, label-free joint poses with two shared refinements."""
import torch
from torch import nn
from torch.nn import functional as F

from .layout import PeriodicBins

ARCHITECTURE = 'recovery_v3_joint'
POSITION_SCALE = .010


class JointBlock(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dimension) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dimension, 4*dimension), nn.GELU(), nn.Linear(4*dimension, dimension))
        self.null = nn.Parameter(torch.zeros(1, 1, dimension))

    def forward(self, hidden, text, valid, text_mask):
        # An always-valid null key makes empty detections finite without GT count.
        query = self.norms[0](hidden)
        keys = torch.cat((query, self.null.expand(len(hidden), -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros(len(hidden), 1, dtype=torch.bool, device=valid.device)), 1)
        hidden = hidden+self.self_attention(query, keys, keys, key_padding_mask=padding, need_weights=False)[0]
        query = self.norms[1](hidden)
        hidden = hidden+self.cross_attention(query, text, text, key_padding_mask=~text_mask, need_weights=False)[0]
        return (hidden+self.ffn(self.norms[2](hidden))).masked_fill(~valid[..., None], 0)


class JointPoseDecoder(nn.Module):
    architecture = ARCHITECTURE
    refinement_steps = 2

    def __init__(self, dimension=128, heads=4, layers=2, bins=18):
        super().__init__()
        if bins != 18 or layers < 1 or dimension % heads:
            raise ValueError('v3 requires 18 periodic bins and valid attention dimensions')
        self.object_projection = nn.Linear(12, dimension)
        self.text_projection = nn.Linear(384, dimension)
        self.encoder = nn.ModuleList(JointBlock(dimension, heads) for _ in range(layers))
        self.initial = nn.Linear(dimension, 2+2*bins)
        self.state_projection = nn.Linear(2+2*bins, dimension)
        self.refinement = JointBlock(dimension, heads)
        self.update = nn.Linear(dimension, 2+2*bins)
        self.orientation = PeriodicBins(bins)

    def forward(self, objects, text, text_mask):
        valid = objects.valid
        if (text.ndim != 3 or text.shape[:2] != text_mask.shape or len(text) != len(valid)
                or text_mask.dtype != torch.bool or not text_mask.any(-1).all()):
            raise ValueError('invalid language memory')
        hidden = self.object_projection(objects.features())
        memory = self.text_projection(text).masked_fill(~text_mask[..., None], 0)
        for block in self.encoder:
            hidden = block(hidden, memory, valid, text_mask)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            z, logits, residual = self.initial(hidden.float()).split((2, 18, 18), -1)
        states = []
        for step in range(self.refinement_steps+1):
            z = z.masked_fill(~valid[..., None], 0)
            logits = logits.masked_fill(~valid[..., None], 0)
            residual = residual.masked_fill(~valid[..., None], 0)
            states.append(dict(displacement=z, angle_logits=logits, angle_residual=residual))
            if step == self.refinement_steps:
                break
            # No argmax, threshold, target prefix or detach in recurrent state.
            state = torch.cat((z, logits.softmax(-1), residual.tanh()), -1)
            refined = self.refinement(hidden+self.state_projection(state), memory, valid, text_mask)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                dz, dl, dr = self.update(refined.float()).split((2, 18, 18), -1)
                z, logits, residual = z+5*dz.tanh(), logits+dl, residual+dr
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            xy = (objects.xy.float()+POSITION_SCALE*z).masked_fill(~valid[..., None], 0)
            angle = self.orientation.decode(logits, residual).masked_fill(~valid, 0)
        return dict(xy=xy, angle=angle, angle_logits=logits, angle_residual=residual,
                    valid=valid, query_index=objects.query_index, states=tuple(states),
                    architecture=ARCHITECTURE, rollout='free_rollout')


def joint_pose_loss(output, targets, orientation):
    """Three final-only terms, scene means, all valid objects including keep."""
    if output.get('architecture') not in (ARCHITECTURE, 'recovery_v4_goal_joint') or output.get('rollout') != 'free_rollout':
        raise ValueError('pose loss requires label-free joint predictions')
    valid = output['valid']
    def mean(value):
        return (value.masked_fill(~valid, 0).sum(-1)/valid.sum(-1).clamp_min(1)).mean()
    with torch.autocast(device_type=output['xy'].device.type, enabled=False):
        position = F.smooth_l1_loss(output['xy'].float()/POSITION_SCALE,
                                   targets['xy'].float()/POSITION_SCALE, reduction='none').sum(-1)
        index, target_residual = orientation.encode(targets['angle'].float())
        logits = output['angle_logits'].float()
        classification = F.cross_entropy(logits.flatten(0, 1), index.flatten(), reduction='none').reshape_as(valid)
        predicted = output['angle_residual'].float().gather(-1, index[..., None]).squeeze(-1).tanh()
        residual = F.smooth_l1_loss(predicted, target_residual, reduction='none')
    return dict(position_huber=mean(position), angle_ce=mean(classification), angle_residual_huber=mean(residual))
