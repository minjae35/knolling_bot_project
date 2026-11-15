"""Structure-centred Arrangement and RGB-ROI-conditioned RT-DETR candidates.

StructFormer-inspired decomposition, not a reproduction of its point-cloud AR
model. Both paths use only observed inputs and predicted goals at inference.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from knolling_model.arrangement.goal_joint_layout import GoalSetBlock
from knolling_model.perception.model import ObjectPerception


class StructurePose(nn.Module):
    """Shared structure token + centred local offsets; frozen accepted baseline.

    All new outputs start at zero, exactly retaining the accepted baseline.
    Mean removal makes centre and local displacement identifiable. No ordering,
    object IDs, target geometry, labels, or rule-based layout enters forward.
    """
    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline.requires_grad_(False)
        d = baseline.input.out_features
        self.input = deepcopy(baseline.input).requires_grad_(True)
        self.blocks = deepcopy(baseline.blocks).requires_grad_(True)
        self.structure = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d))
        self.center = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.local = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.routing = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        for head in (self.center, self.local, self.routing):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        self.orientation = baseline.orientation

    def forward(self, objects, goal, selected):
        base = self.baseline(objects, goal, selected)
        valid = objects.valid
        weight = (selected*valid).unsqueeze(-1)
        source_center = (objects.xy*weight).sum(1)/weight.sum(1).clamp_min(1e-6)
        features = objects.features().clone()
        features[..., :2] = (objects.xy-source_center[:, None])/objects.xy.new_tensor([.6, .4])
        hidden = self.input(torch.cat((features, selected[..., None]), -1))
        token = self.structure(torch.cat((goal, source_center/objects.xy.new_tensor([.6, .4])), -1))
        hidden = torch.cat((token[:, None], hidden), 1)
        mask = torch.cat((torch.ones(len(valid), 1, device=valid.device, dtype=torch.bool), valid), 1)
        for block in self.blocks: hidden = block(hidden, goal, mask)
        delta = self.local(hidden[:, 1:]).masked_fill(~valid[..., None], 0)
        delta = delta-(delta.sum(1)/valid.sum(1).clamp_min(1)[:, None])[:, None]
        proposal = base['proposal_xy']+.1*(self.center(hidden[:, 0])[:, None]+delta)
        logits = base['move_logit']+self.routing(hidden[:, 1:]).squeeze(-1)
        move = (logits >= 0) & valid
        return dict(base, xy=torch.where(move[..., None], proposal, objects.xy).masked_fill(~valid[..., None], 0),
            angle=torch.where(move, base['proposal_angle'], objects.angle).masked_fill(~valid, 0),
            proposal_xy=proposal.masked_fill(~valid[..., None], 0), move_logit=logits.masked_fill(~valid, 0),
            move_mask=move, experiment_architecture='structure_center_relative_v1')


class AppearancePerception(ObjectPerception):
    """Learned local RGB evidence for attributes and foreground confidence.

    Grid positions come exclusively from predicted boxes. Grid detachment avoids
    nondeterministic CUDA grid_sample backward while preserving RGB-CNN gradients.
    Native detector box supervision still trains geometry end to end.
    """
    @classmethod
    def from_baseline(cls, model):
        # Preserve every pre-trained tensor; no detector/head reinitialization.
        model.__class__ = cls
        model.roi_encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model.roi_head = nn.Sequential(nn.Linear(256+32, 128), nn.GELU(), nn.Linear(128, 10))
        nn.init.zeros_(model.roi_head[-1].weight); nn.init.zeros_(model.roi_head[-1].bias)
        return model

    def _forward(self, images, labels):
        out = super()._forward(images, labels)
        boxes = out['stages'][-1]['boxes'].detach()
        b, n, _ = boxes.shape
        axis = torch.linspace(-.5, .5, 16, device=images.device, dtype=images.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = boxes[:, :, None, None, :2]+torch.stack((xx, yy), -1)*boxes[:, :, None, None, 2:]
        # One tall output grid per image avoids 300 replicated full RGB images.
        crop = F.grid_sample(images.detach(), grid.reshape(b, n*16, 16, 2)*2-1,
                             mode='bilinear', padding_mode='zeros', align_corners=False)
        crop = crop.reshape(b, 3, n, 16, 16).permute(0, 2, 1, 3, 4).reshape(b*n, 3, 16, 16)
        features = self.roi_encoder(crop).reshape(b, n, -1)
        delta = self.roi_head(torch.cat((out['stages'][-1]['tokens'], features), -1))
        source = out['source']
        ab = source['size']*delta[..., :2].clamp(-3, 3).exp()
        source['size'] = torch.stack((ab.max(-1).values, ab.min(-1).values), -1)
        source['angle'] = source['angle']+delta[..., 2]
        source['color_logits'] = source['color_logits']+delta[..., 3:6]
        source['shape_logits'] = source['shape_logits']+delta[..., 6:9]
        out['stages'][-1]['logits'] = out['stages'][-1]['logits']+delta[..., 9:10]
        source['objectness_logits'] = out['stages'][-1]['logits'].squeeze(-1)
        return out
