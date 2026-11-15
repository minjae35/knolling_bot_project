"""Typed predicted goals -> single-stage absolute joint poses, no planner."""
import torch
from torch import nn

from .goal_encoder import GoalEncoder
from .layout import PeriodicBins

ARCHITECTURE = 'recovery_v4_goal_joint'


class GoalSetBlock(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dimension), nn.LayerNorm(dimension)])
        self.film = nn.Linear(13, 2*dimension)
        nn.init.xavier_uniform_(self.film.weight, gain=.1)
        nn.init.zeros_(self.film.bias)
        self.ffn = nn.Sequential(nn.Linear(dimension, 4*dimension), nn.GELU(), nn.Linear(4*dimension, dimension))
        self.null = nn.Parameter(torch.zeros(1, 1, dimension))

    def forward(self, hidden, goal, valid):
        q = self.norms[0](hidden)
        keys = torch.cat((q, self.null.expand(len(q), -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros(len(q), 1, dtype=torch.bool, device=q.device)), 1)
        hidden = hidden+self.attention(q, keys, keys, key_padding_mask=padding, need_weights=False)[0]
        gamma, beta = self.film(goal).chunk(2, -1)
        hidden = hidden+self.ffn((1+gamma[:, None])*self.norms[1](hidden)+beta[:, None])
        return hidden.masked_fill(~valid[..., None], 0)


class AbsoluteJointPose(nn.Module):
    def __init__(self, dimension=128, heads=4):
        super().__init__()
        self.input = nn.Linear(13, dimension)
        self.blocks = nn.ModuleList(GoalSetBlock(dimension, heads) for _ in range(2))
        self.output = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 38))
        self.orientation = PeriodicBins(18)

    def forward(self, objects, goal, selected):
        hidden = self.input(torch.cat((objects.features(), selected[..., None]), -1))
        for block in self.blocks:
            hidden = block(hidden, goal, objects.valid)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            raw, logits, residual = self.output(hidden.float()).split((2, 18, 18), -1)
            xy = (raw*raw.new_tensor([.3, .2])+raw.new_tensor([.3, .2])).masked_fill(~objects.valid[..., None], 0)
            logits = logits.masked_fill(~objects.valid[..., None], 0)
            residual = residual.masked_fill(~objects.valid[..., None], 0)
            angle = self.orientation.decode(logits, residual).masked_fill(~objects.valid, 0)
        return dict(xy=xy, angle=angle, angle_logits=logits, angle_residual=residual,
                    valid=objects.valid, query_index=objects.query_index,
                    architecture=ARCHITECTURE, rollout='free_rollout')


class GoalJointLayout(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, dimension=128, heads=4, layers=2, bins=18):
        super().__init__()
        if layers != 2 or bins != 18 or dimension % heads:
            raise ValueError('v4 requires two independent set blocks, 18 bins and valid attention dimensions')
        self.goal_encoder = GoalEncoder(dimension, heads)
        self.pose = AbsoluteJointPose(dimension, heads)

    @property
    def orientation(self):
        return self.pose.orientation

    def predict_layout(self, objects, text, text_mask):
        # Do not use no_grad here: frozen grounding still passes input gradients.
        goal = self.goal_encoder(objects, text, text_mask)
        return dict(goal=goal, layout=self.pose(objects, goal['vector'], goal['selected']))

    def forward(self, objects, text, text_mask):
        result = self.predict_layout(objects, text, text_mask)
        return dict(result['layout'], goal=result['goal'])
