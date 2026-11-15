"""Seven language task slots and a permutation-equivariant referent grounder."""
import torch
from torch import nn
from torch.nn import functional as F

from .goal_labels import VOCAB
from .joint_layout import JointBlock


class GoalEncoder(nn.Module):
    def __init__(self, dimension=128, heads=4):
        super().__init__()
        self.text_projection = nn.Linear(384, dimension)
        self.queries = nn.Parameter(torch.randn(1, 7, dimension)*.02)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dimension), nn.LayerNorm(dimension)])
        self.ffn = nn.Sequential(nn.Linear(dimension, 4*dimension), nn.GELU(), nn.Linear(4*dimension, dimension))
        self.keep = nn.Linear(dimension, 1)
        self.fields = nn.ModuleDict({k: nn.Linear(dimension, len(v)) for k, v in VOCAB.items()})
        self.objects = nn.Linear(12, dimension)
        self.grounder = JointBlock(dimension, heads)
        self.selection = nn.Linear(dimension, 1)

    def forward(self, objects, text, text_mask):
        if (text.ndim != 3 or text.shape[:2] != text_mask.shape or len(text) != len(objects.valid)
                or text_mask.dtype != torch.bool or not text_mask.any(-1).all()):
            raise ValueError('invalid language memory')
        memory = self.text_projection(text).masked_fill(~text_mask[..., None], 0)
        q = self.queries.expand(len(text), -1, -1)
        slots = self.norms[0](q+self.attention(q, memory, memory, key_padding_mask=~text_mask, need_weights=False)[0])
        slots = self.norms[1](slots+self.ffn(slots))
        keep = self.keep(slots[:, 0]).squeeze(-1)
        logits = {k: self.fields[k](slots[:, i+1]) for i, k in enumerate(VOCAB)}
        mask = torch.ones(slots.shape[:2], dtype=torch.bool, device=slots.device)
        hidden = self.grounder(self.objects(objects.features()), slots, objects.valid, mask)
        selected = self.selection(hidden).squeeze(-1).masked_fill(~objects.valid, 0)
        probability = keep.sigmoid()[:, None]
        goal = torch.cat([probability]+[(1-probability)*logits[k].softmax(-1) for k in VOCAB], -1)
        return dict(keep_logit=keep, logits=logits, selected_logit=selected,
                    selected=selected.sigmoid().masked_fill(~objects.valid, 0), vector=goal, valid=objects.valid)


def goal_loss(output, truths, gt_indices):
    """Three equally weighted groups; five attribute CEs share one group."""
    valid = output['valid']
    with torch.autocast(device_type=valid.device.type, enabled=False):
        keep = F.binary_cross_entropy_with_logits(output['keep_logit'].float(), torch.stack([t['goal_keep'] for t in truths]))
        fields = []
        for key in VOCAB:
            labels = torch.stack([t['goal_'+key] for t in truths])
            active = labels != -100
            ce = F.cross_entropy(output['logits'][key].float(), labels, ignore_index=-100, reduction='none')
            fields.append(ce.sum()/active.sum().clamp_min(1))
        labels = torch.stack([F.pad(t['goal_selected'][idx], (0, valid.shape[1]-len(idx))) for t, idx in zip(truths, gt_indices)])
        selection = F.binary_cross_entropy_with_logits(output['selected_logit'].float(), labels.float(), reduction='none')
        selection = (selection.masked_fill(~valid, 0).sum(-1)/valid.sum(-1).clamp_min(1)).mean()
    return dict(keep_bce=keep, attributes_ce=sum(fields)/len(fields), selected_bce=selection)
