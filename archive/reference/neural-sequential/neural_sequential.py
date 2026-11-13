"""Predicted object permutation and scene-conditioned neural 2D placements.

No target inputs in forward. Teacher sequences/prefixes are diagnostic or
supervision only. Selector and pose encoders have disjoint trainable weights.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from knolling_dataset.builder import file_hash
from knolling_model.recovery.storage import load_artifact
from .generalization import new_model
from .phase1_single_row import position_loss

ARCHITECTURE = 'neural_sequential_xy_v1'


class SetFeatures(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.input = deepcopy(reference.input).requires_grad_(True)
        self.blocks = deepcopy(reference.blocks).requires_grad_(True)

    def forward(self, objects, goal, selected):
        h = self.input(torch.cat((objects.features(), selected[..., None]), -1))
        for block in self.blocks:
            h = block(h, goal, objects.valid)
        return h


class ObjectSelector(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.features = SetFeatures(reference)
        d = reference.input.out_features
        self.key = nn.Linear(d, d, bias=False)
        self.query = nn.Sequential(nn.Linear(2*d+13, d), nn.GELU(), nn.Linear(d, d))
        self.begin = nn.Parameter(torch.zeros(d))

    def forward(self, objects, goal, selected, move, teacher_order=None):
        h = self.features(objects, goal, selected)
        b, n, d = h.shape
        previous = self.begin.expand(b, -1)
        available = move & objects.valid
        keys = self.key(h)
        logits, order = [], []
        rows = torch.arange(b, device=h.device)
        for step in range(n):
            active = available.any(-1)
            context = (h*available[..., None]).sum(1)/available.sum(-1, keepdim=True).clamp_min(1)
            query = self.query(torch.cat((context, previous, goal), -1))
            scores = (keys*query[:, None]).sum(-1)/(d**.5)
            scores = scores.masked_fill(~available, -1e9)
            if teacher_order is None:
                # Stable detector/source identities break exact logit ties only.
                ties = (scores == scores.max(-1, keepdim=True).values) & available
                chosen = objects.query_index.masked_fill(~ties, torch.iinfo(torch.long).max).argmin(-1)
            else:
                chosen = teacher_order[:, step].clamp_min(0)
                if not bool((available[rows, chosen] | ~active).all()):
                    raise ValueError('teacher order repeats or selects an unavailable object')
                if not torch.equal(teacher_order[:, step]>=0, active):
                    raise ValueError('teacher sequence has incorrect coverage')
            logits.append(scores)
            order.append(chosen.masked_fill(~active, -1))
            previous = torch.where(active[:, None], h[rows, chosen], previous)
            available = available & ~(F.one_hot(chosen, n).bool() & active[:, None])
        return torch.stack(order, 1), torch.stack(logits, 1)


class ScenePoseDecoder(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.features = SetFeatures(reference)
        d = reference.input.out_features
        self.state = nn.Linear(4, d)
        self.goal_query = nn.Linear(13, d)
        self.attention = nn.MultiheadAttention(d, 4, dropout=0., batch_first=True)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.output = nn.Sequential(nn.Linear(3*d+15, d), nn.GELU(), nn.Linear(d, d),
                                    nn.GELU(), nn.Linear(d, 2))
        nn.init.xavier_uniform_(self.output[-1].weight, gain=.01)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, objects, goal, selected, move, order, teacher_xy=None):
        h = self.features(objects, goal, selected)
        b, n, d = h.shape
        rows = torch.arange(b, device=h.device)
        valid = objects.valid
        scale = h.new_tensor([.6, .4])
        previous_xy = h.new_tensor([.3, .2]).expand(b, -1)
        previous_h = h.new_zeros(b, d)
        scene_xy = objects.xy
        output_xy = objects.xy
        placed = valid & ~move
        for step in range(n):
            active = order[:, step]>=0
            chosen = order[:, step].clamp_min(0)
            current = h[rows, chosen]
            state = torch.cat((scene_xy/scale, placed[..., None], move[..., None]), -1)
            keys = torch.cat((h+self.state(state), self.null.expand(b, -1, -1)), 1)
            padding = torch.cat((~valid, torch.zeros(b, 1, device=h.device, dtype=torch.bool)), 1)
            context = self.attention((current+self.goal_query(goal))[:, None], keys, keys,
                                     key_padding_mask=padding, need_weights=False)[0][:, 0]
            delta = .1*self.output(torch.cat((current, context, previous_h, goal, previous_xy/scale), -1))
            predicted = previous_xy+delta
            where = F.one_hot(chosen, n).bool() & active[:, None]
            output_xy = torch.where(where[..., None], predicted[:, None], output_xy)
            feedback = predicted if teacher_xy is None else teacher_xy[rows, chosen]
            scene_xy = torch.where(where[..., None], feedback[:, None], scene_xy)
            previous_xy = torch.where(active[:, None], feedback, previous_xy)
            previous_h = torch.where(active[:, None], current, previous_h)
            placed = placed | where
        return output_xy.masked_fill(~valid[..., None], 0)


class SequentialPose(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference.requires_grad_(False)
        self.selector = ObjectSelector(reference)
        self.decoder = ScenePoseDecoder(reference)
        self.orientation = reference.orientation

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, objects, goal, selected):
        with torch.no_grad():
            base = self.reference(objects, goal, selected)
        order, _ = self.selector(objects, goal, selected, base['move_mask'])
        xy = self.decoder(objects, goal, selected, base['move_mask'], order)
        return dict(base, xy=xy, proposal_xy=xy, predicted_order=order,
                    experiment_architecture=ARCHITECTURE)

    def selector_loss(self, objects, goal, selected, move, order):
        _, logits = self.selector(objects, goal, selected, move, teacher_order=order)
        labels = order.masked_fill(order<0, -100)
        terms = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='none')
        return terms.sum()/(labels>=0).sum().clamp_min(1)

    @torch.no_grad()
    def diagnose(self, objects, goal, selected, order, targets=None):
        if self.training:
            raise ValueError('teacher-prefix diagnosis is evaluation only')
        base = self.reference(objects, goal, selected)
        xy = self.decoder(objects, goal, selected, base['move_mask'], order, teacher_xy=targets)
        return dict(base, xy=xy, proposal_xy=xy, predicted_order=order,
                    diagnostic='teacher_prefix' if targets is not None else 'teacher_order')


def teacher_order(records, object_ids, n, device):
    """Read existing Teacher labels, never called by prediction."""
    order = torch.full((len(records), n), -1, device=device, dtype=torch.long)
    for b, (record, ids) in enumerate(zip(records, object_ids)):
        moved = [oid for row in record['supervision_v3']['rows'] for oid in row
                 if not record['supervision_v3']['copy_required'][oid]]
        if len(moved)!=len(set(moved)) or set(moved)!={oid for oid in ids if not record['supervision_v3']['copy_required'][oid]}:
            raise ValueError('incomplete or duplicate teacher sequence')
        order[b, :len(moved)] = torch.tensor([ids.index(oid) for oid in moved], device=device)
    return order


def build(config, device='cpu'):
    if file_hash(config['initial_checkpoint'])!=config['initial_sha256']:
        raise ValueError('initial checkpoint integrity failure')
    model = new_model(config, device)
    model.arrangement.load_state_dict(load_artifact(config['initial_checkpoint'])['model'], strict=True)
    model.requires_grad_(False)
    model.arrangement.layout.pose = SequentialPose(model.layout.pose).to(device)
    return model.eval()
