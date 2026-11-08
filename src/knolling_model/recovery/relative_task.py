"""Independent split-pose learning with label-only relative supervision."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .split_pose import SplitJointPose
from .joint_layout import joint_pose_loss, POSITION_SCALE

SCHEMA = 'recovery_v4_relative_task_v1'


class TrainableSplitPose(SplitJointPose):
    def __init__(self, split):
        nn.Module.__init__(self)
        # Same tensors and names, no reinitialization or extra representation.
        for name, module in split.named_children(): self.add_module(name, module)

    def train(self, mode=True):
        return nn.Module.train(self, mode)

    def requires_grad_(self, value=True):
        return nn.Module.requires_grad_(self, value)


def attach(model):
    if not isinstance(model.layout.pose, TrainableSplitPose):
        p = model.layout.pose
        if not isinstance(p, SplitJointPose): p = SplitJointPose(p)
        model.layout.pose = TrainableSplitPose(p)


def configure(model, *, xy=True, orientation=True):
    attach(model)
    model.requires_grad_(False)
    groups = {'xy': [], 'orientation': []}
    for name, p in model.layout.pose.named_parameters():
        key = 'orientation' if name.startswith('orientation_pose.') else 'xy'
        p.requires_grad_(xy if key == 'xy' else orientation)
        if p.requires_grad: groups[key].append(p)
    model.eval()  # No dropout/BN drift; eval does not disable autograd.
    return groups


def pair_indices(metadata, pairs, device):
    """ID joins are supervision only; never supplied to neural forward."""
    by_id = {m['sample_id']: i for i, m in enumerate(metadata)}
    if len(by_id) != len(metadata): raise ValueError('duplicate scene')
    rows = []
    for pair in pairs:
        a, b = [by_id[s] for s in pair['sample_ids']]
        ma, mb = metadata[a], metadata[b]
        if ma['record']['scene_family_id'] != mb['record']['scene_family_id']:
            raise ValueError('counterfactual must share source family')
        ia, ib = ma['object_ids'], mb['object_ids']
        if len(set(ia)) != len(ia) or set(ia) != set(ib): raise ValueError('counterfactual object mismatch')
        row = []
        for j, oid in enumerate(ia):
            for k in range(j+1, len(ia)):
                row.append((a, j, k, b, ib.index(oid), ib.index(ia[k])))
        if row: rows.append(torch.tensor(row, dtype=torch.long, device=device))
    return rows


def losses(output, targets, orientation, pairs):
    terms = joint_pose_loss(output, targets, orientation)
    p, y, valid = output['xy'].float(), targets['xy'].float(), output['valid']
    r = p[:, :, None]-p[:, None, :]
    t = y[:, :, None]-y[:, None, :]
    mask = valid[:, :, None] & valid[:, None, :]
    mask = mask & torch.ones(mask.shape[-2:], dtype=torch.bool, device=p.device).triu(1)
    values = F.smooth_l1_loss(r/POSITION_SCALE, t/POSITION_SCALE, reduction='none').sum(-1)
    terms['relative_position_huber'] = (values.masked_fill(~mask, 0).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)).mean()
    contrast = []
    for idx in pairs:
        a,j,k,b,u,v = idx.unbind(-1)
        pred = (p[a,j]-p[a,k])-(p[b,u]-p[b,v])
        target = (y[a,j]-y[a,k])-(y[b,u]-y[b,v])
        contrast.append(F.smooth_l1_loss(pred/POSITION_SCALE, target/POSITION_SCALE, reduction='none').sum(-1).mean())
    terms['counterfactual_relative_huber'] = torch.stack(contrast).mean() if contrast else p.sum()*0
    return terms


def best_key(metrics):
    """Task success first; retain initial state if no evaluated model improves."""
    e = metrics['integrated_evidence']
    return (len(e['successful_scenes']), len(e['successful_pairs']),
            -metrics['validity']['keep_violations'],
            -metrics['target_position_p95_mm'], -metrics['target_angle_p95_deg'])


def restore(model, saved, optimizers=None):
    if saved.get('schema') != SCHEMA: raise ValueError('not a relative-task checkpoint')
    attach(model); model.load_state_dict(saved['model'], strict=True)
    groups = configure(model)
    if optimizers is not None:
        if set(optimizers) != set(saved['optimizers']): raise ValueError('optimizer branch mismatch')
        for k, opt in optimizers.items(): opt.load_state_dict(saved['optimizers'][k])
    return groups
