"""Explicit, hash-pinned Coarse continuation; never relax ordinary Resume."""
from copy import deepcopy
import math
import re

import torch
from torch.nn import functional as F

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from .storage import load_artifact, tensor_digest
from .provenance import contract_hashes

SCHEMA = 'coarse_extension_v1'


def validate_policy(config):
    policy = config.get('coarse_extension')
    if policy is None:
        return
    if (set(policy) != {'schema', 'source_sha256', 'interval_updates', 'min_relative_improvement', 'patience'}
            or policy['schema'] != SCHEMA
            or not re.fullmatch('[0-9a-f]{64}', policy.get('source_sha256', ''))
            or policy['interval_updates'] != 200 or policy['min_relative_improvement'] != .05
            or policy['patience'] != 2
            or config['stages']['oracle_layout'] != {'max_updates': 1300, 'max_seconds': 600}
            or config['integrated_pose']['xy_solver'] != 'svd_huber_v1'
            or config['integrated_pose']['stages']['oracle_layout'] != dict(
                coarse=1000, head_xy=50, head_angle=50, fine=100, final_xy=50, final_angle=50)):
        raise ValueError('invalid bounded Coarse extension policy')


def load_extension(path, config, fingerprint, receipt):
    from .driver import load_checkpoint, validate_micro_checkpoints
    validate_policy(config)
    policy = config.get('coarse_extension')
    grant = (receipt or {}).get('coarse_extension', {})
    if (not policy or grant.get('approved') is not True
            or file_hash(path) != policy['source_sha256'] or grant.get('sha256') != policy['source_sha256']):
        raise PermissionError('Coarse continuation requires explicit source approval')
    raw = load_artifact(path)
    old = raw.get('config', {})
    if (raw.get('commit') != grant.get('commit') or digest(old) != grant.get('config_sha256')
            or grant.get('contracts') != contract_hashes()
            or grant['contracts'] != contract_hashes(grant['commit'])):
        raise ValueError('Coarse source semantic contract mismatch')
    saved = load_checkpoint(path, old, fingerprint, grant['commit'])
    # Only the requested budget and monitor change. No architecture/optimization edits.
    normalized = deepcopy(config)
    normalized.pop('coarse_extension')
    for key in ('execution', 'allow_training', 'budget_status'):
        if key in old: normalized[key] = old[key]
        else: normalized.pop(key, None)
    normalized['stages']['oracle_layout'] = old['stages']['oracle_layout']
    normalized['integrated_pose']['stages']['oracle_layout']['coarse'] = 200
    if (normalized != old or old['integrated_pose']['stages']['oracle_layout']['coarse'] != 200
            or old.get('coarse_extension') is not None):
        raise ValueError('Coarse extension changed an unapproved training setting')
    state = saved.get('integrated', {})
    phases = state.get('phase_results', [])
    if (saved.get('stage') != 'oracle_layout' or saved.get('status') != 'running'
            or saved.get('micro_case') is not None or saved['cursor']['update'] != 200
            or state.get('phase') != 'head_xy' or state.get('phase_update') != 0
            or state.get('fine_resets') != 0 or state.get('closure_evaluations') != 0
            or any(state.get(k) is not None for k in ('solver', 'cache_path', 'cache_key', 'pending_head'))
            or len(phases) != 1 or phases[0]['phase'] != 'coarse' or phases[0]['updates'] != 200
            or not math.isfinite(saved['cursor']['elapsed_seconds']) or not 0 <= saved['cursor']['elapsed_seconds'] < 600):
        raise ValueError('source is not the uncalibrated 200-update Coarse boundary')
    if file_hash(state['anchor_path']) != saved['anchor_sha256']:
        raise ValueError('Coarse source anchor hash mismatch')
    if tensor_digest(load_artifact(state['anchor_path'])) != tensor_digest(saved['model']):
        raise ValueError('Coarse source model differs from its boundary anchor')
    optimizer = saved['optimizer']
    if (not optimizer['optimizer']['state'] or any(float(s['step']) != 200 for s in optimizer['optimizer']['state'].values())
            or any(g['lr'] != 1e-4 for g in optimizer['optimizer']['param_groups'])):
        raise ValueError('Coarse source Adam state mismatch')
    from .readout import SLICES
    for kind, sl in SLICES.items():
        for name in ('weight', 'bias'):
            if not torch.equal(optimizer['leaves'][kind+'.'+name], saved['model']['layout.pose.output.1.'+name][sl]):
                raise ValueError('Coarse source optimizer/head mismatch')
    evidence = saved.get('micro_evidence', {})
    validated = validate_micro_checkpoints([v['path'] for v in evidence.values()], saved['goal_parent_sha256'],
        old, fingerprint, saved['commit'], saved.get('review_receipt'))
    if any(validated[k]['sha256'] != evidence[k]['sha256'] for k in validated):
        raise ValueError('Coarse source Micro evidence mismatch')
    # In-memory transition only; preserve model, Adam, RNG, sample cursor and elapsed budget.
    state.update(phase='coarse', phase_update=200, phase_seconds=phases[0]['elapsed_seconds'],
        phase_results=[], head_diagnostic=None, coarse_progress=[],
        coarse_extension_source=dict(path=str(path), sha256=policy['source_sha256'], commit=saved['commit'],
                                     source_cursor=deepcopy(saved['cursor'])))
    state['timing']['phase'] = deepcopy(phases[0]['timing'])
    return saved


@torch.no_grad()
def readout_diagnostic(model, data):
    """FP64 linear lower bound and unchanged losses; no head fitting or mutation."""
    x = torch.cat((data.features.cpu().double(), torch.ones(len(data.features), 1, dtype=torch.float64)), 1)
    weights = data.weights.cpu().double()
    a = weights.sqrt()[:, None]*x
    u, s, _ = torch.linalg.svd(a, full_matrices=False)
    rank = int((s > s[0]*max(a.shape)*torch.finfo(torch.float32).eps).sum())
    y = weights.sqrt()[:, None]*data.xy.cpu().double()
    rms = float((y-u[:, :rank]@(u[:, :rank].T@y)).norm()*1000)
    linear = model.layout.pose.output[1]
    raw = F.linear(data.features.to(linear.weight), linear.weight, linear.bias)
    xy = raw[:, :2]*raw.new_tensor([.3, .2])+raw.new_tensor([.3, .2])
    xy_loss = (F.smooth_l1_loss(xy/.010, data.xy.to(xy)/.010, reduction='none').sum(-1)*data.weights.to(xy)).sum()
    ce = (F.cross_entropy(raw[:, 2:20], data.bins.to(raw.device), reduction='none')*data.weights.to(raw)).sum()
    correct = int((raw[:, 2:20].argmax(-1) == data.bins.to(raw.device)).sum())
    if not all(math.isfinite(v) for v in (rms, float(xy_loss), float(ce))):
        raise FloatingPointError('nonfinite Coarse diagnostic')
    return dict(xy_svd_rms_mm=rms, rank=rank, objects=len(x), losses=dict(position_huber=float(xy_loss),
        angle_ce=float(ce)), angle_bin_correct=correct, angle_bin_accuracy=correct/len(x))


def record_progress(history, update, diagnostic):
    if history and update != history[-1]['update']+200:
        raise ValueError('Coarse progress skipped or repeated an interval')
    previous = history[-1] if history else None
    improvement = None if previous is None else (previous['xy_svd_rms_mm']-diagnostic['xy_svd_rms_mm'])/max(previous['xy_svd_rms_mm'], 1e-12)
    weak = bool(previous and improvement < .05)
    row = dict(diagnostic, update=update, relative_improvement=improvement,
               weak_intervals=(previous['weak_intervals']+1 if weak else 0))
    history.append(row)
    return row
