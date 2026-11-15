"""Explicit v4 XY/orientation path composition; no labels or planning in forward."""
from copy import deepcopy
import torch
from torch import nn
from knolling_dataset.schema import digest
from .goal_joint_layout import AbsoluteJointPose
from .storage import tensor_digest

SCHEMA = 'v4_split_pose_v1'
PREFIX = 'layout.pose.orientation_pose.'


class SplitJointPose(AbsoluteJointPose):
    """Keep legacy XY parameter names and Adam mapping; add one frozen branch."""
    def __init__(self, xy_pose):
        nn.Module.__init__(self)
        direction = deepcopy(xy_pose).requires_grad_(False).eval()
        for name, module in xy_pose.named_children(): self.add_module(name, module)
        self.orientation_pose = direction

    def train(self, mode=True):
        super().train(mode)
        self.orientation_pose.eval()
        return self

    def requires_grad_(self, value=True):
        super().requires_grad_(value)
        self.orientation_pose.requires_grad_(False)
        return self

    def forward(self, objects, goal, selected):
        xy = super().forward(objects, goal, selected)
        direction = self.orientation_pose(objects, goal, selected)
        return dict(xy, **{k: direction[k] for k in ('angle', 'angle_logits', 'angle_residual')})


def validate_config(config):
    spec = config.get('split_pose')
    if spec is None: return
    if (set(spec) != {'schema', 'source_sha256', 'xy_sha256', 'orientation_state_sha256'}
            or spec['schema'] != SCHEMA
            or config.get('alternating_pose', {}).get('diagnostic') != 'xy_feature_window_v1'
            or config['model']['architecture'] != 'recovery_v4_goal_joint'):
        raise ValueError('invalid split-pose diagnostic contract')
    for k in ('source_sha256', 'xy_sha256', 'orientation_state_sha256'):
        if len(spec[k]) != 64 or any(ch not in '0123456789abcdef' for ch in spec[k]):
            raise ValueError('invalid split-pose provenance hash')


def load_model_state(model, saved):
    spec = saved.get('config', {}).get('split_pose')
    if spec is not None:
        validate_config(saved['config'])
        direction = {k[len(PREFIX):]: v for k,v in saved['model'].items() if k.startswith(PREFIX)}
        if not direction or tensor_digest(direction) != spec['orientation_state_sha256']:
            raise ValueError('frozen orientation branch integrity mismatch')
        if not isinstance(model.layout.pose, SplitJointPose):
            model.layout.pose = SplitJointPose(model.layout.pose)
    model.load_state_dict(saved['model'], strict=True)


def compose(source, candidate, *, source_sha256, xy_sha256, commit, anchor_path, anchor_sha256):
    """In-memory diagnostic conversion; callers validate both donors first.

    The two-update optimizer/RNG/cursor survive. Old caches and decisions cannot
    authorize this changed forward. Resume must evaluate before another update.
    """
    state = candidate['integrated']
    if (source['cursor']['update'] != 600 or candidate['cursor']['update'] != 2
            or candidate['stage'] != 'oracle_layout' or state['phase'] != 'feature'
            or state.get('pending_feature') or state.get('pending_head') is not None
            or state['candidate_history'][-1]['feature_update'] != 2
            or state['source']['sha256'] != source_sha256
            or candidate['dataset_fingerprint'] != source['dataset_fingerprint']):
        raise ValueError('expected completed two-update candidate from pinned source')
    for k,v in source['model'].items():
        if not k.startswith('layout.pose.') and not torch.equal(v, candidate['model'][k]):
            raise ValueError('shared Goal/input weights differ between donors')
    direction = {k[len('layout.pose.'):]:v.clone() for k,v in source['model'].items() if k.startswith('layout.pose.')}
    out = deepcopy(candidate)
    out['model'].update({PREFIX+k:v for k,v in direction.items()})
    config = out['config']
    config.update(device='cpu', allow_training=False, execution='review_only',
                  budget_status='CPU_split_path_validation_only')
    config['split_pose'] = dict(schema=SCHEMA, source_sha256=source_sha256, xy_sha256=xy_sha256,
                               orientation_state_sha256=tensor_digest(direction))
    out.update(config_sha256=digest(config), commit=commit, status='paused', metrics=None,
               error=None, review_receipt=None, anchor_sha256=anchor_sha256, purpose='diagnostic')
    # This explicit GPU-donor -> CPU diagnostic conversion preserves donor CUDA
    # RNG in provenance; CPU Resume restores the active CPU streams exactly.
    out['integrated']['split_origin_rng'] = deepcopy(out['rng'])
    out['integrated']['split_origin_anchor_rng'] = deepcopy(out['integrated']['anchor_rng'])
    out['rng']['cuda'] = []
    out['integrated']['anchor_rng']['cuda'] = []
    out['integrated'].update(cache_path=None, cache_key=None, cache_sha256=None, solver=None,
        pending_head=None, pending_feature=False, pending_split_evaluation=True,
        anchor_path=str(anchor_path), candidate_history=[], last_trial=None)
    return out


def source_anchor(source):
    anchor = deepcopy(source['model'])
    anchor.update({PREFIX+k[len('layout.pose.'):]:v.clone() for k,v in source['model'].items() if k.startswith('layout.pose.')})
    return anchor
