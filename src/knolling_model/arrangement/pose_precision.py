"""Opt-in, one-way Micro Pose precision phase; no model or loss changes."""
from copy import deepcopy
import math


POLICY = dict(position_mm=10., angle_deg=2., coarse_lr=1e-4, fine_lr=1e-6)


def validate_policy(config):
    policy = config.get('micro_precision')
    if policy is not None and (
            config.get('model', {}).get('architecture') != 'recovery_v4_goal_joint'
            or policy != POLICY or config['layout_lr'] != POLICY['coarse_lr']):
        raise ValueError('invalid v4 micro precision policy')


class PosePrecision:
    """Reset only exp_avg, exactly once, after a complete measured evaluation.

    Optimizer state loading precedes construction on resume. Missing phase or
    inconsistent learning rates fail closed instead of silently resetting again.
    """
    def __init__(self, model, optimizer, *, saved=None, continuing=False):
        self.optimizer = optimizer
        pose = {id(p) for p in model.layout.pose.parameters() if p.requires_grad}
        actual = {id(p) for g in optimizer.param_groups for p in g['params']}
        if not pose or actual != pose or any(g.get('name') != 'layout' for g in optimizer.param_groups):
            raise ValueError('micro precision requires exactly all trainable Pose parameters')
        if continuing and saved is None:
            raise ValueError('missing precision state on resume')
        self.state = deepcopy(saved) if saved is not None else dict(phase='coarse', transition_update=None)
        phase = self.state.get('phase')
        update = self.state.get('transition_update')
        if (phase not in ('coarse', 'fine') or (phase == 'coarse' and update is not None)
                or (phase == 'fine' and (type(update) is not int or update < 0))):
            raise ValueError('invalid precision phase state')
        expected = POLICY['fine_lr' if phase == 'fine' else 'coarse_lr']
        if any(g['lr'] != expected for g in optimizer.param_groups):
            raise ValueError('precision phase/optimizer learning rate mismatch')

    def consider(self, metrics, update):
        if self.state['phase'] == 'fine':
            return False
        m = metrics.get('micro', {})
        values = (m.get('max_position_mm'), m.get('max_angle_deg'))
        if (metrics.get('schema') != 'recovery_metrics_v4'
                or metrics.get('stage') != 'micro_pose'
                or metrics.get('complete_set') is not True
                or metrics.get('source_mode') != 'oracle'
                or metrics.get('evaluation_mode') != 'free_rollout'
                or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values)
                or values[0] > POLICY['position_mm'] or values[1] > POLICY['angle_deg']):
            return False
        for group in self.optimizer.param_groups:
            group['lr'] = POLICY['fine_lr']
            for p in group['params']:
                state = self.optimizer.state.get(p, {})
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
        self.state = dict(phase='fine', transition_update=update)
        return True
