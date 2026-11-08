"""Explicit, receipt-bound Stage 1 budget extension; ordinary resume stays strict."""
from copy import deepcopy
import math
from pathlib import Path

from .storage import load_artifact

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from .evaluation import METRICS_SCHEMA

EXTENSION_SCHEMA = 'recovery_extension_v1'
METRICS = ('target_position_p95_mm', 'move_accuracy', 'pure_axis_success')


def progress_values(metrics, sample_ids=None):
    if (metrics.get('schema') != METRICS_SCHEMA or metrics.get('stage') != 'oracle_layout'
            or metrics.get('complete_set') is not True or metrics.get('source_mode') != 'oracle'
            or metrics.get('evaluation_mode') != 'free_rollout'):
        raise ValueError('extension requires complete oracle free-rollout metrics')
    ids = metrics.get('sample_ids')
    if not ids or len(ids) != len(set(ids)) or (sample_ids is not None and ids != sample_ids):
        raise ValueError('extension evaluation sample set changed')
    values = {key: metrics.get(key) for key in METRICS}
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('extension progress metrics are missing or nonfinite')
    if values['move_accuracy'] > 1 or values['pure_axis_success'] > 1:
        raise ValueError('extension accuracy outside [0,1]')
    return values


def load_approved_extension(path, config, fingerprint, commit, receipt, stage):
    from .driver import authorize, CHECKPOINT_SCHEMA
    authorize(config, receipt, stage, fingerprint, commit)
    grant = receipt.get('extension', {})
    if (stage != 'oracle_layout' or grant.get('schema') != EXTENSION_SCHEMA
            or grant.get('stage') != stage or grant.get('source_checkpoint_sha256') != file_hash(path)):
        raise PermissionError('extension stage or source checkpoint approval mismatch')
    saved = load_artifact(path)
    source_config = saved.get('config', {})
    if (saved.get('schema') != CHECKPOINT_SCHEMA or saved.get('status') != 'budget_exhausted'
            or saved.get('stage') != stage or saved.get('extension') is not None
            or saved.get('dataset_fingerprint') != fingerprint
            or saved.get('commit') != grant.get('source_commit')
            or saved.get('config_sha256') != grant.get('source_config_sha256')
            or saved.get('config_sha256') != digest(source_config)):
        raise ValueError('extension source code/config/data/status contract mismatch')
    origin = saved['cursor']
    old_budget, budget = source_config['stages'][stage], config['stages'][stage]
    extra, window = grant.get('additional_updates'), grant.get('progress_after_updates')
    if (type(extra) is not int or extra <= 0 or type(window) is not int or not 0 < window <= extra
            or origin['update'] != old_budget['max_updates'] or origin['update'] != grant.get('source_update')
            or origin['elapsed_seconds'] != grant.get('source_elapsed_seconds')
            or budget['max_updates'] != origin['update']+extra
            or budget['max_updates'] != grant.get('total_max_updates')
            or budget['max_seconds'] != old_budget['max_seconds']
            or budget['max_seconds'] != grant.get('total_max_seconds')
            or not math.isfinite(origin['elapsed_seconds']) or not 0 <= origin['elapsed_seconds'] < budget['max_seconds']):
        raise ValueError('extension exceeds approved cumulative budget')
    permitted = deepcopy(source_config)
    permitted['stages'][stage]['max_updates'] = budget['max_updates']
    if config != permitted:
        raise ValueError('extension may change only this stage update cap')
    if file_hash(Path(path).parent/'anchor.pt') != saved['anchor_sha256']:
        raise ValueError('extension rollback anchor hash mismatch')
    reference = progress_values(saved['metrics'])
    state = dict(schema=EXTENSION_SCHEMA, source_checkpoint=str(Path(path).resolve()),
                 source_checkpoint_sha256=grant['source_checkpoint_sha256'], source_commit=saved['commit'],
                 source_config_sha256=saved['config_sha256'], origin_update=origin['update'],
                 origin_elapsed_seconds=origin['elapsed_seconds'],
                 progress_update=origin['update']+window, reference_metrics=reference,
                 sample_ids=saved['metrics']['sample_ids'], checked=False,
                 total_max_updates=budget['max_updates'], total_max_seconds=budget['max_seconds'])
    return saved, state


def assess_progress(state, update, metrics):
    """One endpoint decision after the approved window, persisted across pauses."""
    if state is None or state['checked'] or update < state['progress_update']:
        return False
    current = progress_values(metrics, state['sample_ids'])
    reference = state['reference_metrics']
    improved = dict(target_position_p95_mm=current['target_position_p95_mm'] < reference['target_position_p95_mm']-1e-6,
                    move_accuracy=current['move_accuracy'] > reference['move_accuracy']+1e-9,
                    pure_axis_success=current['pure_axis_success'] > reference['pure_axis_success']+1e-9)
    state.update(checked=True, checked_at=update, observed_metrics=current, improved=improved)
    return not any(improved.values())
