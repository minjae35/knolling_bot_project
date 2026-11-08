"""Bounded one-stage execution, resumable checkpoints and retention rollback.

CLI training is disabled by the shipped config and additionally requires a
review receipt bound to the exact commit, config and dataset. Importing this
module, loading data, and GPU forward/backward tests never launch training.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch

from knolling_dataset.schema import digest
from knolling_dataset.builder import file_hash
from knolling_model.data import Collator, S0Dataset
from .batches import stage_batches
from .counterfactuals import RecoverySuite
from .evaluation import evaluate_recovery
from .model import RecoveryModel
from .stages import STAGES, ALL_STAGES, stage_order, configure_stage, parameter_groups, stage_loss, stage_gate
from .runtime import precision_context
from .pose_precision import PosePrecision, validate_policy
from .storage import CheckpointStore, common_store, load_artifact

CHECKPOINT_SCHEMA = 'recovery_checkpoint_v2'
CONFIG_SCHEMA = 'recovery_driver_v2'


def read_config(path):
    config = json.loads(Path(path).read_text())
    architecture = config.get('model', {}).get('architecture', 'recovery_v2_ar')
    order = stage_order(architecture)
    if config.get('schema') != CONFIG_SCHEMA or tuple(config.get('stages', {})) != order:
        raise ValueError('invalid recovery execution schema or stage order')
    if config.get('device') not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if type(config.get('seed')) is not int:
        raise ValueError('an integer seed is required')
    architecture = config.get('model', {}).get('architecture', 'recovery_v2_ar')
    if architecture not in ('recovery_v2_ar', 'recovery_v3_joint', 'recovery_v4_goal_joint'):
        raise ValueError('unknown model architecture')
    if architecture == 'recovery_v3_joint' and (not config.get('data_v3') or
            config.get('evaluation_schema') != 'recovery_metrics_v3'):
        raise ValueError('v3 requires pinned v3 data and evaluation')
    if architecture == 'recovery_v4_goal_joint':
        from .goal_labels import CONTRACT
        if (not config.get('data_v3') or config.get('evaluation_schema') != 'recovery_metrics_v4'
                or digest(config.get('goal_contract')) != digest(CONTRACT) or not config.get('micro_pose')):
            raise ValueError('v4 requires pinned data, goal contract, micro budgets and v4 evaluation')
        if (not isinstance(config.get('goal_lr'), (int, float)) or not math.isfinite(config['goal_lr'])
                or config['goal_lr'] <= 0):
            raise ValueError('invalid goal learning rate')
    for key in ('batch_size', 'evaluate_every', 'checkpoint_every', 'cpu_threads'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'invalid {key}')
    for key in ('layout_lr', 'perception_lr', 'weight_decay', 'gradient_clip'):
        if not isinstance(config.get(key), (int,float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'invalid {key}')
    specs = list(config['stages'].values()) + ([config['micro_pose']] if architecture == 'recovery_v4_goal_joint' else [])
    for spec in specs:
        if type(spec.get('max_updates')) is not int or spec['max_updates'] <= 0:
            raise ValueError('invalid update budget')
        if not isinstance(spec.get('max_seconds'), (int,float)) or not math.isfinite(spec['max_seconds']) or spec['max_seconds'] <= 0:
            raise ValueError('invalid time budget')
    for stage, precision in config.get('precision', {}).items():
        if stage not in (*order, 'micro_pose') or precision not in ('fp32', 'bf16') or (precision == 'bf16' and config['device'] != 'cuda'):
            raise ValueError('invalid stage precision')
    for stage, size in config.get('batch_sizes', {}).items():
        if stage not in (*order, 'micro_pose') or type(size) is not int or size < 1:
            raise ValueError('invalid stage batch size')
    if type(config.get('num_workers', 0)) is not int or config.get('num_workers', 0) < 0:
        raise ValueError('invalid DataLoader workers')
    validate_policy(config)
    from .integrated import validate_config
    validate_config(config)
    return config


def authorize(config, receipt, stage, fingerprint, commit):
    """Must execute before creating an output directory/model/CUDA context."""
    if config.get('execution') != 'approved_training' or config.get('allow_training') is not True:
        raise PermissionError('training remains disabled pending independent review')
    if not (receipt and receipt.get('approved') is True and receipt.get('reviewer')
            and receipt.get('commit') == commit and receipt.get('config_sha256') == digest(config)
            and receipt.get('dataset_fingerprint') == fingerprint and stage in receipt.get('stages', [])):
        raise PermissionError('review receipt does not authorize this code/config/data/stage')
    if config['device'] == 'cuda' and receipt.get('gpu_training') is not True:
        raise PermissionError('GPU training is not authorized by the review receipt')


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_exclusive(path, payload):
    """Atomic new checkpoint; never overwrite a previous artifact."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # Atomic publication, EEXIST if path already exists.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_exclusive(path, payload):
    """Publish a complete JSON report without overwriting an existing result."""
    path = Path(path)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.report-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, config, fingerprint, commit):
    return validate_checkpoint(load_artifact(path), config, fingerprint, commit)


def validate_checkpoint(saved, config, fingerprint, commit):
    if (saved.get('schema') != CHECKPOINT_SCHEMA or saved.get('config_sha256') != digest(config)
            or saved.get('dataset_fingerprint') != fingerprint or saved.get('commit') != commit
            or saved.get('architecture','recovery_v2_ar') != config.get('model',{}).get('architecture','recovery_v2_ar')
            or saved.get('evaluation_schema','recovery_metrics_v2') != config.get('evaluation_schema','recovery_metrics_v2')):
        raise ValueError('checkpoint code/config/data contract mismatch')
    if saved.get('architecture') == 'recovery_v4_goal_joint':
        from .goal_labels import CONTRACT
        if (digest(saved.get('goal_contract')) != digest(CONTRACT) or
                saved.get('purpose') != ('diagnostic' if saved.get('stage') == 'micro_pose' else 'production')):
            raise ValueError('checkpoint goal schema or purpose mismatch')
    return saved


def validate_micro_checkpoints(paths, parent_sha, config, fingerprint, commit, receipt=None):
    """Three independent diagnostic passes, each rooted at the SAME grounding file."""
    from .evaluation_v4 import MICRO_CASES
    if not paths or len(paths) != len(MICRO_CASES):
        raise ValueError('oracle layout requires all three passed micro diagnostics')
    evidence = {}
    for path in paths:
        grants = [g for g in (receipt or {}).get('micro_evidence_transfer', []) if g.get('sha256') == file_hash(path)]
        if grants:
            from .provenance import load_evidence_transfer
            if len(grants) != 1: raise ValueError('ambiguous Micro evidence grant')
            saved = load_evidence_transfer(path, config, fingerprint, grants[0])
        else:
            saved = load_checkpoint(path, config, fingerprint, commit)
        if config.get('integrated_pose') and saved.get('config', {}).get('integrated_pose') != config['integrated_pose']:
            raise ValueError('integrated recipe requires its own three Micro regression passes')
        case = saved.get('micro_case')
        if (case not in MICRO_CASES or case in evidence or saved['stage'] != 'micro_pose'
                or saved['status'] != 'passed' or saved.get('goal_parent_sha256') != parent_sha
                or not stage_gate('micro_pose', saved.get('metrics', {}))
                or saved['metrics']['micro'].get('case') != case
                or saved['metrics']['micro'].get('sample_ids') != list(MICRO_CASES[case])):
            raise ValueError('micro diagnostic failed or has a different grounding parent/case')
        evidence[case] = dict(path=str(path), sha256=file_hash(path), metrics=saved['metrics'])
    return evidence


def load_grounding_transfer(path, config, fingerprint, receipt):
    """Explicit pinned, passed Grounding weights only; never a Micro resume.

    The current receipt authorizes this source hash/commit/config. Only the new
    precision policy may differ in the legacy route. Integrated transfer also
    verifies semantic hashes and an explicit list of allowed config changes.
    The original checkpoint remains immutable and retains its original contract.
    """
    grant = (receipt or {}).get('grounding_transfer', {})
    if config.get('integrated_pose'):
        from .provenance import load_evidence_transfer
        saved = load_evidence_transfer(path, config, fingerprint, grant)
        if (saved['stage'] != 'grounding' or not stage_gate('grounding', saved.get('metrics', {}))
                or not stage_gate('grounding', saved.get('completed', {}).get('grounding', {}))):
            raise ValueError('grounding transfer requires a passed production parent')
        return saved
    if not config.get('micro_precision') or file_hash(path) != grant.get('sha256'):
        raise ValueError('unapproved grounding transfer')
    raw = load_artifact(path)
    source_config = raw.get('config', {})
    if (raw.get('commit') != grant.get('commit')
            or digest(source_config) != grant.get('config_sha256')):
        raise ValueError('grounding transfer source contract mismatch')
    target_config = deepcopy(config)
    target_config.pop('micro_precision', None)
    if target_config != source_config:
        raise ValueError('grounding transfer may change only micro precision policy')
    saved = load_checkpoint(path, source_config, fingerprint, grant['commit'])
    if (saved['stage'] != 'grounding' or saved['status'] != 'passed'
            or not stage_gate('grounding', saved.get('metrics', {}))
            or not stage_gate('grounding', saved.get('completed', {}).get('grounding', {}))):
        raise ValueError('grounding transfer requires a passed production parent')
    return saved


@dataclass
class Cursor:
    update: int = 0
    epoch: int = 0
    batch: int = 0
    elapsed_seconds: float = 0.


class StageEngine:
    """Execution core. Production callers use main() and its review receipt guard.

    Dependencies are injectable only to unit-test lifecycle on scalar fixtures;
    the CLI uses the real RecoveryModel, stage loss, batches and evaluator.
    """
    def __init__(self, model, config, fingerprint, commit, output, batches, evaluate,
                 *, loss_fn=stage_loss, configure_fn=configure_stage, gate_fn=stage_gate,
                 clock=time.monotonic, stop_event=None, review_receipt=None, observer=None,
                 prepare_continuation=None, readout_batches=None, expected_ids=None):
        self.model, self.config, self.fingerprint, self.commit = model, config, fingerprint, commit
        self.output = Path(output)
        self.batches, self.evaluate, self.loss_fn = batches, evaluate, loss_fn
        self.configure, self.gate, self.clock = configure_fn, gate_fn, clock
        self.stop = stop_event or threading.Event()
        self.review_receipt = deepcopy(review_receipt)
        self.observer = observer
        self.prepare_continuation = prepare_continuation
        self.readout_batches, self.expected_ids = readout_batches, expected_ids
        self.serial = 0
        self.storage = CheckpointStore(common_store(self.output))

    def run(self, stage, *, parent=None, resume=None, extend_from=None, micro_case=None, micro_checkpoints=None):
        if self.config.get('integrated_pose') and stage in ('micro_pose', 'oracle_layout'):
            from .integrated import IntegratedEngine
            return IntegratedEngine(self).run(stage, parent=parent, resume=resume, extend_from=extend_from,
                                             micro_case=micro_case, micro_checkpoints=micro_checkpoints)
        from .extensions import load_approved_extension, assess_progress
        architecture = self.config.get('model', {}).get('architecture', 'recovery_v2_ar')
        v4 = architecture == 'recovery_v4_goal_joint'
        order = stage_order(architecture)
        if stage not in ((*order, 'micro_pose') if v4 else order) or sum(bool(p) for p in (parent, resume, extend_from)) > 1:
            raise ValueError('invalid stage/checkpoint request')
        architecture = self.config.get('model', {}).get('architecture', 'recovery_v2_ar')
        if getattr(self.model, 'architecture', 'recovery_v2_ar') != architecture:
            raise ValueError('model/config architecture mismatch')
        if architecture in ('recovery_v3_joint', 'recovery_v4_goal_joint') and extend_from:
            raise ValueError('joint architectures do not reuse the legacy budget extension contract')
        if stage == 'micro_pose':
            from .evaluation_v4 import MICRO_CASES
            if micro_case not in MICRO_CASES:
                raise ValueError('micro pose requires an explicit fixed case')
        elif micro_case is not None:
            raise ValueError('micro case is diagnostic-only')
        extension = None
        if stage == 'micro_pose' and self.review_receipt is not None:
            cases = self.review_receipt.get('cases')
            if cases is not None and micro_case not in cases:
                raise PermissionError('micro case is not authorized')
        if extend_from:
            prior, extension = load_approved_extension(extend_from, self.config, self.fingerprint,
                                                       self.commit, self.review_receipt, stage)
        else:
            if parent and stage == 'micro_pose' and (self.review_receipt or {}).get('grounding_transfer'):
                prior = load_grounding_transfer(parent, self.config, self.fingerprint, self.review_receipt)
            else:
                prior = load_checkpoint(resume or parent, self.config, self.fingerprint, self.commit) if (resume or parent) else None
        continuing = bool(resume or extend_from)
        completed = {} if prior is None else prior['completed']
        position = 1 if stage == 'micro_pose' else order.index(stage)
        if extend_from:
            pass  # Validated explicitly above; never relabel a budget checkpoint as paused.
        elif resume:
            if prior['stage'] != stage or prior['status'] not in ('paused', 'running'):
                raise ValueError('only an interrupted checkpoint of the same stage can resume')
            extension = deepcopy(prior.get('extension'))
            if prior.get('micro_case') != micro_case:
                raise ValueError('resume micro case mismatch')
        elif position:
            previous = 'grounding' if stage == 'micro_pose' else order[position-1]
            if prior is None or prior['stage'] != previous or prior['status'] != 'passed' or not self.gate(previous, completed.get(previous, {})):
                raise ValueError('previous stage must pass on the full pinned evaluation suite')
        elif prior is not None:
            raise ValueError('first stage must start fresh or resume itself')
        goal_parent = prior.get('goal_parent_sha256') if prior else None
        if v4 and parent and prior['stage'] == 'grounding':
            goal_parent = file_hash(parent)
        micro_evidence = deepcopy(prior.get('micro_evidence', {})) if prior else {}
        if v4 and stage == 'oracle_layout' and not continuing:
            micro_evidence = validate_micro_checkpoints(micro_checkpoints, goal_parent, self.config, self.fingerprint, self.commit)
        if prior:
            self.model.load_state_dict(prior['model'], strict=True)
        self.configure(self.model, stage)
        optimizer = torch.optim.AdamW(parameter_groups(self.model, self.config['layout_lr'], self.config['perception_lr'], self.config.get('goal_lr', 3e-4)),
                                      weight_decay=self.config['weight_decay'], foreach=False)
        cursor = Cursor(**prior['cursor']) if continuing else Cursor()
        if continuing:
            optimizer.load_state_dict(prior['optimizer'])
            anchor_path = Path(resume or extend_from).parent/'anchor.pt'
            if file_hash(anchor_path) != prior['anchor_sha256']:
                raise ValueError('rollback anchor hash mismatch')
            anchor = load_artifact(anchor_path)
            baseline = prior['baseline']
            if self.prepare_continuation is not None:
                self.prepare_continuation(stage)
            restore_rng(prior['rng'])
        else:
            anchor = cpu_copy(self.model.state_dict())
            baseline = None
        precision = None
        if stage == 'micro_pose' and self.config.get('micro_precision'):
            precision = PosePrecision(self.model, optimizer, continuing=continuing,
                                      saved=prior.get('pose_precision') if continuing else None)
        # Every execution/resume gets a new directory; no existing run is mutated.
        self.output.mkdir(parents=True, exist_ok=False)
        self.storage.save_anchor(self.output/'anchor.pt', anchor, stage)
        anchor_sha = file_hash(self.output/'anchor.pt')
        start = self.clock()
        previous_elapsed = cursor.elapsed_seconds
        spec = self.config['micro_pose'] if stage == 'micro_pose' else self.config['stages'][stage]
        def elapsed():
            return previous_elapsed+self.clock()-start
        def exhausted():
            return cursor.update >= spec['max_updates'] or elapsed() >= spec['max_seconds']
        def observe(event, **values):
            if self.observer is not None:
                self.observer(event, stage=stage, update=cursor.update, elapsed_seconds=elapsed(), **values)
        def checkpoint(status, metrics=None, error=None):
            cursor.elapsed_seconds = elapsed()
            self.serial += 1
            path = self.output/f'checkpoint-{cursor.update:06d}-{self.serial:04d}.pt'
            self.storage.save_checkpoint(path, dict(schema=CHECKPOINT_SCHEMA, stage=stage, status=status,
                architecture=architecture, evaluation_schema=self.config.get('evaluation_schema','recovery_metrics_v2'),
                config_sha256=digest(self.config), dataset_fingerprint=self.fingerprint, commit=self.commit,
                config=deepcopy(self.config), review_receipt=self.review_receipt,
                model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                cursor=vars(cursor).copy(), rng=rng_state(), baseline=baseline, completed=completed,
                goal_contract=deepcopy(self.config.get('goal_contract')), goal_parent_sha256=goal_parent,
                purpose='diagnostic' if stage == 'micro_pose' else 'production',
                micro_case=micro_case, micro_evidence=micro_evidence,
                metrics=metrics, error=error, anchor_sha256=anchor_sha, extension=extension,
                pose_precision=deepcopy(precision.state) if precision else None))
            observe('checkpoint', status=status, checkpoint=str(path), error=error)
            return dict(status=status, checkpoint=str(path), updates=cursor.update,
                        elapsed_seconds=cursor.elapsed_seconds, metrics=metrics, error=error)
        def measured():
            with precision_context(self.config, stage):
                metrics = self.evaluate(stage, baseline)
            observe('evaluation', metrics=metrics)
            if precision and precision.consider(metrics, cursor.update):
                observe('precision_transition', **precision.state,
                        learning_rates=[g['lr'] for g in optimizer.param_groups])
            return metrics
        def finish(metrics):
            if self.gate(stage, metrics):
                completed[stage] = metrics
                return checkpoint('passed', metrics)
            return None
        def rollback(metrics, reason):
            self.model.load_state_dict(anchor, strict=True)
            self.model.zero_grad(set_to_none=True)
            # Failed checkpoints are deliberately non-resumable: moments belong
            # to the rejected trajectory. Restart from the prior passed stage.
            return checkpoint('rolled_back', metrics, reason)
        def check_progress(metrics):
            if assess_progress(extension, cursor.update, metrics):
                return checkpoint('early_stopped', metrics, 'no improvement over the approved extension window')
            return None
        try:
            observe('start', max_updates=spec['max_updates'], max_seconds=spec['max_seconds'])
            if extend_from:
                observe('extension', authorization=extension)
                checkpoint('running', prior['metrics'])
            if not continuing:
                baseline = measured()
                if stage == 'joint' and not self.gate('predicted_layout', completed['predicted_layout']):
                    raise ValueError('joint requires a passed predicted-layout reference')
                metrics = measured()
                if self.config.get('stop_on_initial_pass', True):
                    done = finish(metrics)
                    if done:
                        return done
                checkpoint('running', metrics)
            while not exhausted():
                if self.stop.is_set():
                    return checkpoint('paused')
                seen = False
                for index, item in enumerate(self.batches(stage, cursor.epoch)):
                    if index < cursor.batch:
                        continue
                    seen = True
                    if self.stop.is_set():
                        return checkpoint('paused')
                    if exhausted():
                        break
                    optimizer.zero_grad(set_to_none=True)
                    attempt_rng = rng_state()
                    inputs, truths, replay = item.loss_arguments()
                    with precision_context(self.config, stage):
                        terms, _ = self.loss_fn(self.model, inputs, truths, replay=replay)
                        loss = sum(terms.values())
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], self.config['gradient_clip'], error_if_nonfinite=True)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('nonfinite gradient')
                    # Do not commit an update after a stop signal or expired wall budget.
                    if self.stop.is_set() or elapsed() >= spec['max_seconds']:
                        optimizer.zero_grad(set_to_none=True)
                        # This batch has not advanced the cursor or optimizer.
                        # Replaying it after resume must use the same DN/dropout RNG.
                        restore_rng(attempt_rng)
                        if self.stop.is_set():
                            return checkpoint('paused')
                        # A timeout must reach the common final evaluation below,
                        # including joint regression detection and rollback.
                        break
                    optimizer.step()
                    if not torch.stack([torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad]).all():
                        raise FloatingPointError('nonfinite parameter after update')
                    cursor.update += 1
                    cursor.batch = index+1
                    if self.observer is not None:
                        values = torch.stack([loss.detach(), norm.detach(), *[v.detach() for v in terms.values()]]).float().cpu().tolist()
                        observe('update', loss=values[0], gradient_norm=values[1], losses=dict(zip(terms, values[2:])),
                                learning_rates=[g['lr'] for g in optimizer.param_groups],
                                pose_phase=precision.state['phase'] if precision else None,
                                remaining_seconds=max(0., spec['max_seconds']-elapsed()))
                    progress_due = extension is not None and not extension['checked'] and cursor.update >= extension['progress_update']
                    if precision or cursor.update % self.config['evaluate_every'] == 0 or exhausted() or progress_due:
                        metrics = measured()
                        if stage == 'joint' and not metrics.get('full_set_no_regression', False):
                            return rollback(metrics, 'full-set regression')
                        done = check_progress(metrics) or finish(metrics)
                        if done:
                            return done
                    if cursor.update % self.config['checkpoint_every'] == 0:
                        checkpoint('running')
                    if exhausted():
                        break
                if exhausted():
                    break
                # If resume skipped the entire completed epoch, advance once.
                if not seen and cursor.batch == 0:
                    raise ValueError('empty training stream')
                cursor.epoch += 1
                cursor.batch = 0
            metrics = measured()
            if stage == 'joint' and not metrics.get('full_set_no_regression', False):
                return rollback(metrics, 'full-set regression at budget')
            return check_progress(metrics) or finish(metrics) or checkpoint('budget_exhausted', metrics)
        except Exception as exc:
            return rollback(None, f'{type(exc).__name__}: {exc}')


@contextmanager
def stop_signals(event):
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: event.set())
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/recovery_execution.json')
    parser.add_argument('--stage', choices=ALL_STAGES, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--parent-checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--micro-case', choices=('translation', 'keep', 'axis'))
    parser.add_argument('--micro-checkpoints', nargs=3, help='three passed diagnostic checkpoint files for oracle layout')
    parser.add_argument('--extend-from', help='explicitly approved Stage 1 budget checkpoint; never ordinary resume')
    args = parser.parse_args()
    config = read_config(args.config)
    # Fail closed before data/model loading, CUDA setup or artifact creation.
    if config.get('allow_training') is not True:
        raise PermissionError('training disabled; this revision is awaiting independent review')
    receipt = json.loads(Path(args.approval).read_text())
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'], text=True).strip():
        raise PermissionError('training requires a clean reviewed checkout')
    parent = S0Dataset(config['dataset'], config['selection'])
    v4 = config.get('model',{}).get('architecture') == 'recovery_v4_goal_joint'
    v3 = config.get('model',{}).get('architecture') == 'recovery_v3_joint'
    if v3 or v4:
        from .dataset_v3 import RecoveryDatasetV3
        suite = RecoveryDatasetV3(parent, config['data_v3']['path'], config['data_v3']['sha256'])
        if v4:
            from .goal_labels import GoalDataset
            suite = GoalDataset(suite)
    else:
        suite = RecoverySuite(parent, config['counterfactuals']['path'], config['counterfactuals']['sha256'])
    pairs = getattr(suite, 'pairs', ())
    authorize(config, receipt, args.stage, suite.fingerprint, commit)
    fingerprint = suite.fingerprint
    if args.stage == 'micro_pose':
        from .micro import MicroDataset
        suite = MicroDataset(suite, args.micro_case)
    if Path(args.output).resolve().is_relative_to(parent.root):
        raise ValueError('run output cannot be inside the parent dataset')
    from knolling_model.runner import seed_everything
    from knolling_model.model import load_pretrained
    from .runtime import CachedDataset, FrozenTextCache, ProgressLog, warm_text_cache
    seed_everything(config['seed'], config['cpu_threads'])
    base, tokenizer = load_pretrained(config['pretrained'], local_files_only=True)
    model = RecoveryModel(base.text_encoder, base.perception.detector, **config['model']).to(config['device'])
    if config.get('cache_frozen_text', False):
        model.encode_text = FrozenTextCache(model.encode_text)
    if config.get('cache_dataset', False):
        suite = CachedDataset(suite)
    collator = Collator(tokenizer)
    def batches(stage, epoch):
        size = config.get('batch_sizes', {}).get(stage, config['batch_size'])
        for item in stage_batches(suite, collator, stage, seed=config['seed'], epoch=epoch, batch_size=size,
                                  num_workers=config.get('num_workers', 0), pin_memory=config.get('pin_memory', False), uniform=v4):
            yield item.to(config['device'], non_blocking=config.get('pin_memory', False))
    def stream():
        for i in range(0,len(suite),config['batch_size']):
            yield collator([suite[j] for j in range(i,min(len(suite),i+config['batch_size']))]).to(config['device'])
    def evaluation(stage, baseline):
        if v4:
            from .evaluation_v4 import evaluate_v4
            return evaluate_v4(model, stream(), stage=stage, pairs=pairs,
                expected_ids=[r['sample_id'] for r in suite.records], baseline=baseline, micro_case=args.micro_case)[0]
        if v3:
            from .evaluation_v3 import evaluate_v3
            return evaluate_v3(model, stream(), stage=stage, pairs=pairs,
                expected_ids=[r['sample_id'] for r in suite.records], baseline=baseline)[0]
        return evaluate_recovery(model, stream(), stage=stage,
            expected_ids=[r['sample_id'] for r in suite.records], baseline=baseline)[0]
    event = threading.Event()
    observer = ProgressLog(args.output) if config.get('progress', False) else None
    def prepare_continuation(stage):
        if stage != 'perception':
            warm_text_cache(model, suite, collator, config['batch_size'], config['device'])
    engine = StageEngine(model, config, fingerprint, commit, args.output, batches, evaluation,
                         stop_event=event, review_receipt=receipt, observer=observer,
                         prepare_continuation=prepare_continuation, readout_batches=stream,
                         expected_ids=[r['sample_id'] for r in suite.records])
    try:
        with stop_signals(event):
            result = engine.run(args.stage, parent=args.parent_checkpoint, resume=args.resume, extend_from=args.extend_from,
                                micro_case=args.micro_case, micro_checkpoints=args.micro_checkpoints)
    finally:
        if observer is not None:
            observer.close()
    save_json_exclusive(Path(args.output)/'result.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result['status'] not in ('passed','paused'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
