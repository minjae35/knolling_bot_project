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
from .stages import STAGES, configure_stage, parameter_groups, stage_loss, stage_gate

CHECKPOINT_SCHEMA = 'recovery_checkpoint_v2'
CONFIG_SCHEMA = 'recovery_driver_v2'


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != CONFIG_SCHEMA or tuple(config.get('stages', {})) != STAGES:
        raise ValueError('invalid recovery execution schema or stage order')
    if config.get('device') not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if type(config.get('seed')) is not int:
        raise ValueError('an integer seed is required')
    for key in ('batch_size', 'evaluate_every', 'checkpoint_every', 'cpu_threads'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'invalid {key}')
    for key in ('layout_lr', 'perception_lr', 'weight_decay', 'gradient_clip'):
        if not isinstance(config.get(key), (int,float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'invalid {key}')
    for spec in config['stages'].values():
        if type(spec.get('max_updates')) is not int or spec['max_updates'] <= 0:
            raise ValueError('invalid update budget')
        if not isinstance(spec.get('max_seconds'), (int,float)) or not math.isfinite(spec['max_seconds']) or spec['max_seconds'] <= 0:
            raise ValueError('invalid time budget')
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
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, config, fingerprint, commit):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('schema') != CHECKPOINT_SCHEMA or saved.get('config_sha256') != digest(config)
            or saved.get('dataset_fingerprint') != fingerprint or saved.get('commit') != commit):
        raise ValueError('checkpoint code/config/data contract mismatch')
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
                 clock=time.monotonic, stop_event=None, review_receipt=None):
        self.model, self.config, self.fingerprint, self.commit = model, config, fingerprint, commit
        self.output = Path(output)
        self.batches, self.evaluate, self.loss_fn = batches, evaluate, loss_fn
        self.configure, self.gate, self.clock = configure_fn, gate_fn, clock
        self.stop = stop_event or threading.Event()
        self.review_receipt = deepcopy(review_receipt)
        self.serial = 0

    def run(self, stage, *, parent=None, resume=None):
        if stage not in STAGES or (parent and resume):
            raise ValueError('invalid stage/checkpoint request')
        prior = load_checkpoint(resume or parent, self.config, self.fingerprint, self.commit) if (resume or parent) else None
        completed = {} if prior is None else prior['completed']
        position = STAGES.index(stage)
        if resume:
            if prior['stage'] != stage or prior['status'] not in ('paused', 'running'):
                raise ValueError('only an interrupted checkpoint of the same stage can resume')
        elif position:
            previous = STAGES[position-1]
            if prior is None or prior['stage'] != previous or prior['status'] != 'passed' or not self.gate(previous, completed.get(previous, {})):
                raise ValueError('previous stage must pass on the full pinned evaluation suite')
        elif prior is not None:
            raise ValueError('first stage must start fresh or resume itself')
        if prior:
            self.model.load_state_dict(prior['model'], strict=True)
        self.configure(self.model, stage)
        optimizer = torch.optim.AdamW(parameter_groups(self.model, self.config['layout_lr'], self.config['perception_lr']),
                                      weight_decay=self.config['weight_decay'], foreach=False)
        cursor = Cursor(**prior['cursor']) if resume else Cursor()
        if resume:
            optimizer.load_state_dict(prior['optimizer'])
            anchor_path = Path(resume).parent/'anchor.pt'
            if file_hash(anchor_path) != prior['anchor_sha256']:
                raise ValueError('rollback anchor hash mismatch')
            anchor = torch.load(anchor_path, map_location='cpu', weights_only=True)
            baseline = prior['baseline']
            restore_rng(prior['rng'])
        else:
            anchor = cpu_copy(self.model.state_dict())
            baseline = None
        # Every execution/resume gets a new directory; no existing run is mutated.
        self.output.mkdir(parents=True, exist_ok=False)
        save_exclusive(self.output/'anchor.pt', anchor)
        anchor_sha = file_hash(self.output/'anchor.pt')
        start = self.clock()
        previous_elapsed = cursor.elapsed_seconds
        spec = self.config['stages'][stage]
        def elapsed():
            return previous_elapsed+self.clock()-start
        def exhausted():
            return cursor.update >= spec['max_updates'] or elapsed() >= spec['max_seconds']
        def checkpoint(status, metrics=None, error=None):
            cursor.elapsed_seconds = elapsed()
            self.serial += 1
            path = self.output/f'checkpoint-{cursor.update:06d}-{self.serial:04d}.pt'
            save_exclusive(path, dict(schema=CHECKPOINT_SCHEMA, stage=stage, status=status,
                config_sha256=digest(self.config), dataset_fingerprint=self.fingerprint, commit=self.commit,
                config=deepcopy(self.config), review_receipt=self.review_receipt,
                model=cpu_copy(self.model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                cursor=vars(cursor).copy(), rng=rng_state(), baseline=baseline, completed=completed,
                metrics=metrics, error=error, anchor_sha256=anchor_sha))
            return dict(status=status, checkpoint=str(path), updates=cursor.update,
                        elapsed_seconds=cursor.elapsed_seconds, metrics=metrics, error=error)
        def measured():
            return self.evaluate(stage, baseline)
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
        try:
            if not resume:
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
                        return checkpoint('paused' if self.stop.is_set() else 'budget_exhausted')
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad):
                        raise FloatingPointError('nonfinite parameter after update')
                    cursor.update += 1
                    cursor.batch = index+1
                    if cursor.update % self.config['evaluate_every'] == 0 or exhausted():
                        metrics = measured()
                        if stage == 'joint' and not metrics.get('full_set_no_regression', False):
                            return rollback(metrics, 'full-set regression')
                        done = finish(metrics)
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
            return finish(metrics) or checkpoint('budget_exhausted', metrics)
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
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--parent-checkpoint')
    parser.add_argument('--resume')
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
    suite = RecoverySuite(parent, config['counterfactuals']['path'], config['counterfactuals']['sha256'])
    authorize(config, receipt, args.stage, suite.fingerprint, commit)
    if Path(args.output).resolve().is_relative_to(parent.root):
        raise ValueError('run output cannot be inside the parent dataset')
    from knolling_model.runner import seed_everything
    from knolling_model.model import load_pretrained
    seed_everything(config['seed'], config['cpu_threads'])
    base, tokenizer = load_pretrained(config['pretrained'], local_files_only=True)
    model = RecoveryModel(base.text_encoder, base.perception.detector, **config['model']).to(config['device'])
    collator = Collator(tokenizer)
    def batches(stage, epoch):
        for item in stage_batches(suite, collator, stage, seed=config['seed'], epoch=epoch, batch_size=config['batch_size']):
            yield item.to(config['device'])
    def evaluation(stage, baseline):
        def stream():
            for i in range(0,len(suite),config['batch_size']):
                yield collator([suite[j] for j in range(i,min(len(suite),i+config['batch_size']))]).to(config['device'])
        return evaluate_recovery(model, stream(), stage=stage,
            expected_ids=[r['sample_id'] for r in suite.records], baseline=baseline)[0]
    event = threading.Event()
    engine = StageEngine(model, config, suite.fingerprint, commit, args.output, batches, evaluation,
                         stop_event=event, review_receipt=receipt)
    with stop_signals(event):
        result = engine.run(args.stage, parent=args.parent_checkpoint, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result['status'] not in ('passed','paused'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
