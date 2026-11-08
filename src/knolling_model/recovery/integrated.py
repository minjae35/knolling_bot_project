"""Bounded A/B1/B2/C/D1/D2 training, opt-in and disabled in shipped configs.

Only micro_pose and oracle_layout use this engine. Other stages retain the
existing driver. Atomic accepted-head journals precede observer/report calls.
"""
from copy import deepcopy
from contextlib import contextmanager
from functools import wraps
import math
from pathlib import Path

import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from .driver import (CHECKPOINT_SCHEMA, Cursor, cpu_copy, load_checkpoint,
                     load_grounding_transfer, restore_rng, rng_state, save_exclusive,
                     validate_micro_checkpoints, validate_checkpoint)
from .storage import load_artifact
from .svd_readout import SVDNumericalStall
from .readout import HeadFitter, PoseAdam, ReadoutData, collect_readout, make_head_fitter

PHASES = ('coarse', 'head_xy', 'head_angle', 'fine', 'final_xy', 'final_angle')
HEADS = {'head_xy': 'xy', 'head_angle': 'angle', 'final_xy': 'xy', 'final_angle': 'angle'}
SCHEMA = 'integrated_pose_v1'
JOURNAL = 'integrated_readout_journal_v1'


def timed(category):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self.measure(category):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


def validate_config(config):
    policy = config.get('integrated_pose')
    if policy is None: return
    if (policy.get('schema') != SCHEMA or config['model']['architecture'] != 'recovery_v4_goal_joint'
            or config.get('micro_precision') is not None or config['layout_lr'] != 1e-4
            or set(policy) - {'schema', 'stages', 'head_seconds', 'head_time_mode', 'finalize_reserve_seconds', 'xy_solver'}
            or policy.get('xy_solver', 'lbfgs_v1') not in ('lbfgs_v1', 'svd_huber_v1')
            or not {'schema', 'stages', 'head_seconds'} <= set(policy)
            or policy.get('head_time_mode', 'wall') not in ('wall', 'solver')
            or type(policy.get('finalize_reserve_seconds', 0)) not in (int, float)
            or not math.isfinite(policy.get('finalize_reserve_seconds', 0))
            or policy.get('finalize_reserve_seconds', 0) < 0
            or type(policy['head_seconds']) not in (float, int) or not 0 < policy['head_seconds'] <= 10
            or set(policy['stages']) != {'micro_pose', 'oracle_layout'}):
        raise ValueError('invalid integrated pose policy')
    for stage, caps in policy['stages'].items():
        limit = config['micro_pose'] if stage == 'micro_pose' else config['stages'][stage]
        if (tuple(caps) != PHASES or any(type(v) is not int or v <= 0 for v in caps.values())
                or sum(caps.values()) != limit['max_updates']
                or policy.get('finalize_reserve_seconds', 0) >= limit['max_seconds']
                or config.get('precision', {}).get(stage, 'fp32') != 'fp32'):
            raise ValueError('integrated phase budgets/precision mismatch')


def evidence(metrics):
    e = metrics.get('integrated_evidence')
    if (metrics.get('complete_set') is not True or not e or
            not {'sample_ids', 'successful_scenes', 'successful_pairs', 'keep_violations',
                 'max_position_mm', 'max_angle_deg', 'angle_errors_deg'} <= e.keys()
            or not e['sample_ids'] or len(set(e['sample_ids'])) != len(e['sample_ids'])
            or any(not isinstance(e[k], (int, float)) or not math.isfinite(e[k]) or e[k] < 0
                   for k in ('max_position_mm', 'max_angle_deg', 'keep_violations'))):
        raise ValueError('incomplete integrated evaluation evidence')
    return e


def near(metrics):
    e = evidence(metrics)
    return e['max_position_mm'] <= 10 and e['max_angle_deg'] <= 2


def retained(current, previous, *, angles=False):
    a, b = evidence(current), evidence(previous)
    if (set(a['sample_ids']) != set(b['sample_ids'])
            or not set(b['successful_scenes']) <= set(a['successful_scenes'])
            or not set(b['successful_pairs']) <= set(a['successful_pairs'])
            or a['keep_violations'] > b['keep_violations']): return False
    if angles:
        # Retain previously correct orientations; allow improvement of failures.
        if set(a['angle_errors_deg']) != set(b['angle_errors_deg']): return False
        return all(a['angle_errors_deg'][k] <= 2 + 1e-6 for k, v in b['angle_errors_deg'].items() if v <= 2)
    return True


def load_integrated(path, config, fingerprint, commit):
    """A journal references an immutable, contract-validated full checkpoint."""
    raw = load_artifact(path)
    if raw.get('schema') != JOURNAL:
        return validate_checkpoint(raw, config, fingerprint, commit)
    base = Path(raw['base'])
    if base.parent.resolve() != Path(path).parent.resolve() or file_hash(base) != raw['base_sha256']:
        raise ValueError('readout journal base mismatch')
    saved = load_checkpoint(base, config, fingerprint, commit)
    if saved.get('integrated', {}).get('phase') != raw['integrated']['phase']:
        raise ValueError('journal phase mismatch')
    from .readout import SLICES
    s = SLICES[HEADS[raw['integrated']['phase']]]
    for name in ('weight', 'bias'):
        saved['model']['layout.pose.output.1.' + name][s].copy_(raw[name])
    for key in ('integrated', 'cursor', 'rng'):
        saved[key] = raw[key]
    # Older journals also precede evaluation. Recover their pending decision
    # without modifying the artifact or relaxing code/config/data validation.
    if 'pending_head' not in saved['integrated']:
        state = saved['integrated']; diagnostic = state.get('head_diagnostic') or {}
        cap = config['integrated_pose']['stages'][saved['stage']][state['phase']]
        state['pending_head'] = dict(phase_done=bool(diagnostic.get('converged')
            or diagnostic.get('passed') or state['phase_update'] >= cap))
    saved['metrics'] = None; saved['status'] = 'running'
    return saved


class InterruptedTrial(Exception):
    def __init__(self, reason): self.reason = reason


class IntegratedEngine:
    def __init__(self, engine):
        self.e = engine
        self.model, self.config = engine.model, engine.config
        self.cache = self.solver = None
        self.serial = 0; self.base = None; self.observer_errors = []
        self._hashes = {}
        self._timing_stack = []

    def charge_time(self):
        now = self.e.clock()
        category = self._timing_stack[-1] if self._timing_stack else 'other'
        for scope in ('total', 'phase'):
            times = self.state['timing'][scope]
            times[category] = times.get(category, 0.) + max(0., now - self._timing_mark)
        self._timing_mark = now

    @contextmanager
    def measure(self, category):
        self.charge_time(); self._timing_stack.append(category)
        device = next(self.model.parameters()).device
        try:
            if category in ('solver', 'adam') and device.type == 'cuda': torch.cuda.synchronize(device)
            yield
        finally:
            if category in ('solver', 'adam') and device.type == 'cuda': torch.cuda.synchronize(device)
            self.charge_time(); self._timing_stack.pop()

    @timed('hashing')
    def immutable_hash(self, path):
        """Trust only files hashed in this process with unchanged identity/metadata.

        Resume still verifies full contents through load_integrated/restore_cache.
        Never cache a digest supplied by an unverified checkpoint or journal.
        """
        path = Path(path).resolve(); s = path.stat()
        stamp = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        prior = self._hashes.get(path)
        if prior is not None:
            if prior[0] != stamp: raise ValueError('immutable artifact changed')
            return prior[1]
        value = file_hash(path)
        after = path.stat()
        if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('immutable artifact changed during hashing')
        self._hashes[path] = (stamp, value)
        return value

    def elapsed(self): return self.elapsed_origin + self.e.clock() - self.started

    def phase_elapsed(self):
        return self.state['phase_seconds'] + self.elapsed() - self.phase_clock

    def capture_state(self):
        self.charge_time()
        state = deepcopy(self.state)
        state['phase_seconds'] = self.phase_elapsed()
        if self.solver: state['solver'] = cpu_copy(self.solver.state_dict())
        return state

    def observe(self, event, **values):
        if self.e.observer is None: return
        try:
            self.e.observer(event, stage=self.stage, update=self.cursor.update,
                phase=self.state['phase'], elapsed_seconds=self.elapsed(), **values)
        except Exception as exc:
            # Diagnostics must never discard accepted learned weights.
            self.observer_errors.append(f'{type(exc).__name__}: {exc}')

    @timed('checkpoint_io')
    def checkpoint(self, status='running', metrics=None, error=None):
        self.cursor.elapsed_seconds = self.elapsed(); self.serial += 1
        path = self.e.output / f'checkpoint-{self.cursor.update:06d}-{self.serial:04d}.pt'
        self.e.storage.save_checkpoint(path, dict(schema=CHECKPOINT_SCHEMA, stage=self.stage, status=status,
            architecture=self.model.architecture, evaluation_schema=self.config['evaluation_schema'],
            config_sha256=digest(self.config), config=deepcopy(self.config),
            dataset_fingerprint=self.e.fingerprint, commit=self.e.commit, review_receipt=self.e.review_receipt,
            model=self.model.state_dict(), optimizer=self.adam.state_dict(),
            cursor=vars(self.cursor).copy(), rng=rng_state(), baseline=self.baseline,
            completed=deepcopy(self.completed), goal_contract=deepcopy(self.config['goal_contract']),
            goal_parent_sha256=self.goal_parent, purpose='diagnostic' if self.stage == 'micro_pose' else 'production',
            micro_case=self.case, micro_evidence=self.micro_evidence, metrics=metrics,
            error=error, anchor_sha256=self.anchor_sha, integrated=self.capture_state(),
            observer_errors=list(self.observer_errors)))
        self.base = path
        result = dict(status=status, checkpoint=str(path), updates=self.cursor.update,
                      elapsed_seconds=self.cursor.elapsed_seconds, metrics=metrics, error=error,
                      phase=self.state['phase'], observer_errors=list(self.observer_errors))
        if status not in ('running', 'paused'):
            save_exclusive(self.e.output/'terminal.pt', dict(status=status, checkpoint=str(path)))
        self.observe('checkpoint', status=status, checkpoint=str(path), error=error)
        return result

    @timed('journal_io')
    def journal(self):
        """Small, resumable head snapshot saved BEFORE observers/evaluation/audits."""
        self.cursor.elapsed_seconds = self.elapsed(); self.serial += 1
        path = self.e.output / f'readout-{self.cursor.update:06d}-{self.serial:04d}.pt'
        save_exclusive(path, dict(schema=JOURNAL, base=str(self.base.resolve()), base_sha256=self.immutable_hash(self.base),
            weight=self.solver.weight.detach().cpu().clone(), bias=self.solver.bias.detach().cpu().clone(),
            integrated=self.capture_state(), cursor=vars(self.cursor).copy(), rng=rng_state()))
        return path

    def check(self, *, head=False):
        if self.e.stop.is_set(): raise InterruptedTrial('paused')
        if self.elapsed() >= self.spec['max_seconds'] - self.policy.get('finalize_reserve_seconds', 0):
            raise InterruptedTrial('time_budget')
        self.charge_time()
        head_seconds = (self.state['timing']['phase'].get('solver', 0.)
                        if self.policy.get('head_time_mode', 'wall') == 'solver' else self.phase_elapsed())
        if head and head_seconds >= self.policy['head_seconds']: raise InterruptedTrial('head_time_budget')

    @timed('evaluation')
    def measured(self):
        metrics = self.e.evaluate(self.stage, self.baseline)
        if set(evidence(metrics)['sample_ids']) != set(self.e.expected_ids):
            raise ValueError('evaluation/calibration scene mismatch')
        self.observe('evaluation', metrics=metrics)
        return metrics

    def rollback(self, metrics, reason):
        self.model.load_state_dict(self.anchor, strict=True)
        self.model.zero_grad(set_to_none=True)
        return self.checkpoint('rolled_back', metrics, reason)

    def assess(self, metrics):
        if self.state['phase'] != 'coarse' and not retained(metrics, self.baseline,
                                                          angles=self.state['phase'] == 'fine'):
            return self.rollback(metrics, 'phase regression')
        if self.e.gate(self.stage, metrics):
            self.completed[self.stage] = metrics
            return self.checkpoint('passed', metrics)
        return None

    def resolve_head(self):
        """Finish an accepted candidate before another update, including resume."""
        pending = self.state['pending_head']
        metrics = self.measured()
        self.state['pending_head'] = None
        done = self.assess(metrics)
        if done: return done
        if pending['phase_done']: return self.advance(metrics)
        return None

    def set_phase(self, phase, metrics):
        self.charge_time()
        # Adjacent XY -> angle phases have identical feature-producing weights.
        reuse = (self.state['phase'], phase) in (('head_xy', 'head_angle'), ('final_xy', 'final_angle'))
        cached = {k: self.state[k] for k in ('cache_path', 'cache_sha256', 'cache_key')} if reuse else {}
        cache = self.cache if reuse else None
        self.state.update(phase=phase, phase_update=0, phase_seconds=0., solver=None,
                          cache_path=None, cache_sha256=None, cache_key=None, head_diagnostic=None,
                          pending_head=None)
        self.state.update(cached); self.state['timing']['phase'] = {}
        self.phase_clock = self.elapsed()
        self.cache, self.solver = cache, None
        with self.measure('checkpoint_io'):
            self.anchor = cpu_copy(self.model.state_dict()); self.baseline = metrics
            self.serial += 1
            anchor_path = self.e.output / f'phase-anchor-{self.serial:04d}.pt'
            self.e.storage.save_anchor(anchor_path, self.anchor, self.stage)
            self.anchor_sha = self.immutable_hash(anchor_path); self.state['anchor_path'] = str(anchor_path.resolve())
        if phase == 'fine':
            if self.state['fine_resets'] != 0: raise ValueError('duplicate fine reset')
            self.adam.enter_fine(); self.state['fine_resets'] = 1
        elif phase in HEADS:
            self.model.requires_grad_(False); self.model.zero_grad(set_to_none=True); self.model.eval()
        self.checkpoint('running', metrics)
        self.observe('phase_transition', next_phase=phase)

    @timed('feature_cache')
    def restore_cache(self):
        path = Path(self.state['cache_path'])
        if file_hash(path) != self.state['cache_sha256']: raise ValueError('cache artifact hash mismatch')
        raw = torch.load(path, map_location='cpu', weights_only=True)
        self.cache = ReadoutData(**raw)
        self.cache.validate(self.model, self.e.fingerprint, self.e.expected_ids)
        if self.cache.key != self.state['cache_key']: raise ValueError('cache key mismatch')
        self.solver = self.make_solver(self.state['solver'])

    def make_solver(self, state=None):
        return make_head_fitter(self.model, self.cache, HEADS[self.state['phase']], state,
                                xy_solver=self.policy.get('xy_solver', 'lbfgs_v1'))

    @timed('feature_cache')
    def prepare_head(self):
        if self.solver is not None: return
        if self.cache is not None:
            # In-memory reuse only after the controlled frozen-head transition.
            self.solver = self.make_solver()
            return
        if self.state['cache_path']:
            self.restore_cache(); return
        self.cache = collect_readout(self.model, self.e.readout_batches(), expected_ids=self.e.expected_ids,
            fingerprint=self.e.fingerprint, check=lambda: self.check(head=True))
        self.serial += 1; path = self.e.output / f'features-{self.serial:04d}.pt'
        save_exclusive(path, self.cache.payload())
        self.state.update(cache_path=str(path.resolve()), cache_sha256=file_hash(path), cache_key=self.cache.key)
        self.solver = self.make_solver()
        self.state['solver'] = cpu_copy(self.solver.state_dict())
        # Base already captures phase/anchor/optimizer. Publish cache and solver
        # metadata durably without duplicating the full frozen model a second time.
        self.journal()

    @timed('adam')
    def adam_step(self, item):
        self.check(); self.adam.activate(); self.adam.optimizer.zero_grad(set_to_none=True)
        rng = rng_state()
        inputs, truths, replay = item.loss_arguments()
        try:
            with self.adam.forward_context():
                terms, _ = self.e.loss_fn(self.model, inputs, truths, replay=replay)
                loss = sum(terms.values())
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.adam.parameters, self.config['gradient_clip'], error_if_nonfinite=True)
            self.check()
        except BaseException:
            restore_rng(rng); self.adam.optimizer.zero_grad(set_to_none=True)
            raise
        self.adam.optimizer.step(); self.adam.sync()
        if not all(torch.isfinite(p).all() for p in self.adam.parameters):
            raise FloatingPointError('nonfinite parameter')
        return float(loss.detach()), float(norm), {k: float(v.detach()) for k, v in terms.items()}

    def advance(self, metrics):
        self.charge_time()
        phase = self.state['phase']
        self.state['phase_results'].append(dict(phase=phase, updates=self.state['phase_update'],
            elapsed_seconds=self.phase_elapsed(), metrics=metrics,
            readout=deepcopy(self.state.get('head_diagnostic')), timing=deepcopy(self.state['timing']['phase'])))
        if phase == 'head_angle' and not near(metrics):
            return self.checkpoint('fine_ineligible', metrics, 'full-set maximum errors exceed 10mm/2deg')
        index = PHASES.index(phase)
        if index == len(PHASES)-1:
            return self.checkpoint('budget_exhausted', metrics)
        self.set_phase(PHASES[index+1], metrics)
        return None

    def run(self, stage, *, parent=None, resume=None, extend_from=None, micro_case=None, micro_checkpoints=None):
        e = self.e; validate_config(self.config)
        if self.model.architecture != self.config['model']['architecture']:
            raise ValueError('model/config architecture mismatch')
        if stage not in ('micro_pose', 'oracle_layout') or bool(parent) == bool(resume) or extend_from:
            raise ValueError('integrated execution requires exactly one parent or interrupted resume')
        if not e.readout_batches or not e.expected_ids:
            raise ValueError('integrated execution requires complete calibration stream')
        from .evaluation_v4 import MICRO_CASES
        if stage == 'micro_pose':
            if micro_case not in MICRO_CASES or list(e.expected_ids) != list(MICRO_CASES[micro_case]):
                raise ValueError('integrated micro case sample mismatch')
            cases = (e.review_receipt or {}).get('cases')
            if cases is not None and micro_case not in cases: raise PermissionError('micro case not authorized')
        elif micro_case is not None: raise ValueError('micro case is diagnostic-only')
        self.stage, self.case = stage, micro_case
        self.policy = self.config['integrated_pose']; self.caps = self.policy['stages'][stage]
        self.spec = self.config['micro_pose'] if stage == 'micro_pose' else self.config['stages'][stage]
        if resume:
            terminal = Path(resume).parent/'terminal.pt'
            if terminal.exists(): raise ValueError('terminal integrated run cannot resume an older snapshot')
            prior = load_integrated(resume, self.config, e.fingerprint, e.commit)
            if (prior['stage'] != stage or prior['status'] not in ('paused', 'running')
                    or prior.get('micro_case') != micro_case or prior.get('integrated', {}).get('schema') != SCHEMA):
                raise ValueError('only interrupted integrated checkpoints can resume')
        else:
            if (e.review_receipt or {}).get('grounding_transfer'):
                prior = load_grounding_transfer(parent, self.config, e.fingerprint, e.review_receipt)
            else: prior = load_checkpoint(parent, self.config, e.fingerprint, e.commit)
            if prior['stage'] != 'grounding' or prior['status'] != 'passed' or not e.gate('grounding', prior['completed'].get('grounding', {})):
                raise ValueError('integrated stage must start from passed Grounding weights')
        self.completed = deepcopy(prior['completed'])
        self.goal_parent = prior['goal_parent_sha256'] if resume else file_hash(parent)
        self.micro_evidence = deepcopy(prior.get('micro_evidence', {}))
        if stage == 'oracle_layout' and not resume:
            self.micro_evidence = validate_micro_checkpoints(micro_checkpoints, self.goal_parent, self.config, e.fingerprint, e.commit, e.review_receipt)
        self.model.load_state_dict(prior['model'], strict=True); e.configure(self.model, stage)
        self.adam = PoseAdam(self.model, 1e-4, self.config['weight_decay'])
        self.cursor = Cursor(**prior['cursor']) if resume else Cursor()
        self.state = deepcopy(prior['integrated']) if resume else dict(schema=SCHEMA, phase='coarse',
            phase_update=0, phase_seconds=0., closure_evaluations=0, fine_resets=0,
            cache_path=None, cache_sha256=None, cache_key=None, solver=None, anchor_path=None,
            phase_results=[], head_diagnostic=None, pending_head=None)
        self.state.setdefault('pending_head', None)
        self.state.setdefault('timing', dict(total={}, phase={}))
        pending = self.state['pending_head']
        if pending is not None and (self.state['phase'] not in HEADS or not isinstance(pending, dict)
                or set(pending) != {'phase_done'} or type(pending['phase_done']) is not bool):
            raise ValueError('invalid pending head decision')
        if (self.state['phase'] not in PHASES or not 0 <= self.state['phase_update'] <= self.caps[self.state['phase']]
                or self.state['fine_resets'] != int(self.state['phase'] in ('fine', 'final_xy', 'final_angle'))
                or not 0 <= self.cursor.update <= self.spec['max_updates']):
            raise ValueError('invalid integrated phase cursor')
        if resume:
            self.adam.load_state_dict(prior['optimizer'])
            expected_lr = 1e-6 if self.state['fine_resets'] else 1e-4
            if any(g['lr'] != expected_lr for g in self.adam.optimizer.param_groups):
                raise ValueError('integrated optimizer phase mismatch')
            anchor_path = Path(self.state['anchor_path'])
            if file_hash(anchor_path) != prior['anchor_sha256']: raise ValueError('phase anchor mismatch')
            self.anchor = load_artifact(anchor_path)
            self.baseline = prior['baseline']
            if e.prepare_continuation: e.prepare_continuation(stage)
            restore_rng(prior['rng'])
        else:
            self.anchor = cpu_copy(self.model.state_dict()); self.baseline = None
        e.output.mkdir(parents=True, exist_ok=False)
        self.elapsed_origin = self.cursor.elapsed_seconds; self.started = e.clock(); self.phase_clock = self.elapsed_origin
        self._timing_mark = e.clock()
        with self.measure('checkpoint_io'):
            e.storage.save_anchor(e.output/'anchor.pt', self.anchor, stage)
            self.anchor_sha = self.immutable_hash(e.output/'anchor.pt'); self.state['anchor_path'] = str((e.output/'anchor.pt').resolve())
        if self.state['phase'] in HEADS:
            self.model.requires_grad_(False); self.model.eval()
        try:
            self.observe('start', max_updates=self.spec['max_updates'], max_seconds=self.spec['max_seconds'])
            if not resume:
                metrics = self.measured(); self.baseline = metrics
                if self.config.get('stop_on_initial_pass', True):
                    done = self.assess(metrics)
                    if done: return done
                self.checkpoint('running', metrics)
                if near(metrics):
                    self.set_phase('head_xy', metrics)
            else:
                self.checkpoint('running', prior.get('metrics'))
            while True:
                # Journals (and a checkpoint made while resuming one) may hold
                # an accepted but unassessed candidate. Resolve it even if the
                # stop/deadline is now set; never issue another solver update.
                if self.state['pending_head'] is not None:
                    if self.solver is None and self.state['cache_path']:
                        self.restore_cache()
                    done = self.resolve_head()
                    if done: return done
                    continue
                self.check()
                phase = self.state['phase']
                if self.cursor.update >= self.spec['max_updates']:
                    metrics = self.measured()
                    return self.assess(metrics) or self.checkpoint('budget_exhausted', metrics)
                if phase in HEADS:
                    try:
                        self.prepare_head()
                        if self.solver.measurements()['passed'] or self.state['phase_update'] >= self.caps[phase]:
                            phase_done = True
                            self.state['head_diagnostic'] = dict(self.solver.measurements(), reason='criterion_or_iteration_cap')
                        else:
                            def count(): self.state['closure_evaluations'] += 1
                            with self.measure('solver'):
                                loss, converged = self.solver.step(lambda: self.check(head=True), count)
                            self.cursor.update += 1; self.state['phase_update'] += 1
                            self.state['head_diagnostic'] = dict(self.solver.measurements(), loss=loss,
                                reason='converged' if converged else 'accepted', converged=converged)
                            phase_done = converged or self.state['head_diagnostic']['passed'] or self.state['phase_update'] >= self.caps[phase]
                            self.state['pending_head'] = dict(phase_done=bool(phase_done))
                            path = self.journal()
                            self.observe('update', loss=loss, gradient_norm=None, losses={HEADS[phase]: loss},
                                         remaining_seconds=max(0., self.spec['max_seconds']-self.elapsed()),
                                         journal=str(path), closure_evaluations=self.state['closure_evaluations'])
                    except SVDNumericalStall as exc:
                        # Solver has restored the accepted head/coordinates/Adam-free state.
                        # Publish the terminal head decision BEFORE evaluation, without an update.
                        self.state['head_diagnostic'] = dict(self.solver.measurements(),
                            reason='fp32_numerical_stall', rounding=exc.diagnostic)
                        phase_done = True
                        self.state['pending_head'] = dict(phase_done=True)
                        path = self.journal()
                        self.observe('head_stalled', journal=str(path), diagnostic=self.state['head_diagnostic'])
                    except InterruptedTrial as exc:
                        if exc.reason != 'head_time_budget': raise
                        self.state['head_diagnostic'] = dict(reason='head_time_budget',
                            **(self.solver.measurements() if self.solver else {}))
                        phase_done = True
                    # Every accepted candidate is durable before any evaluation.
                    self.state['pending_head'] = dict(phase_done=bool(phase_done))
                    done = self.resolve_head()
                    if done: return done
                else:
                    if self.state['phase_update'] >= self.caps[phase]:
                        metrics = self.measured(); done = self.assess(metrics) or self.advance(metrics)
                        if done: return done
                        continue
                    seen = False
                    for index, item in enumerate(e.batches(stage, self.cursor.epoch)):
                        if index < self.cursor.batch: continue
                        seen = True
                        loss, norm, terms = self.adam_step(item)
                        self.cursor.batch = index+1; self.cursor.update += 1; self.state['phase_update'] += 1
                        self.observe('update', loss=loss, gradient_norm=norm, losses=terms,
                                     remaining_seconds=max(0., self.spec['max_seconds']-self.elapsed()))
                        evaluate_now = (stage == 'micro_pose' or self.cursor.update % self.config['evaluate_every'] == 0
                                        or self.state['phase_update'] >= self.caps[phase])
                        if evaluate_now:
                            metrics = self.measured(); done = self.assess(metrics)
                            if done: return done
                            if (phase == 'coarse' and near(metrics)) or self.state['phase_update'] >= self.caps[phase]:
                                done = self.advance(metrics)
                                if done: return done
                                break
                        if self.cursor.update % self.config['checkpoint_every'] == 0: self.checkpoint()
                    else:
                        if not seen and self.cursor.batch == 0: raise ValueError('empty training stream')
                        self.cursor.epoch += 1; self.cursor.batch = 0
        except InterruptedTrial as exc:
            if exc.reason == 'paused': return self.checkpoint('paused')
            metrics = self.measured()
            return self.assess(metrics) or self.checkpoint('budget_exhausted', metrics, exc.reason)
        except Exception as exc:
            return self.rollback(None, f'{type(exc).__name__}: {exc}')
        finally:
            self.charge_time()
            from .driver import save_json_exclusive
            try:
                save_json_exclusive(e.output/'timing.json', dict(
                    elapsed_seconds=self.elapsed(), timing=deepcopy(self.state['timing']),
                    head_time_mode=self.policy.get('head_time_mode', 'wall'),
                    max_seconds=self.spec['max_seconds'],
                    finalize_reserve_seconds=self.policy.get('finalize_reserve_seconds', 0)))
            except Exception as exc:
                self.observe('timing_report_error', error=f'{type(exc).__name__}: {exc}')
