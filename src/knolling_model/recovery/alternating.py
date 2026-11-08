"""Opt-in full-set alternating readout/feature diagnostic, never an auto stage.

Approximate variable projection: retain Huber/CE, calibrate a global readout,
then differentiate through features with ALL final Linear rows held fixed.
Legacy integrated checkpoints/journals and their execution path are unchanged.
"""
from copy import deepcopy
from contextlib import contextmanager
import math
from pathlib import Path

import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from knolling_model.coordinates import angle_delta
from .driver import Cursor, cpu_copy, load_checkpoint, restore_rng, rng_state
from .integrated import (IntegratedEngine, InterruptedTrial, HEADS, evidence,
                         load_integrated, retained, timed)
from .readout import collect_readout
from .storage import load_artifact
from .svd_readout import SVDNumericalStall

SCHEMA = 'alternating_pose_v1'
SOURCE_SHA = '5e892f898fc028a54821a0d5898367da593a3aa312fbe37f619faa3ec149dfa5'
POLICY = dict(schema=SCHEMA, source_sha256=SOURCE_SHA, records=76, pairs=35,
              reference_rms_mm=23.453627479249874, relative_improvement=.05,
              max_feature_updates=10, max_seconds=120, head_iterations=50,
              head_seconds=10, head_time_mode='solver', finalize_reserve_seconds=10,
              xy_solver='svd_huber_v1')
XY_POLICY = dict(POLICY, diagnostic='xy_feature_only_v1')
XY_WINDOW_POLICY = dict(POLICY, diagnostic='xy_feature_window_v1')


def validate_config(config):
    p = config.get('alternating_pose')
    if p is None: return
    if (p not in (POLICY, XY_POLICY, XY_WINDOW_POLICY) or any(k in config for k in ('integrated_pose', 'coarse_extension', 'micro_precision'))
            or config['model']['architecture'] != 'recovery_v4_goal_joint'
            or config['stages']['oracle_layout'] != dict(max_updates=10, max_seconds=120)
            or config['layout_lr'] != 1e-4 or config['precision']['oracle_layout'] != 'fp32'):
        raise ValueError('invalid bounded alternating policy')


def load_source(path, config, fingerprint, receipt):
    """Explicit new trial from a pinned terminal source, NOT an exhausted resume."""
    from .provenance import contract_hashes
    grant = (receipt or {}).get('alternating_source', {})
    if (grant.get('approved') is not True or grant.get('sha256') != SOURCE_SHA
            or file_hash(path) != SOURCE_SHA):
        raise PermissionError('alternating source requires explicit pinned approval')
    raw = load_artifact(path); old = raw['config']
    if (raw['commit'] != grant.get('commit') or digest(old) != grant.get('config_sha256')
            or grant.get('contracts') != contract_hashes()
            or grant['contracts'] != contract_hashes(raw['commit'])):
        raise ValueError('alternating source semantic contract mismatch')
    normalized = deepcopy(config); normalized.pop('alternating_pose')
    for k in ('integrated_pose', 'coarse_extension', 'execution', 'allow_training', 'budget_status', 'device'):
        if k in old: normalized[k] = old[k]
        else: normalized.pop(k, None)
    normalized['stages']['oracle_layout'] = old['stages']['oracle_layout']
    if normalized != old: raise ValueError('unapproved alternating training setting change')
    saved = load_checkpoint(path, old, fingerprint, raw['commit'])
    if (saved['stage'] != 'oracle_layout' or saved['status'] != 'coarse_stalled'
            or saved['cursor']['update'] != 600 or saved['integrated']['phase'] != 'coarse'
            or saved['integrated'].get('pending_coarse') is not None
            or saved['integrated'].get('solver') is not None):
        raise ValueError('expected the completed 600-update Coarse source')
    evidence(saved['metrics'])
    return saved


class FeatureAdam:
    """Fresh shared-only Adam for a new algorithm; resume retains its exact state."""
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.shared = {n: p for n, p in model.layout.pose.named_parameters()
                       if n not in ('output.1.weight', 'output.1.bias')}
        self.parameters = list(self.shared.values())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay, foreach=False)
        self.activate()

    def activate(self):
        self.model.requires_grad_(False); self.model.zero_grad(set_to_none=True)
        for p in self.parameters: p.requires_grad_(True)
        self.model.train()

    def state_dict(self):
        return dict(schema=SCHEMA, names=list(self.shared), optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        if state.get('schema') != SCHEMA or state.get('names') != list(self.shared):
            raise ValueError('shared optimizer mapping mismatch')
        self.optimizer.load_state_dict(state['optimizer'])


def retained_trial(now, before):
    """Preserve scalar angle/Keep quality AND each previously correct object."""
    for item in (now, before):
        numbers = [item[k] for k in ('xy_svd_rms_mm', 'hv_relative_error_mm', 'angle_p95_deg',
                                    'keep_rms_mm', 'keep_angle_rms_deg')]
        numbers += [v for row in item['keep'].values() for v in row.values()]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in numbers):
            return False
    if set(now['keep']) != set(before['keep']): return False
    if (now['angle_p95_deg'] > before['angle_p95_deg']+1e-5
            or now['bin_correct'] < before['bin_correct']
            or now['keep_rms_mm'] > before['keep_rms_mm']+1e-4
            or now['keep_angle_rms_deg'] > before['keep_angle_rms_deg']+1e-5): return False
    return all(not (old['xy_mm'] <= 1 and old['angle_deg'] <= 1)
               or (now['keep'][key]['xy_mm'] <= 1+1e-4 and now['keep'][key]['angle_deg'] <= 1+1e-5)
               for key, old in before['keep'].items())


def trial_gate(current, baseline):
    values = [current[k] for k in ('xy_svd_rms_mm', 'hv_relative_error_mm', 'angle_p95_deg', 'keep_rms_mm')]
    return (all(math.isfinite(v) and v >= 0 for v in values)
            and current['xy_svd_rms_mm'] <= POLICY['reference_rms_mm']*.95
            and current['xy_svd_rms_mm'] <= baseline['xy_svd_rms_mm']*.95
            and current['hv_relative_error_mm'] < baseline['hv_relative_error_mm']-1e-6
            and retained_trial(current, baseline))


@contextmanager
def alternating_evaluation(model):
    """Use the Driver's FP32 dispatch in reloads too, without building a graph.

    CUDA Linear dispatch can depend on leaf requires_grad even under no_grad.
    A fully frozen reload therefore need not be bitwise equal to FeatureAdam's
    eval forward. Normalize those flags, then restore flags/modes/gradients.
    This is evaluation plumbing only: no optimizer, label input or gate change.
    """
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.requires_grad_(False)
        for name, p in model.layout.pose.named_parameters():
            if name not in ('output.1.weight', 'output.1.bias'): p.requires_grad_(True)
        model.eval()
        with torch.no_grad(): yield
    finally:
        for p, flag in flags: p.requires_grad_(flag)
        for m, training in modes: m.training = training


def evaluate_alternating(model, batches, **kwargs):
    """Use for standalone Checkpoint reloads as well as Driver evaluation."""
    from .evaluation_v4 import evaluate_v4
    with alternating_evaluation(model):
        return evaluate_v4(model, batches, **kwargs)


@torch.no_grad()
def trial_measurements(model, batches, expected_ids, fingerprint, pairs, check=lambda: None):
    """Real forward outputs, ID-aligned contrasts, and the original RMS diagnostic."""
    from .coarse_extension import readout_diagnostic
    captured, batch_list = [], list(batches)
    hook = model.layout.pose.register_forward_hook(lambda m, a, o: captured.append(o))
    try:
        with alternating_evaluation(model):
            data = collect_readout(model, batch_list, expected_ids=expected_ids, fingerprint=fingerprint, check=check)
    finally: hook.remove()
    rows, keep = {}, {}
    for batch, out in zip(batch_list, captured):
        for i, (meta, truth) in enumerate(zip(batch.metadata, batch.supervision)):
            for j, oid in enumerate(meta['object_ids']):
                key = (meta['sample_id'], oid)
                p, y = out['xy'][i, j].cpu().double(), truth['target_xy'][j].cpu().double()
                ae = float(angle_delta(out['angle'][i, j], truth['target_angle'][j]).abs()*180/math.pi)
                rows[key] = (p, y, ae)
                if meta['record']['supervision_v3']['copy_required'][oid]:
                    keep['/'.join(key)] = dict(xy_mm=float((p-y).norm()*1000), angle_deg=ae)
    if set(rows) != {(r[0], r[1]) for r in data.mapping}: raise ValueError('prediction/readout identity mismatch')
    hv = []
    for pair in pairs:
        if pair['factor'] != 'axis_alignment': continue
        a, b = pair['sample_ids']; ids = [oid for sid, oid in rows if sid == a]
        if not ids or set(ids) != {oid for sid, oid in rows if sid == b}: raise ValueError('H/V object mismatch')
        dp = torch.stack([rows[b, oid][0]-rows[a, oid][0] for oid in ids])
        dy = torch.stack([rows[b, oid][1]-rows[a, oid][1] for oid in ids])
        error = (dp-dp.mean(0))-(dy-dy.mean(0))
        hv.append(float(error.square().sum(-1).mean()))
    if not hv: raise ValueError('missing H/V evaluation pairs')
    diag = readout_diagnostic(model, data)
    return dict(xy_svd_rms_mm=diag['xy_svd_rms_mm'], rank=diag['rank'],
        hv_relative_error_mm=math.sqrt(sum(hv)/len(hv))*1000, hv_pairs=len(hv),
        angle_p95_deg=float(torch.quantile(torch.tensor([r[2] for r in rows.values()], dtype=torch.float64), .95)),
        bin_correct=diag['angle_bin_correct'], keep=keep,
        keep_rms_mm=math.sqrt(sum(v['xy_mm']**2 for v in keep.values())/max(1, len(keep))),
        keep_angle_rms_deg=math.sqrt(sum(v['angle_deg']**2 for v in keep.values())/max(1, len(keep))))


class AlternatingEngine(IntegratedEngine):
    """Reuse atomic storage, readout journals, caches, timing and exact RNG restore."""
    @property
    def xy_only(self):
        return self.config['alternating_pose'].get('diagnostic') in ('xy_feature_only_v1', 'xy_feature_window_v1')

    @property
    def provisional_xy(self):
        return self.config['alternating_pose'].get('diagnostic') == 'xy_feature_window_v1'

    def measured(self):
        with alternating_evaluation(self.model):
            return super().measured()

    @timed('evaluation')
    def diagnostics(self):
        rng = rng_state()
        try:
            return trial_measurements(self.model, self.e.readout_batches(), self.e.expected_ids,
                                      self.e.fingerprint, self.e.pairs)
        finally: restore_rng(rng)

    def anchor_current(self, metrics, diagnostic):
        self.anchor = cpu_copy(self.model.state_dict()); self.baseline = metrics
        self.state['accepted_trial'] = deepcopy(diagnostic)
        self.state['anchor_optimizer'] = cpu_copy(self.adam.state_dict())
        self.state['anchor_rng'] = rng_state()
        self.state['anchor_update'] = self.cursor.update
        self.serial += 1
        path = self.e.output/f'phase-anchor-{self.serial:04d}.pt'
        self.e.storage.save_anchor(path, self.anchor, self.stage)
        self.anchor_sha = self.immutable_hash(path); self.state['anchor_path'] = str(path.resolve())

    def rollback(self, metrics, reason):
        self.model.load_state_dict(self.anchor, strict=True)
        self.adam.load_state_dict(self.state['anchor_optimizer']); restore_rng(self.state['anchor_rng'])
        self.model.zero_grad(set_to_none=True)
        self.state.update(solver=None, pending_head=None, pending_feature=False,
                          rejected_metrics=metrics, rolled_back_to_update=self.state['anchor_update'])
        self.solver = None
        # Attempted feature updates remain charged; restored model/optimizer are
        # explicitly tied to the accepted anchor, not mislabeled as new weights.
        return self.checkpoint('rolled_back', self.baseline, reason)

    def phase(self, name, *, reuse=False):
        self.charge_time()
        self.state.update(phase=name, phase_update=0, phase_seconds=0., solver=None,
                          pending_head=None, pending_feature=False, head_diagnostic=None)
        if not reuse:
            self.cache = None
            self.state.update(cache_path=None, cache_key=None, cache_sha256=None)
        self.solver = None; self.state['timing']['phase'] = {}; self.phase_clock = self.elapsed()
        self.checkpoint()
        self.observe('phase_transition', next_phase=name)

    def assess_complete(self):
        metrics = self.measured(); diagnostic = self.diagnostics()
        self.state['last_trial'] = diagnostic
        if self.provisional_xy:
            # Finite, fully recalibrated candidates may explore through temporary
            # regressions. Never replace the original rollback anchor with them.
            if not retained_trial(diagnostic, diagnostic):
                raise FloatingPointError('nonfinite XY candidate diagnostic')
            task_ok = retained(metrics, self.baseline, angles=True)
            pose_ok = retained_trial(diagnostic, self.state['reference_trial'])
            passed = (self.cursor.update > 0 and task_ok and pose_ok
                      and trial_gate(diagnostic, self.state['reference_trial']))
            old, now = evidence(self.baseline), evidence(metrics)
            candidate = dict(feature_update=self.cursor.update, diagnostic=deepcopy(diagnostic),
                task_retained=task_ok, pose_retained=pose_ok, passed=passed,
                lost_scenes=sorted(set(old['successful_scenes'])-set(now['successful_scenes'])),
                lost_pairs=sorted(set(old['successful_pairs'])-set(now['successful_pairs'])),
                keep_violations=now['keep_violations'])
            self.state.setdefault('candidate_history', []).append(candidate)
            self.observe('candidate_evaluation', candidate=candidate)
            if passed:
                return self.checkpoint('passed', metrics)
            if self.cursor.update >= self.policy['max_feature_updates']:
                return self.rollback(metrics, 'final XY diagnostic gate failed at feature cap')
            self.phase('feature')
            return None
        if (not retained(metrics, self.baseline, angles=True)
                or not retained_trial(diagnostic, self.state['accepted_trial'])):
            return self.rollback(metrics, 'angle/Keep/task regression after recalibration')
        # The separate XY diagnostic keeps the original orientation/residual
        # heads. Both modes still require measured angle/Keep non-regression.
        if trial_gate(diagnostic, self.state['reference_trial']):
            return self.checkpoint('passed', metrics)
        self.anchor_current(metrics, diagnostic)
        if self.cursor.update >= 10:
            return self.checkpoint('budget_exhausted', metrics, 'feature update cap')
        self.phase('feature')
        return None

    def resolve_head(self):
        pending = self.state['pending_head']
        # Preserve the evaluation-before-next-update journal contract. Partial
        # calibration cannot pass or trigger phase-regression prematurely.
        # The inner solver already verifies its fixed-feature candidate. Avoid
        # a full 76-scene geometry pass after EVERY solver iteration; assess
        # the actual forward once both heads finish. Resume resolves this
        # durable completion decision before any new solver/feature update.
        self.solver.measurements()
        self.state['pending_head'] = None
        if pending['phase_done']:
            if self.state['phase'] == 'head_xy' and not self.xy_only: self.phase('head_angle', reuse=True)
            else: return self.assess_complete()
        return None

    @timed('adam')
    def feature_step(self):
        """One scene-weighted full-76 update; fixed heads, no GT in inputs."""
        self.check(); self.adam.activate(); self.adam.optimizer.zero_grad(set_to_none=True)
        rng = rng_state(); count = 0; totals = {}
        try:
            for batch in self.e.readout_batches():
                self.check()
                terms, _ = self.e.loss_fn(self.model, batch.inputs, batch.supervision)
                n = len(batch.metadata); count += n
                loss = sum(terms.values())*n/len(self.e.expected_ids)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite feature loss')
                loss.backward()
                for k, v in terms.items(): totals[k] = totals.get(k, 0.)+float(v.detach())*n/len(self.e.expected_ids)
            if count != len(self.e.expected_ids): raise ValueError('incomplete feature stream')
            norm = torch.nn.utils.clip_grad_norm_(self.adam.parameters, self.config['gradient_clip'], error_if_nonfinite=True)
            self.check()
        except BaseException:
            restore_rng(rng); self.model.zero_grad(set_to_none=True); raise
        self.adam.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.adam.parameters): raise FloatingPointError('nonfinite feature parameter')
        self.cursor.update += 1
        self.state.update(pending_feature=True, solver=None, cache_path=None, cache_key=None, cache_sha256=None)
        self.cache = self.solver = None
        self.checkpoint()  # Durable before observer, evaluation or another step.
        self.observe('update', loss=sum(totals.values()), losses=totals, gradient_norm=float(norm),
                     remaining_seconds=max(0., 120-self.elapsed()))

    def resolve_feature(self):
        self.measured()  # Resolve a durable feature candidate before new work.
        self.phase('head_xy')

    def head_step(self):
        self.prepare_head()
        complete = self.solver.measurements()['passed']
        if not complete:
            if self.state['phase_update'] >= self.policy['head_iterations']:
                raise InterruptedTrial('inner_not_converged')
            try:
                def count(): self.state['closure_evaluations'] += 1
                with self.measure('solver'):
                    loss, converged = self.solver.step(lambda: self.check(head=True), count)
                self.state['phase_update'] += 1; self.state['solver_updates'] += 1
                complete = converged or self.solver.measurements()['passed']
                self.state['head_diagnostic'] = dict(self.solver.measurements(), loss=loss, converged=converged)
            except SVDNumericalStall as exc:
                # A precision-limited accepted head is usable for alternating
                # descent, but is NOT advertised as an exact Huber minimizer.
                complete = True
                self.state['head_diagnostic'] = dict(self.solver.measurements(), reason='fp32_numerical_stall', rounding=exc.diagnostic)
        if complete:
            variables = [self.solver.z] if hasattr(self.solver, 'z') else [self.solver.weight, self.solver.bias]
            gradients = torch.autograd.grad(self.solver.loss(), variables)
            self.state['calibrations'].append(dict(feature_update=self.cursor.update,
                phase=self.state['phase'], solver_updates=self.state['phase_update'],
                gradient_inf=max(float(g.abs().max()) for g in gradients),
                diagnostic=deepcopy(self.state.get('head_diagnostic') or self.solver.measurements())))
        self.state['pending_head'] = dict(phase_done=bool(complete))
        path = self.journal()
        self.observe('head_candidate', journal=str(path), solver_updates=self.state['solver_updates'])
        return self.resolve_head()

    def run(self, stage, *, parent=None, resume=None, extend_from=None, micro_case=None, micro_checkpoints=None):
        e = self.e; validate_config(self.config)
        if (stage != 'oracle_layout' or extend_from or micro_case or micro_checkpoints
                or bool(parent) == bool(resume) or len(e.expected_ids or []) != 76
                or len(set(e.expected_ids)) != 76 or len(e.pairs or []) != 35):
            raise ValueError('alternating trial requires full 76/35 and one pinned parent or interrupted resume')
        if resume:
            if (Path(resume).parent/'terminal.pt').exists(): raise ValueError('terminal alternating trial cannot resume')
            prior = load_integrated(resume, self.config, e.fingerprint, e.commit)
            if (prior['status'] not in ('running', 'paused') or prior['stage'] != stage
                    or prior['integrated'].get('schema') != SCHEMA):
                raise ValueError('invalid alternating resume')
        else: prior = load_source(parent, self.config, e.fingerprint, e.review_receipt)
        self.stage, self.case = stage, None
        self.policy = self.config['alternating_pose']; self.spec = self.config['stages'][stage]
        self.model.load_state_dict(prior['model'], strict=True); e.configure(self.model, stage)
        self.adam = FeatureAdam(self.model, self.config['layout_lr'], self.config['weight_decay'])
        self.completed = deepcopy(prior['completed']); self.goal_parent = prior['goal_parent_sha256']
        self.micro_evidence = deepcopy(prior['micro_evidence'])
        if resume:
            self.cursor = Cursor(**prior['cursor']); self.state = deepcopy(prior['integrated'])
            if (self.state['phase'] not in ('head_xy', 'head_angle', 'feature') or not 0 <= self.cursor.update <= 10
                    or self.xy_only and self.state['phase'] == 'head_angle'
                    or not 0 <= self.state['phase_update'] <= self.policy['head_iterations']
                    or not 0 <= self.cursor.elapsed_seconds <= 120
                    or type(self.state['pending_feature']) is not bool
                    or self.state['solver_updates'] < 0
                    or self.state['pending_feature'] and self.state['phase'] != 'feature'
                    or self.state['pending_head'] is not None and self.state['phase'] not in HEADS):
                raise ValueError('invalid alternating phase/pending state')
            pending = self.state['pending_head']
            if pending is not None and (not isinstance(pending, dict) or set(pending) != {'phase_done'}
                                        or type(pending['phase_done']) is not bool):
                raise ValueError('invalid pending alternating head decision')
            self.adam.load_state_dict(prior['optimizer']); self.baseline = prior['baseline']
            if file_hash(self.state['anchor_path']) != prior['anchor_sha256']: raise ValueError('alternating anchor hash mismatch')
            self.anchor = load_artifact(self.state['anchor_path'])
        else:
            self.cursor = Cursor()
            self.state = dict(schema=SCHEMA, phase='head_xy', phase_update=0, phase_seconds=0., solver=None,
                solver_updates=0, closure_evaluations=0, pending_head=None, pending_feature=False, calibrations=[],
                cache_path=None, cache_sha256=None, cache_key=None, timing=dict(total={}, phase={}),
                source=dict(path=str(Path(parent).resolve()), sha256=file_hash(parent), cursor=deepcopy(prior['cursor'])))
            self.anchor = cpu_copy(prior['model']); self.baseline = prior['metrics']
        e.output.mkdir(parents=True, exist_ok=False)
        self.elapsed_origin = self.cursor.elapsed_seconds; self.started = e.clock(); self.phase_clock = self.elapsed_origin
        self._timing_mark = e.clock()
        if e.prepare_continuation: e.prepare_continuation(stage)
        restore_rng(prior['rng'])
        # Initial reference and new run clock include diagnostics/cache/setup.
        if not resume:
            reference = self.diagnostics()
            if abs(reference['xy_svd_rms_mm']-POLICY['reference_rms_mm']) > .01:
                raise ValueError('600-update RMS reference does not reproduce')
            self.state['reference_trial'] = reference
            self.anchor_current(prior['metrics'], reference)
        else:
            old_anchor = self.anchor
            path = e.output/'anchor.pt'; e.storage.save_anchor(path, old_anchor, stage)
            self.anchor_sha = self.immutable_hash(path); self.state['anchor_path'] = str(path.resolve())
        try:
            self.observe('start', max_updates=10, max_seconds=120)
            self.checkpoint('running', prior.get('metrics'))
            while True:
                if self.state['pending_feature']:
                    self.resolve_feature(); continue
                if self.state['pending_head'] is not None:
                    if self.solver is None: self.restore_cache()
                    result = self.resolve_head()
                    if result: return result
                    continue
                self.check()
                if self.state['phase'] in HEADS:
                    result = self.head_step()
                    if result: return result
                else:
                    if self.cursor.update >= 10: raise InterruptedTrial('feature update cap')
                    self.feature_step()
        except InterruptedTrial as exc:
            if exc.reason == 'paused': return self.checkpoint('paused')
            # No incomplete calibration may satisfy the experiment Gate. Restore
            # the last fully evaluated state, including optimizer and RNG.
            try: metrics = self.measured()
            except Exception: metrics = None
            return self.rollback(metrics, exc.reason)
        except Exception as exc:
            return self.rollback(None, f'{type(exc).__name__}: {exc}')
        finally:
            self.charge_time()
            from .driver import save_json_exclusive
            save_json_exclusive(e.output/'timing.json', dict(elapsed_seconds=self.elapsed(),
                max_seconds=120, feature_updates=self.cursor.update, solver_updates=self.state['solver_updates'],
                timing=deepcopy(self.state['timing'])))
