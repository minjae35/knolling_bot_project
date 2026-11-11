"""One pose-loss replacement. No architecture, selector, data, or inference change."""
import run_phase1_size_control as b
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from knolling_model.data import supervision
from knolling_model.experiments.direct_center import oracle_batch
from knolling_model.experiments.neural_sequential import SequentialPose, teacher_order
from knolling_model.experiments.offset_supervision import ANCHOR_X, x_offset_loss
from knolling_model.experiments.phase1_single_row import position_loss
from verify_phase1_single_row import arguments

ROOT = Path('artifacts/phase1_offset_supervision')
CONTROL = Path('artifacts/phase1_size_diversity_B_environment_fixed_once')
DIAGNOSIS = Path('artifacts/phase1_step_decomposition/summary.json')
PLAN = Path('configs/phase1_offset_supervision.json')


def objective(model, records, device):
    objects, text, mask, targets, _, _, _, ids = oracle_batch(model, records, device)
    out = model.layout(objects, text, mask)
    order = teacher_order(records, ids, objects.valid.shape[1], device)
    goal, selected, move = out['goal']['vector'], out['goal']['selected'], out['move_mask']
    ce = model.layout.pose.selector_loss(objects, goal, selected, move, order)
    # The prefix forward is supervision only and must not move the RNG stream the control used.
    rng = b.rng_state()
    try:
        prefix = model.layout.pose.decoder(objects, goal, selected, move, order, teacher_xy=targets['xy'].detach())
    finally:
        b.restore_rng(rng)
    pose = x_offset_loss(prefix, order, targets['xy'])
    with torch.no_grad():
        monitor = position_loss(out, targets['xy'])
    return ce + pose, ce, pose, monitor, out


def trace_prefix(decoder, objects, goal, selected, move, order, teacher_xy):
    """Read the decoder's own previous-x anchor. Checked against decoder.forward."""
    h = decoder.features(objects, goal, selected)
    b, n, d = h.shape
    rows = torch.arange(b, device=h.device)
    valid = objects.valid
    scale = h.new_tensor([.6, .4])
    previous_xy = h.new_tensor([.3, .2]).expand(b, -1)
    previous_h = h.new_zeros(b, d)
    scene_xy = objects.xy.clone()
    output_xy = objects.xy.clone()
    placed = valid & ~move
    anchors = []
    for step in range(n):
        active = order[:, step] >= 0
        chosen = order[:, step].clamp_min(0)
        current = h[rows, chosen]
        state = torch.cat((scene_xy / scale, placed[..., None], move[..., None]), -1)
        keys = torch.cat((h + decoder.state(state), decoder.null.expand(b, -1, -1)), 1)
        padding = torch.cat((~valid, torch.zeros(b, 1, device=h.device, dtype=torch.bool)), 1)
        context = decoder.attention((current + decoder.goal_query(goal))[:, None], keys, keys,
                                    key_padding_mask=padding, need_weights=False)[0][:, 0]
        delta = .1 * decoder.output(torch.cat((current, context, previous_h, goal, previous_xy / scale), -1))
        predicted = previous_xy + delta
        where = F.one_hot(chosen, n).bool() & active[:, None]
        output_xy = torch.where(where[..., None], predicted[:, None], output_xy)
        if bool(active[0]):
            anchors.append((step, int(chosen[0]), float(previous_xy[0, 0])))
        feedback = teacher_xy[rows, chosen]
        scene_xy = torch.where(where[..., None], feedback[:, None], scene_xy)
        previous_xy = torch.where(active[:, None], feedback, previous_xy)
        previous_h = torch.where(active[:, None], current, previous_h)
        placed = placed | where
    return output_xy.masked_fill(~valid[..., None], 0), anchors


def half_x(size, theta):
    return .5 * (size[0] * abs(math.cos(theta)) + size[1] * abs(math.sin(theta)))


def edge_mm(left, right, instruction):
    gap = (right[0] - half_x(right[2:], right[1])) - (left[0] + half_x(left[2:], left[1]))
    return abs(gap - instruction) * 1000


def step_gaps(model, records, device):
    """Autonomous, teacher-order, and Teacher-prefix gap error by generation step."""
    grouped = {n: {step: {k: [] for k in ('free', 'teacher_order', 'teacher_prefix')} for step in range(1, n)}
               for n in (2, 3, 4)}
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            for record in records:
                n = len(record['source']['objects'])
                if n not in grouped:
                    continue
                obj, text, mask, target, _ = arguments(model, [record], device)
                out = model.layout(obj, text, mask)
                ids = sorted(o['object_id'] for o in record['source']['objects'])
                order = teacher_order([record], [ids], obj.valid.shape[1], device)
                goal, selected, move = out['goal']['vector'], out['goal']['selected'], out['move_mask']
                decoder = model.layout.pose.decoder
                roll = decoder(obj, goal, selected, move, order)
                local = decoder(obj, goal, selected, move, order, teacher_xy=target)
                teacher = [int(v) for v in order[0].tolist() if v >= 0]
                lookup = {oid: i for i, oid in enumerate(ids)}
                neighbors = [(lookup[a], lookup[b]) for a, b in record['supervision_v3']['horizontal_neighbors']]
                if list(zip(teacher, teacher[1:])) != neighbors:
                    raise ValueError('teacher order is not the left-to-right row')
                _, labels = supervision(record)
                gt_angle = labels['target_angle']
                size = obj.size[0].detach().cpu()
                pred_angle = out['angle'][0].detach().cpu()
                modes = {'free': out['xy'][0].detach().cpu(), 'teacher_order': roll[0].detach().cpu(),
                         'teacher_prefix': local[0].detach().cpu()}
                gap_m = float(record['instruction']['gap_m'])
                gt = target[0].detach().cpu()
                for left, right in neighbors:
                    step = teacher.index(right)
                    for name, xy in modes.items():
                        if name == 'teacher_prefix':
                            left_pose = (float(gt[left, 0]), float(gt_angle[left]), float(size[left, 0]), float(size[left, 1]))
                        else:
                            left_pose = (float(xy[left, 0]), float(pred_angle[left]), float(size[left, 0]), float(size[left, 1]))
                        right_pose = (float(xy[right, 0]), float(pred_angle[right]), float(size[right, 0]), float(size[right, 1]))
                        grouped[n][step][name].append(edge_mm(left_pose, right_pose, gap_m))
    finally:
        model.train(was)
    report = {}
    for n, steps in grouped.items():
        report[str(n)] = {}
        for step, modes in steps.items():
            report[str(n)][str(step)] = {name: None if not values else float(np.median(values))
                                         for name, values in modes.items()}
    return report


def mean_median(report, n, mode):
    values = [report[str(n)][step][mode] for step in report[str(n)]]
    if any(v is None for v in values):
        raise ValueError('missing step gap')
    return float(np.mean(values))


def diagnosis_medians():
    raw = json.loads(DIAGNOSIS.read_text())
    found = {}
    for n, block in raw['by_count'].items():
        found[n] = {}
        for row in block['steps']:
            if 'gap' not in row:
                continue
            found[n][str(row['step'])] = dict(teacher_order=row['gap']['rollout_abs_mm']['median'],
                                              teacher_prefix=row['gap']['local_abs_mm']['median'])
    return found


def code():
    paths = [__file__, str(PLAN), 'src/knolling_model/experiments/offset_supervision.py',
             'src/knolling_model/experiments/neural_sequential.py']
    return dict(b.code_hashes(), **{p: b.file_hash(p) for p in paths})


def preflight():
    plan = json.loads(PLAN.read_text())
    p, c, ds, schedule = b.contracts()
    b.require(plan['max_updates'] == 2250 and plan['max_driver_seconds'] == 1200, 'budget changed')
    b.require(plan['huber_scale_m'] == .01 and plan['anchor_x_m'] == ANCHOR_X, 'offset scale changed')
    b.require(not (ROOT / 'execution.json').exists(), 'experiment already started')
    env = b.environment_receipt()
    hist = [json.loads(line) for line in (CONTROL / 'history.jsonl').read_text().splitlines()]
    exposure = b.Exposure(791, 1203)
    visits = [0] * 791
    for item, row in zip(schedule, hist):
        records = b.paired_records(item, exposure, visits, ds)
        b.require([r['sample_id'] for r in records] == row['sample_ids'], 'control exposure changed')
    b.require(len(hist) == 2250 and sum(visits) == 35595, 'control exposure length')
    _, val = b.datasets(c)
    b.require(len(val.records) == 94 and all(r['split'] == 'validation' for r in val.records), 'validation split')
    torch.set_num_threads(4)
    torch.manual_seed(1203)
    initial = b.load_artifact(p['initial_checkpoint'])
    model = b.build(c, 'cpu')
    model.arrangement.load_state_dict(initial['model'], strict=True)
    initial_digest = b.tensor_digest(model.arrangement.state_dict())
    frozen = b.frozen_digest(model)
    probes = [ds.record_for_visit(next(i for i in range(len(ds)) if len(ds.record_for_visit(i, 0)['source']['objects']) == n), 1)
              for n in (2, 3, 4)]
    decoder = model.layout.pose.decoder
    all_anchors = []
    for row in range(len(probes)):
        one = [probes[row]]
        o, t, m, y, _, _, _, one_ids = oracle_batch(model, one, 'cpu')
        layout = model.layout(o, t, m)
        one_order = teacher_order(one, one_ids, o.valid.shape[1], 'cpu')
        got, row_anchors = trace_prefix(decoder, o, layout['goal']['vector'], layout['goal']['selected'],
                                        layout['move_mask'], one_order, y['xy'].detach())
        live = decoder(o, layout['goal']['vector'], layout['goal']['selected'], layout['move_mask'],
                       one_order, teacher_xy=y['xy'].detach())
        b.require(torch.equal(got, live), 'single-scene prefix trace drifted')
        free = decoder(o, layout['goal']['vector'], layout['goal']['selected'], layout['move_mask'],
                       layout['predicted_order'])
        b.require(torch.equal(free, layout['xy']), 'inference xy differs from the autonomous decoder')
        cursor = ANCHOR_X
        for step, chosen, anchor in row_anchors:
            b.require(abs(anchor - cursor) < 1e-6, f'anchor {anchor} != fixed previous {cursor} at step {step}')
            cursor = float(y['xy'][0, chosen, 0])
            all_anchors.append(anchor)
        pose = x_offset_loss(live, one_order, y['xy'])
        b.require(torch.isfinite(pose), 'nonfinite offset')
        y_only = live.detach().clone()
        y_only[..., 1] += 3
        b.require(torch.equal(x_offset_loss(y_only, one_order, y['xy']), pose.detach()), 'Y entered the offset loss')
    model.zero_grad(set_to_none=True)
    total, ce, pose, monitor, out = objective(model, probes, 'cpu')
    selector = list(model.layout.pose.selector.parameters())
    decoder_params = [p for p in model.layout.pose.decoder.parameters() if p.requires_grad]
    pose.backward(retain_graph=True)
    b.require(all(p.grad is None for p in selector), 'X-offset loss reached the selector')
    b.require(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in decoder_params),
              'X-offset loss has no decoder gradient')
    b.require(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'frozen gradient')
    model.zero_grad(set_to_none=True)
    ce.backward(retain_graph=True)
    selector_only = [None if p.grad is None else p.grad.detach().clone() for p in selector]
    model.zero_grad(set_to_none=True)
    total.backward()
    for alone, param in zip(selector_only, selector):
        torch.testing.assert_close(param.grad, alone)
    b.require(b.frozen_digest(model) == frozen, 'preflight changed frozen weights')
    # Reproduce the saved CPU diagnosis with this same gap function before any update.
    control = b.load_artifact(CONTROL / 'checkpoint-002250.pt')
    b.require(control['update'] == 2250 and control['architecture'] == 'neural_sequential_xy_v1', 'control checkpoint')
    model.arrangement.load_state_dict(control['model'], strict=True)
    measured = step_gaps(model, val.records, 'cpu')
    reference = diagnosis_medians()
    for n, steps in reference.items():
        for step, modes in steps.items():
            for mode, expected_value in modes.items():
                got = measured[n][step][mode]
                b.require(abs(got - expected_value) < .05, f'control step gap drifted {n}:{step}:{mode}')
    model.arrangement.load_state_dict(initial['model'], strict=True)
    b.require(b.tensor_digest(model.arrangement.state_dict()) == initial_digest, 'initial weights not restored')
    b.require(not torch.cuda.is_initialized(), 'CPU preflight initialized CUDA')
    b.require('teacher_xy' not in inspect_forward(), 'inference forward accepts a prefix')
    ROOT.mkdir(exist_ok=True)
    b.write(ROOT / 'preflight.json', dict(passed=True, code=code(), environment=env, optimizer_steps=0,
        anchor_checks=len(all_anchors), selector_isolated=True, decoder_gradient=True,
        inference_closed=True, control_step_gap_median_mm=measured,
        diagnosis_reproduced=True, monitor_absolute_xy=float(monitor.detach()),
        test_used=False))
    print('CPU PREFLIGHT PASS', flush=True)


def inspect_forward():
    import inspect
    return inspect.getsource(SequentialPose.forward)


def adopt(final, steps, baseline_steps, plan):
    rule = plan['adopt']
    checks = {}
    for mode, task_key, gap_key in (('oracle', 'oracle_task_strictly_greater_than', 'oracle_gap_p95_mm_strictly_less_than'),
                                    ('rgb', 'rgb_task_strictly_greater_than', 'rgb_gap_p95_mm_strictly_less_than')):
        checks[mode + '_task'] = final[mode]['task'] > rule[task_key]
        checks[mode + '_gap'] = final[mode]['gap_mm']['p95'] < rule[gap_key]
    for n in ('3', '4'):
        checks[n + '_free_gap'] = mean_median(steps, n, 'free') < mean_median(baseline_steps, n, 'free')
    return dict(passed=all(checks.values()), checks=checks)


def train():
    pre = json.loads((ROOT / 'preflight.json').read_text())
    b.require(pre['passed'] and pre['code'] == code() and pre['diagnosis_reproduced'], 'preflight stale')
    plan = json.loads(PLAN.read_text())
    p, c, ds, schedule = b.contracts()
    b.require(pre['environment'] == b.environment_receipt(), 'cache changed')
    b.require(not b.subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy')
    with (ROOT / 'execution.json').open('x') as stream:
        json.dump(dict(plan=plan, code=code(), started=time.time(), retry=False), stream, indent=2)
    b.register_terminal_run(ROOT, 2250)
    b.setup()
    start = time.monotonic()

    def deadline(*_):
        raise TimeoutError('1200-second experiment limit')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1200)
    _, val = b.datasets(c)
    model = b.build(c, 'cuda')
    initial = b.load_artifact(p['initial_checkpoint'])
    model.arrangement.load_state_dict(initial['model'], strict=True)
    opt = b.optimizer(model, c)
    opt.load_state_dict(initial['optimizer'])
    exposure = b.Exposure(791, 1203)
    state = initial['exposure']
    exposure.order = state['order']
    exposure.cursor = state['cursor']
    exposure.generator.set_state(state['generator'])
    visits = [0] * 791
    b.restore_rng(initial['rng'])
    frozen = b.frozen_digest(model)
    store = b.CheckpointStore(ROOT / 'shared')
    hist = [json.loads(line) for line in (CONTROL / 'history.jsonl').read_text().splitlines()]

    def assess():
        rng = b.rng_state()
        try:
            report = {k: b.evaluate(model, val, 'cuda', k) for k in ('oracle', 'rgb')}
            report['diagnostics'] = b.diagnose(model, val, 'cuda')
            return report
        finally:
            b.restore_rng(rng)

    def save(step):
        path = ROOT / f'checkpoint-{step:06d}.pt'
        store.save_checkpoint(path, dict(stage='oracle_layout', architecture=b.ARCHITECTURE, model=model.arrangement.state_dict(),
            optimizer=opt.state_dict(), rng=b.rng_state(), exposure=exposure.state(), parent_visits=list(visits),
            update=step, next_schedule_index=step, schedule_sha256=p['schedule_sha256'],
            elapsed_seconds=time.monotonic() - start, plan_sha256=b.file_hash(PLAN)))
        return str(path.resolve())

    initial_report = assess()
    b.require(initial_report == json.loads((CONTROL / 'evaluation-000000.json').read_text()), 'initial evaluation mismatch')
    save(0)
    updates = 0
    reports = []
    stop = 'update_budget'
    final = None
    last = None
    with tqdm(total=2250, desc='X-offset pose loss', unit='update') as bar:
        for item, row in zip(schedule, hist):
            if time.monotonic() - start >= 1080:
                stop = 'time_budget'
                break
            records = b.paired_records(item, exposure, visits, ds)
            opt.zero_grad(set_to_none=True)
            total, ce, pose, monitor, _ = objective(model, records, 'cuda')
            b.require(torch.isfinite(total), 'nonfinite loss')
            b.require(float(ce.detach()) == row['selector_ce'], 'selector loss diverged from the control')
            total.backward()
            norms = {group['name']: float(torch.nn.utils.clip_grad_norm_(group['params'], c['clip_norm'], error_if_nonfinite=True))
                     for group in opt.param_groups}
            opt.step()
            updates = item['update']
            b.require(all(torch.isfinite(param).all() for group in opt.param_groups for param in group['params']), 'nonfinite weights')
            with (ROOT / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=updates, loss=float(total.detach()), selector_ce=float(ce.detach()),
                    offset_huber=float(pose.detach()), monitor_absolute_xy=float(monitor.detach()), gradient_norms=norms,
                    seconds=time.monotonic() - start, sample_ids=[r['sample_id'] for r in records])) + '\n')
            bar.update()
            bar.set_postfix(offset=f'{float(pose.detach()):.3f}', xy=f'{float(monitor.detach()):.3f}')
            if updates % 250 == 0:
                final = assess()
                last = save(updates)
                b.write(ROOT / f'evaluation-{updates:06d}.json', final)
                b.require(b.frozen_digest(model) == frozen and all(
                    final[k]['frozen_outputs'] == initial_report[k]['frozen_outputs'] for k in ('oracle', 'rgb')), 'frozen regression')
                item_report = dict(update=updates, seconds=time.monotonic() - start,
                    new={k: b.summary(final[k]) for k in ('oracle', 'rgb')},
                    by_count={k: {str(n): b.by_count(final[k], n) for n in (2, 3, 4)} for k in ('oracle', 'rgb')})
                reports.append(item_report)
                b.write(ROOT / 'progress.json', dict(updates=updates, total=2250, evaluations=reports))
                print('EVALUATION', json.dumps(item_report), flush=True)
            b.require(shutil.disk_usage('.').free >= 12 * 2 ** 30, 'disk reserve')
    if final is None or reports[-1]['update'] != updates:
        final = assess()
        last = save(updates)
        b.write(ROOT / f'evaluation-{updates:06d}.json', final)
    model.arrangement.load_state_dict(b.load_artifact(last)['model'], strict=True)
    b.require(assess() == final, 'reload mismatch')
    signal.alarm(0)
    steps = step_gaps(model, val.records, 'cuda')
    b.write(ROOT / 'step-gaps.json', steps)
    verdict = adopt(final, steps, pre['control_step_gap_median_mm'], plan) if updates == 2250 else dict(
        passed=False, checks=dict(matched_endpoint=False))
    control_eval = json.loads((CONTROL / 'evaluation-002250.json').read_text())
    regressions = {}
    for mode in ('oracle', 'rgb'):
        before = {r['sample_id'] for r in control_eval[mode]['rows'] if r['task_success']}
        after = {r['sample_id'] for r in final[mode]['rows'] if r['task_success']}
        regressions[mode] = dict(retained=sorted(before & after), lost=sorted(before - after), gained=sorted(after - before))
    adopted = bool(updates == 2250 and verdict['passed'])
    b.write(ROOT / 'result.json', dict(status='PASS' if adopted else 'FAIL', adopted=adopted, stop_reason=stop,
        updates=updates, seconds=time.monotonic() - start, checkpoint=last, gate=verdict,
        selected={k: b.summary(final[k]) for k in ('oracle', 'rgb')},
        by_count={k: {str(n): b.by_count(final[k], n) for n in (2, 3, 4)} for k in ('oracle', 'rgb')},
        step_gap_median_mm=steps, control_step_gap_median_mm=pre['control_step_gap_median_mm'],
        regressions=regressions, evaluations=reports, reload_exact=True, frozen_exact=b.frozen_digest(model) == frozen,
        test_used=False, retry=False, main_updated=False))
    print('COMPLETE', stop, updates, 'ADOPTED' if adopted else 'NOT-ADOPTED', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'train'])
    action = parser.parse_args().action
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if action == 'preflight':
                preflight()
            else:
                with open('/tmp/knolling-planned-gpu.lock', 'a') as gpu:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    train()
        except BaseException as ex:
            b.write(ROOT / f'failure-{action}.json', dict(status='FAIL', error=str(ex), traceback=traceback.format_exc(), retry=False))
            raise


if __name__ == '__main__':
    main()
