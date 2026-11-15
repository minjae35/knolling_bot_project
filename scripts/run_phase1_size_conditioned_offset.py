"""One authorized run: zero-init size inputs on the existing offset head.

Dataset, loss, optimizer, seed and paired exposure match the size-diversity
control. The new columns start at zero. No retry and no Test.
"""
import json
import math
import signal
import time
import traceback
from pathlib import Path

import run_phase1_size_control as b
import torch
from tqdm import tqdm
from knolling_dataset.builder import file_hash
from knolling_model.data import supervision
from knolling_model.experiments.neural_sequential import teacher_order
from knolling_model.experiments.size_conditioned_offset import EXTRA, widen_decoder
from knolling_model.arrangement.contracts import Objects
from knolling_model.arrangement.driver import restore_rng
from knolling_model.arrangement.storage import CheckpointStore, load_artifact
from verify_phase1_single_row import arguments

ROOT = Path('artifacts/phase1_size_conditioned_offset')
SPEC_PATH = Path('configs/phase1_size_conditioned_offset.json')
CONTROL = Path('artifacts/phase1_size_diversity_B_environment_fixed_once')


def spec():
    value = json.loads(SPEC_PATH.read_text())
    b.require(value['training']['max_updates'] == 2250, 'update budget changed')
    b.require(value['training']['max_driver_seconds'] == 1200, 'time budget changed')
    b.require(value['training']['seed'] == 1203 and value['test'] == 'never', 'seed or test changed')
    b.require(value['adopt']['require_complete_2250_updates'] is True, 'incomplete endpoint cannot pass')
    return value


def control_history():
    rows = [json.loads(line) for line in (CONTROL / 'history.jsonl').read_text().splitlines() if line]
    b.require(len(rows) == 2250 and [r['update'] for r in rows] == list(range(1, 2251)), 'control history')
    return rows


def counts(report):
    return {str(n): b.by_count(report, n) for n in (2, 3, 4)}


def successes(report):
    return {r['sample_id'] for r in report['rows'] if r['task_success']}


def envelope(a, b, theta, half_delta):
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    return a + 2 * half_delta * c, b + 2 * half_delta * s


def replaced(obj, size):
    return Objects(obj.xy, size, obj.angle, obj.color, obj.shape, obj.valid, obj.query_index).validate()


def model_inputs(model, record, device):
    obj, text, mask, target, _ = arguments(model, [record], device)
    out = model.layout(obj, text, mask)
    ids = sorted(o['object_id'] for o in record['source']['objects'])
    order = teacher_order([record], [ids], obj.valid.shape[1], device)
    _, labels = supervision(record)
    return obj, out, order, target, [float(v) for v in labels['target_angle']]


@torch.no_grad()
def measure_size_response(model, records, device):
    decoder = model.layout.pose.decoder
    groups = {n: {'current': [], 'previous': []} for n in (3, 4)}
    for record in records:
        n = len(record['source']['objects'])
        if n not in groups:
            continue
        obj, out, order, target, theta = model_inputs(model, record, device)
        goal, selected, move = out['goal']['vector'], out['goal']['selected'], out['move_mask']
        teacher = [int(v) for v in order[0].tolist() if v >= 0]

        def offsets(size):
            xy = decoder(replaced(obj, size), goal, selected, move, order, teacher_xy=target)
            return [float(xy[0, teacher[step], 0] - target[0, teacher[step - 1], 0]) for step in range(1, n)]

        base = offsets(obj.size)
        for step in range(1, n):
            for role, index in (('current', teacher[step]), ('previous', teacher[step - 1])):
                changed = obj.size.detach().clone()
                a, b_ = envelope(float(changed[0, index, 0]), float(changed[0, index, 1]), theta[index], 0.001)
                changed[0, index, 0], changed[0, index, 1] = a, b_
                groups[n][role].append((offsets(changed)[step - 1] - base[step - 1]) * 1000)
    return {str(n): {role: _stats(values) for role, values in roles.items()} for n, roles in groups.items()}


def _stats(values):
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    return dict(n=int(arr.size), median=float(np.median(arr)), mean=float(arr.mean()),
                std=float(arr.std(ddof=1)), maximum_abs=float(np.max(np.abs(arr))))


def gate(report, plan):
    adopt = plan['adopt']
    checks = {}
    for mode in ('oracle', 'rgb'):
        report_mode = report[mode]
        checks[f'{mode}_task'] = report_mode['task'] > adopt[f'{mode}_task_strictly_greater_than']
        checks[f'{mode}_gap'] = report_mode['gap_mm']['p95'] < adopt[f'{mode}_gap_p95_mm_strictly_less_than']
        found = counts(report_mode)
        checks[f'{mode}_3'] = found['3']['success'] > adopt[f'{mode}_three_object_strictly_greater_than']
        checks[f'{mode}_4'] = found['4']['success'] > adopt[f'{mode}_four_object_strictly_greater_than']
        kept = successes(report_mode)
        checks[f'{mode}_retained'] = set(plan['control_successes'][mode]) <= kept
    checks['complete'] = report['updates'] == 2250
    return dict(passed=all(checks.values()), checks=checks)


def preflight(plan):
    ROOT.mkdir(exist_ok=True)
    b.require(not (ROOT / 'execution.json').exists(), 'run already claimed')
    p, c, ds, schedule = b.contracts()
    b.require(file_hash(plan['initial_checkpoint']) == plan['initial_sha256'], 'initial checkpoint')
    b.require(file_hash(plan['control_checkpoint']) == file_hash(CONTROL / 'checkpoint-002250.pt'), 'control checkpoint')
    b.require(file_hash(p['initial_checkpoint']) == plan['initial_sha256'], 'control initial mismatch')
    history = control_history()
    exposure = b.Exposure(791, 1203)
    visits = [0] * 791
    for step, item in enumerate(schedule, 1):
        records = b.paired_records(item, exposure, visits, ds)
        b.require(history[step - 1]['parent_ids'] == item['parent_ids'], 'history parent mismatch')
        b.require(history[step - 1]['sample_ids'] == [r['sample_id'] for r in records], 'history sample mismatch')
    b.require(set(visits) == {45} and sum(visits) == 35595, 'exposure totals')
    control_eval = json.loads((CONTROL / 'evaluation-002250.json').read_text())
    for mode in ('oracle', 'rgb'):
        actual = sorted(r['sample_id'] for r in control_eval[mode]['rows'] if r['task_success'])
        b.require(actual == sorted(plan['control_successes'][mode]), 'frozen success list mismatch')
    b.write(ROOT / 'preflight.json', dict(passed=True, updates=2250, presentations=sum(visits),
                                           control_history=len(history), cuda_initialized=torch.cuda.is_initialized(),
                                           code=b.code_hashes(), runner=file_hash(__file__),
                                           spec=file_hash(SPEC_PATH)))
    print('CPU PREFLIGHT PASS', flush=True)
    return p, c, ds, schedule, history


def run():
    plan = spec()
    p, c, ds, schedule, history = preflight(plan)
    b.require(not torch.cuda.is_initialized(), 'CUDA started during CPU preflight')
    with (ROOT / 'execution.json').open('x') as stream:
        json.dump(dict(authorized='User approved the pre-registered size-conditioned offset run',
                       retry=False, started=time.time()), stream)
    b.setup()
    start = time.monotonic()

    def deadline(*_):
        raise TimeoutError('driver 1200-second limit')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1200)
    _, val = b.datasets(c)
    model = b.build(c, 'cuda')
    initial = load_artifact(p['initial_checkpoint'])
    model.arrangement.load_state_dict(initial['model'], strict=True)
    probe = [next(r for r in val.records if len(r['source']['objects']) == n) for n in (2, 3, 4)]
    obj, text, mask, _, _ = arguments(model, probe, 'cuda')
    with torch.no_grad():
        before = model.layout(obj, text, mask)['xy'].clone()
    widen_decoder(model.layout.pose.decoder)
    b.require(torch.count_nonzero(model.layout.pose.decoder.output[0].weight[:, -EXTRA:]) == 0, 'new columns are not zero')
    with torch.no_grad():
        after = model.layout(obj, text, mask)['xy']
    b.require(torch.equal(before, after), 'widened forward changed the initial decoder')
    opt = b.optimizer(model, c)
    saved_opt = initial['optimizer']
    b.require(not saved_opt['state'], 'initial optimizer is not empty')
    try:
        opt.load_state_dict(saved_opt)
    except (ValueError, RuntimeError):
        b.require(len(saved_opt['param_groups']) == len(opt.param_groups), 'optimizer groups changed')
        for old, new in zip(saved_opt['param_groups'], opt.param_groups):
            b.require(old['lr'] == new['lr'] and old['weight_decay'] == new['weight_decay'], 'optimizer hyperparameters changed')
            b.require(len(old['params']) == len(new['params']), 'optimizer parameter count changed')
    exposure = b.Exposure(791, 1203)
    state = initial['exposure']
    exposure.order, exposure.cursor = state['order'], state['cursor']
    exposure.generator.set_state(state['generator'])
    visits = [0] * 791
    restore_rng(initial['rng'])
    frozen = b.frozen_digest(model)
    store = CheckpointStore(ROOT / 'shared')
    control_initial = json.loads((CONTROL / 'evaluation-000000.json').read_text())

    def assess():
        rng = b.rng_state()
        try:
            report = {mode: b.evaluate(model, val, 'cuda', mode) for mode in ('oracle', 'rgb')}
            report['diagnostics'] = b.diagnose(model, val, 'cuda')
            return report
        finally:
            restore_rng(rng)

    def save(step):
        path = ROOT / f'checkpoint-{step:06d}.pt'
        store.save_checkpoint(path, dict(stage='oracle_layout', architecture=b.ARCHITECTURE,
                                          model=model.arrangement.state_dict(), optimizer=opt.state_dict(),
                                          rng=b.rng_state(), exposure=exposure.state(), parent_visits=list(visits),
                                          update=step, elapsed_seconds=time.monotonic() - start, config=c,
                                          spec_sha256=file_hash(SPEC_PATH)))
        return str(path.resolve())

    initial_report = assess()
    b.require(initial_report == control_initial, 'initial evaluation differs from the control')
    b.write(ROOT / 'evaluation-000000.json', initial_report)
    save(0)
    updates = 0
    stop = 'update_budget'
    final = None
    last_path = None
    with tqdm(total=2250, desc='size-conditioned offset', unit='update') as bar:
        for item in schedule:
            if time.monotonic() - start >= 1080:
                stop = 'time_budget'
                break
            records = b.paired_records(item, exposure, visits, ds)
            opt.zero_grad(set_to_none=True)
            loss, ce, xy, _ = b.objective(model, records, 'cuda')
            b.require(float(ce.detach()) == history[item['update'] - 1]['selector_ce'], 'selector loss diverged from the control')
            b.require(torch.isfinite(loss), 'nonfinite loss')
            loss.backward()
            norms = {g['name']: float(torch.nn.utils.clip_grad_norm_(g['params'], c['clip_norm'], error_if_nonfinite=True))
                     for g in opt.param_groups}
            opt.step()
            updates = item['update']
            b.require(all(torch.isfinite(t).all() for g in opt.param_groups for t in g['params']), 'nonfinite weights')
            row = dict(update=updates, loss=float(loss.detach()), selector_ce=float(ce.detach()),
                       pose_huber=float(xy.detach()), gradient_norms=norms, seconds=time.monotonic() - start,
                       parent_ids=item['parent_ids'], sample_ids=[r['sample_id'] for r in records])
            with (ROOT / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            bar.update()
            if updates % 250 == 0:
                b.require(b.frozen_digest(model) == frozen, 'frozen weights changed')
                final = assess()
                b.write(ROOT / f'evaluation-{updates:06d}.json', final)
                last_path = save(updates)
                print('EVALUATION', updates, {mode: final[mode]['task'] for mode in ('oracle', 'rgb')}, flush=True)
    if final is None or updates % 250 != 0:
        final = assess()
        b.write(ROOT / f'evaluation-{updates:06d}.json', final)
        last_path = save(updates)
    model.arrangement.load_state_dict(load_artifact(last_path)['model'], strict=True)
    reloaded = assess()
    b.require(reloaded == final, 'checkpoint reload mismatch')
    signal.alarm(0)
    response = measure_size_response(model, val.records, 'cuda')
    weight = model.layout.pose.decoder.output[0].weight[:, -EXTRA:]
    report = dict(final, updates=updates)
    verdict = gate(report, plan)
    regressions = {}
    control_final = json.loads((CONTROL / 'evaluation-002250.json').read_text())
    for mode in ('oracle', 'rgb'):
        before, after = successes(control_final[mode]), successes(final[mode])
        regressions[mode] = dict(retained=sorted(before & after), lost=sorted(before - after), gained=sorted(after - before))
    adopted = bool(verdict['passed'] and updates == 2250 and stop == 'update_budget')
    b.require(b.frozen_digest(model) == frozen, 'frozen regression')
    b.require(file_hash(plan['control_checkpoint']) == file_hash(CONTROL / 'checkpoint-002250.pt'), 'control checkpoint changed')
    b.write(ROOT / 'result.json', dict(status='PASS' if adopted else 'FAIL', adopted=adopted, stop_reason=stop,
                                        updates=updates, seconds=time.monotonic() - start, gate=verdict,
                                        selected={mode: dict(b.summary(final[mode]), by_count=counts(final[mode]))
                                                  for mode in ('oracle', 'rgb')},
                                        control={mode: dict(b.summary(control_final[mode]), by_count=counts(control_final[mode]))
                                                 for mode in ('oracle', 'rgb')},
                                        regressions=regressions, size_response_mm_per_mm=response,
                                        new_column_l2=float(weight.norm()), new_column_max_abs=float(weight.abs().max()),
                                        checkpoint=last_path, reload_exact=True, frozen_exact=True,
                                        test_used=False, retry=False, main_updated=False))
    b.write(ROOT / 'size-response.json', response)
    print('COMPLETE', stop, updates, 'ADOPTED' if adopted else 'NOT-ADOPTED', flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    try:
        run()
    except BaseException as ex:
        b.write(ROOT / 'failure.json', dict(status='FAIL', adopted=False, error=str(ex),
                                             traceback=traceback.format_exc(), retry=False))
        raise


if __name__ == '__main__':
    main()
