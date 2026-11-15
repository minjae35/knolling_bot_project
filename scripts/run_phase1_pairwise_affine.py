"""One pairwise-affine run against the existing 2,250-update control.

Same initial sequential weights, paired size-variant exposure, selector CE, and
XY Huber. The only new parameters are the affine path and residual. No Test
split, no second run, and no geometry solver.
"""
import json
import os
from collections import Counter
from pathlib import Path
import signal
import time
import traceback

os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import numpy as np
import torch
from tqdm import tqdm
from knolling_dataset.builder import file_hash
from knolling_dataset.validator import inferred_rows, polygon
from knolling_model.data import Batch
from knolling_model.diagnostics import positioned_objects
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.pairwise_affine import (
    ARCHITECTURE, build, load_sequential_init, position_loss, teacher_order)
from knolling_model.experiments.phase1_evaluation import phase1_gate, semantic_errors_phase1
from knolling_model.experiments.size_diversity import OracleSizeVariants, require
from knolling_model.arrangement.contracts import Objects, oracle_objects
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.arrangement.driver import restore_rng, rng_state
from knolling_model.arrangement.evaluation import _oracle_view
from knolling_model.arrangement.model import legacy_evaluation_view
from knolling_model.arrangement.storage import CheckpointStore, load_artifact

SPEC = json.loads(Path(os.environ.get('KNOLLING_RUN_SPEC', 'configs/phase1_pairwise_affine.json')).read_text())
ROOT = Path(SPEC['output'])
BASE = json.loads(Path(SPEC['base_config']).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def setup():
    torch.set_num_threads(4)
    torch.manual_seed(SPEC['seed'])
    torch.cuda.manual_seed_all(SPEC['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Exposure:
    def __init__(self, n, seed):
        self.n = n
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor = [], 0

    def next(self, size):
        if self.cursor == len(self.order):
            self.order = torch.randperm(self.n, generator=self.generator).tolist()
            self.cursor = 0
        result = self.order[self.cursor:self.cursor+size]
        self.cursor += len(result)
        return result

    def state(self):
        return dict(order=self.order, cursor=self.cursor, generator=self.generator.get_state())


def datasets():
    result = {}
    for split, count in (('train', 791), ('validation', 94)):
        manifest = json.loads(Path(BASE[split+'_manifest']).read_text())
        require(manifest['original_split'] == split, split)
        require(file_hash(Path(BASE['dataset'])/'annotations.json') == BASE['dataset_sha256'], 'dataset')
        data = ExpandedDataset(BASE['dataset'], split)
        by = {record['sample_id']: record for record in data.records}
        data.records = [by[sample_id] for sample_id in manifest['sample_ids']]
        data.pairs = manifest['counterfactual_pairs']
        require(len(data) == count, f'{split} count')
        result[split] = data
    require(len(result['validation'].pairs) == 23, 'validation pairs')
    return result['train'], result['validation']


def stats(values):
    return dict(p95=float(np.percentile(values, 95)), maximum=float(max(values))) if values else dict(p95=None, maximum=None)


def arguments(model, records, device):
    truths = [{key: value.to(device) for key, value in recovery_supervision(record)[1].items()} for record in records]
    objects = oracle_objects(truths)
    tokens = model.tokenizer([record['instruction']['text'] for record in records],
                             padding=True, truncation=False, return_tensors='pt')
    mask = tokens['attention_mask'].to(device).bool()
    text = model.encode_text(tokens['input_ids'].to(device), mask)
    return objects, text, mask, truths


def oracle_batch(model, records, device):
    objects, text, mask, truths = arguments(model, records, device)
    ids = []
    targets = torch.zeros_like(objects.xy)
    for index, (record, truth) in enumerate(zip(records, truths)):
        objids, _ = recovery_supervision(record)
        ids.append(objids)
        targets[index, :len(objids)] = truth['target_xy']
    return objects, text, mask, targets, ids


@torch.no_grad()
def evaluate(model, data, device, mode):
    rows = []
    semantics = []
    for index, record in enumerate(data.records):
        sample = data[index]
        truth = {key: value.to(device) for key, value in recovery_supervision(record)[1].items()}
        if mode == 'oracle':
            objects, text, mask, _ = arguments(model, [record], device)
            layout = model.layout(objects, text, mask)
            view = _oracle_view(objects, layout)
        else:
            output = model.predict(sample['image'][None].to(device), [sample['text']], [record['sample_id']])
            layout = output['layout']
            view = legacy_evaluation_view(output)
        semantics.append(dict(sample_id=record['sample_id'], angle=layout['angle'].cpu().tolist(),
                              move=layout['move_mask'].cpu().tolist(), queries=layout['query_index'].cpu().tolist()))
        rows.append(evaluate_batch(view, Batch({}, [truth], [sample['metadata']]),
                                   semantic_check=semantic_errors_phase1)[0])
    gaps, per_scene = [], []
    for record, row in zip(data.records, rows):
        values = []
        if 'predicted_targets' in row:
            for line in inferred_rows(positioned_objects(record, row['predicted_targets'])):
                for left, right in zip(line, line[1:]):
                    gap = polygon(right, right.target_pose).bounds[0]-polygon(left, left.target_pose).bounds[2]
                    values.append(abs(gap-record['instruction']['gap_m'])*1000)
        gaps.extend(values)
        per_scene.append(dict(sample_id=record['sample_id'], objects=len(record['source']['objects']),
                              task=row['task_success'], gap_error_mm=values, errors=row['errors']))
    gate = phase1_gate(rows, data.records, data.pairs, [record['sample_id'] for record in data.records])
    require(gate['complete'], 'incomplete validation')
    return dict(mode=mode, task=gate['successful_scenes'], total=len(data), cf=gate['counterfactual_passed'],
                pairs=len(data.pairs), failures=dict(Counter(error for row in rows for error in row['errors'])),
                gap_mm=stats(gaps), xy_mm=stats([value for row in rows for value in row['target_position_mm']]),
                angle_deg=stats([value for row in rows for value in row['target_angle_deg']]),
                rows=rows, per_scene=per_scene, frozen_outputs=semantics)


def by_count(report, count):
    rows = [row for row in report['per_scene'] if row['objects'] == count]
    return dict(success=sum(bool(row['task']) for row in rows), total=len(rows))


def summary(report):
    return {key: report[key] for key in ('mode', 'task', 'total', 'cf', 'pairs', 'failures', 'gap_mm', 'xy_mm', 'angle_deg')}


def objective(model, records, device):
    objects, text, mask, targets, ids = oracle_batch(model, records, device)
    output = model.layout(objects, text, mask)
    order = teacher_order(records, ids, objects.valid.shape[1], device)
    ce = model.layout.pose.selector_loss(objects, output['goal']['vector'], output['goal']['selected'],
                                         output['move_mask'], order)
    xy = position_loss(output, targets)
    return ce+xy, ce, xy


def optimizer(model):
    pose = model.layout.pose
    return torch.optim.AdamW(
        [{'params': list(pose.selector.parameters()), 'name': 'selector'},
         {'params': list(pose.decoder.parameters()), 'name': 'pose'}],
        lr=BASE['lr'], weight_decay=BASE['weight_decay'], foreach=False)


def paired_records(item, exposure, visits, variants, parents):
    indices = exposure.next(BASE['batch_size'])
    require(item['parent_ids'] == [parents.parent_ids[index] for index in indices], 'parent minibatch mismatch')
    records = []
    for index, selected in zip(indices, item['selected']):
        require(selected['variant_index'] == visits[index] % 8, 'variant visit mismatch')
        record = variants.record_for_visit(index, visits[index])
        require(record['sample_id'] == selected['sample_id'], 'sample mismatch')
        records.append(record)
        visits[index] += 1
    return records


def control_history():
    rows = [json.loads(line) for line in Path(SPEC['control_history']).read_text().splitlines() if line]
    require(len(rows) == SPEC['max_updates'], 'control history length')
    return rows


def replaced(objects, index, delta):
    size = objects.size.clone()
    size[0, index, 0] += delta
    return Objects(objects.xy, size, objects.angle, objects.color, objects.shape, objects.valid, objects.query_index)


@torch.no_grad()
def major_axis_response(model, record, device):
    objects, text, mask, _ = arguments(model, [record], device)
    layout = model.layout(objects, text, mask)
    ids, _ = recovery_supervision(record)
    order = teacher_order([record], [ids], objects.valid.shape[1], device)
    goal, selected = layout['goal']['vector'], layout['goal']['selected']
    base = model.layout.pose.diagnose(objects, goal, selected, order)
    chosen = int(order[0, 1])
    changed = model.layout.pose.diagnose(replaced(objects, chosen, 0.001), goal, selected, order)
    return float((changed['xy'][0, chosen, 0]-base['xy'][0, chosen, 0])*1000)


def adopt(report, regressions):
    rule = SPEC['adopt']
    checks = dict(
        oracle_task=report['oracle']['task'] > rule['oracle_task_greater_than'],
        rgb_task=report['rgb']['task'] > rule['rgb_task_greater_than'],
        oracle_3=by_count(report['oracle'], 3)['success'] >= rule['oracle_3_at_least'],
        rgb_3=by_count(report['rgb'], 3)['success'] >= rule['rgb_3_at_least'],
        oracle_4=by_count(report['oracle'], 4)['success'] >= rule['oracle_4_at_least'],
        rgb_4=by_count(report['rgb'], 4)['success'] >= rule['rgb_4_at_least'])
    if 'min_retention' in rule:
        checks['oracle_retention'] = regressions['oracle']['retention'] >= rule['min_retention']
        checks['rgb_retention'] = regressions['rgb']['retention'] >= rule['min_retention']
    return dict(passed=all(checks.values()), checks=checks)


def run():
    ROOT.mkdir(exist_ok=True)
    require(not (ROOT/'execution.json').exists(), 'run already claimed')
    history = control_history()
    schedule = json.loads(Path(SPEC['schedule']).read_text())
    require(len(schedule) == SPEC['max_updates'], 'schedule length')
    variants = OracleSizeVariants(SPEC['variants'])
    _, validation = datasets()
    exposure = Exposure(791, SPEC['seed'])
    visits = [0]*791
    for item in schedule:
        paired_records(item, exposure, visits, variants, variants)
    require(set(visits) == {45} and sum(visits) == 35595, 'exposure totals')
    initial = load_artifact(SPEC['initial_checkpoint'])
    require(initial['update'] == 0 and not initial['optimizer']['state'], 'initial optimizer is not empty')
    setup()
    model = build(BASE, 'cpu')
    missing = load_sequential_init(model, SPEC['initial_checkpoint'])
    probes = {}
    for count in (3, 4):
        record = next(row for row in validation.records if len(row['source']['objects']) == count)
        probes[str(count)] = major_axis_response(model, record, 'cpu')
        require(probes[str(count)] > 0.15, f'CPU size response too small for {count} objects: {probes[str(count)]}')
    model.arrangement.encode_text.values.clear()
    write(ROOT/'cpu-size-response.json', dict(millimetres_per_millimetre=probes, new_parameters=missing))
    print('CPU SIZE RESPONSE', probes, flush=True)
    model.to('cuda')
    restore_rng(initial['rng'])
    exposure = Exposure(791, SPEC['seed'])
    state = initial['exposure']
    exposure.order, exposure.cursor = state['order'], state['cursor']
    exposure.generator.set_state(state['generator'])
    visits = [0]*791
    opt = optimizer(model)
    store = CheckpointStore(ROOT/'shared')
    start = time.monotonic()

    def assess():
        held = rng_state()
        try:
            return {mode: evaluate(model, validation, 'cuda', mode) for mode in ('oracle', 'rgb')}
        finally:
            restore_rng(held)

    def save(step):
        path = ROOT/f'checkpoint-{step:06d}.pt'
        store.save_checkpoint(path, dict(stage='oracle_layout', architecture=ARCHITECTURE,
            model=model.arrangement.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
            exposure=exposure.state(), update=step, elapsed_seconds=time.monotonic()-start))
        return str(path)

    report = assess()
    write(ROOT/'evaluation-000000.json', {mode: summary(report[mode]) | dict(
        by_count={str(n): by_count(report[mode], n) for n in (2, 3, 4)}) for mode in report})
    save(0)
    write(ROOT/'execution.json', dict(architecture=ARCHITECTURE, retry=False, test_used=False))
    updates, stop, final = 0, 'update_budget', report

    def deadline(*_):
        raise TimeoutError('1200-second budget expired')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SPEC['alarm_seconds'])
    with tqdm(total=SPEC['max_updates'], desc='pairwise affine', unit='update') as bar:
        for item in schedule:
            if time.monotonic()-start >= SPEC['loop_seconds']:
                stop = 'time_budget'
                break
            records = paired_records(item, exposure, visits, variants, variants)
            opt.zero_grad(set_to_none=True)
            loss, ce, xy = objective(model, records, 'cuda')
            require(float(ce.detach()) == history[item['update']-1]['selector_ce'],
                    f"selector CE diverged at update {item['update']}")
            require(torch.isfinite(loss), 'nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.layout.pose.selector.parameters(), BASE['clip_norm'], error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(model.layout.pose.decoder.parameters(), BASE['clip_norm'], error_if_nonfinite=True)
            opt.step()
            updates = item['update']
            bar.update()
            if updates % SPEC['evaluate_every'] == 0:
                final = assess()
                write(ROOT/f'evaluation-{updates:06d}.json', final)
                save(updates)
                print('EVALUATION', updates, {mode: final[mode]['task'] for mode in final},
                      {mode: {n: by_count(final[mode], n)['success'] for n in (3, 4)} for mode in final}, flush=True)
    if updates and updates % SPEC['evaluate_every'] != 0:
        final = assess()
        write(ROOT/f'evaluation-{updates:06d}.json', final)
        save(updates)
    signal.alarm(0)
    reloaded = load_artifact(ROOT/f'checkpoint-{updates:06d}.pt')
    model.arrangement.load_state_dict(reloaded['model'], strict=True)
    replay = assess()
    require(replay == final, 'reload mismatch')
    control = json.loads(Path(SPEC['control_evaluation']).read_text())
    regressions = {}
    for mode in ('oracle', 'rgb'):
        before = {row['sample_id'] for row in control[mode]['rows'] if row['task_success']}
        after = {row['sample_id'] for row in final[mode]['rows'] if row['task_success']}
        regressions[mode] = dict(retained=sorted(before & after), lost=sorted(before-after), gained=sorted(after-before),
                                 retention=len(before & after)/max(len(before), 1))
    verdict = adopt(final, regressions)
    offset = model.layout.pose.decoder.offset
    adopted = bool(verdict['passed'] and updates == SPEC['max_updates'] and stop == 'update_budget')
    write(ROOT/'result.json', dict(
        status='PASS' if adopted else 'FAIL', adopted=adopted, stop_reason=stop, updates=updates,
        seconds=time.monotonic()-start, gate=verdict,
        selected={mode: summary(final[mode]) | dict(by_count={str(n): by_count(final[mode], n) for n in (2, 3, 4)})
                  for mode in ('oracle', 'rgb')},
        control={mode: summary(control[mode]) | dict(by_count={str(n): by_count(control[mode], n) for n in (2, 3, 4)})
                 for mode in ('oracle', 'rgb')},
        regressions=regressions,
        coefficients=dict(bias=float(offset.bias), w_prev=offset.w_prev.detach().cpu().tolist(),
                          w_curr=offset.w_curr.detach().cpu().tolist(), w_gap=offset.w_gap.detach().cpu().tolist(),
                          w_pair=offset.w_pair.detach().cpu().tolist()),
        cpu_size_response_mm_per_mm=probes, test_used=False, retry=False))
    print('COMPLETE', stop, updates, 'ADOPTED' if adopted else 'NOT-ADOPTED', flush=True)


def main():
    try:
        run()
    except BaseException as exc:
        write(ROOT/'failure.json', dict(status='FAIL', error=str(exc), traceback=traceback.format_exc(), retry=False))
        raise


if __name__ == '__main__':
    main()
