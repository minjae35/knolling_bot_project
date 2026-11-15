"""Fine-tune only the row-origin Y readout if a GT-Y swap fixes the vertical gap.

Δx, the relation head, row/order, and the origin X weights stay fixed. The
diagnosis uses GT row, order, and Y. Inference does not.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement
from knolling_model.arrangement.contracts import Objects
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.arrangement.evaluation import _oracle_view
from knolling_model.arrangement.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate, stats
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of

ROOT = Path('artifacts/row_layout_origin_y')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DX = Path('artifacts/row_layout_dx_fixed_origin/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 100
MAX_UPDATES = 400
MIN_UPDATES = 200
PATIENCE = 2
CONFIG = dict(supervision='origin_y_only', lr=LR, per_class=PER_CLASS, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE, seed=1203,
              train_scenes=1273, val_scenes=152, init='dx_fixed_origin_y_readout', coordinate_clip=1.)


class Cycle:
    def __init__(self, indices, generator):
        self.indices = list(indices)
        self.generator = generator
        self.order = []
        self.cursor = 0

    def next(self, count):
        picked = []
        while len(picked) < count:
            if self.cursor >= len(self.order):
                self.order = torch.randperm(len(self.indices), generator=self.generator).tolist()
                self.cursor = 0
            take = min(count-len(picked), len(self.order)-self.cursor)
            picked.extend(self.indices[self.order[index]] for index in range(self.cursor, self.cursor+take))
            self.cursor += take
        return picked


def batch_indices(cycles):
    chosen = []
    for cycle in cycles:
        chosen.extend(cycle.next(PER_CLASS))
    return chosen


def prepare(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.row_layout[-1].weight.requires_grad_(True)
    model.row_layout[-1].bias.requires_grad_(True)


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                origin_y_mm=point['origin_y_mm'], horizontal_gap_mm=point['horizontal_gap_mm'],
                vertical_gap_mm=point['vertical_gap_mm'], vertical_within_2mm=point['vertical_within_2mm'],
                boundary=point['boundary'], collision=point['collision'], failures=point['failures'])


def teacher_with_optional_gt_y(model, head, stored, records):
    """Score the teacher pose, then the same pose with GT row-origin Y."""
    model.eval()
    summary = {key: dict(task=0, vertical=0, vertical_ids=[]) for key in ('predicted', 'gt_y')}
    vertical_values = {key: [] for key in summary}
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk = [records[index] for index in range(start, min(start+16, len(records)))]
            packed = collate_rows([stored[index] for index in range(start, start+len(chunk))], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                           packed['row'], packed['slot'], False)
            replaced = output['xy'].clone()
            replaced[..., 1] = packed['target_xy'][..., 1]
            poses = dict(predicted=output['xy'], gt_y=replaced)
            for local, record in enumerate(chunk):
                kept = stored[start+local]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                ids, truth = recovery_supervision(record)
                meta = dict(sample_id=record['sample_id'], object_ids=ids, record=record)
                for name, xy in poses.items():
                    layout = dict(xy=xy[local:local+1, :kept].cpu(),
                                  angle=output['angle'][local:local+1, :kept].cpu(),
                                  valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                    judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [meta]),
                                            semantic_check=semantic_errors_v3)[0]
                    detail = arrangement_errors(record, judged['predicted_targets']) if judged.get('predicted_targets') else None
                    summary[name]['task'] += int(bool(judged['task_success']))
                    if 'vertical_gap' in judged['errors']:
                        summary[name]['vertical'] += 1
                        summary[name]['vertical_ids'].append(record['sample_id'])
                    if detail is not None:
                        vertical_values[name].extend(detail['vertical_mm'])
    for name, values in vertical_values.items():
        summary[name]['vertical_gap_mm'] = stats(values)
        summary[name]['vertical_within_2mm'] = None if not values else sum(value <= 2. for value in values)/len(values)
    baseline_ids = set(summary['predicted']['vertical_ids'])
    summary['fixed_vertical_failures'] = len(baseline_ids-set(summary['gt_y']['vertical_ids']))
    summary['vertical_failures_before'] = len(baseline_ids)
    summary['clear'] = (summary['vertical_failures_before'] > 0
                        and summary['fixed_vertical_failures'] >= 0.7*summary['vertical_failures_before']
                        and summary['gt_y']['task'] > summary['predicted']['task'])
    return summary


def y_loss(output, target_xy, valid, slot):
    chosen = valid & (slot == 0)
    if not bool(chosen.any()):
        raise ValueError('origin Y loss needs a row origin')
    return F.smooth_l1_loss(output['xy'][..., 1][chosen], target_xy[..., 1][chosen], beta=.010)


def zero_x_grad(layer):
    layer.weight.grad[0].zero_()
    layer.bias.grad[0].zero_()


def passed(teacher, free, oracle, single, baseline):
    return (teacher['task'] > baseline['teacher']['task'] and free['task'] >= baseline['free']['task']
            and teacher['horizontal_gap_mm']['median'] <= baseline['teacher']['horizontal_gap_mm']['median']+0.02
            and free['horizontal_gap_mm']['median'] <= baseline['free']['horizontal_gap_mm']['median']+0.02
            and teacher['vertical_within_2mm'] > baseline['teacher']['vertical_within_2mm']
            and teacher['origin_y_mm']['median'] < baseline['teacher']['origin_y_mm']['median']
            and oracle >= 60 and single >= 62)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('origin Y result already exists')
    if not DX.is_file():
        raise RuntimeError('improved Δx checkpoint is missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    dx_held = stamp(DX)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    records = [record for record in train_data.records if multi_row(record)]
    counts = Counter(len(record['source']['objects']) for record in records)
    multi_val = [record for record in val_data.records if multi_row(record)]
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    phase_val = subset(val_data, 'validation', 94)
    if sum(counts.values()) != 1273 or len(multi_val) != 152 or len(single_val) != 104:
        raise RuntimeError(f'unexpected pools train={dict(counts)} val={len(multi_val)} single={len(single_val)}')
    groups = {}
    for count in (2, 4, 6, 8):
        groups[count] = [index for index, record in enumerate(records) if len(record['source']['objects']) == count]
        groups[count].sort(key=lambda index: records[index]['sample_id'])
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    modular = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, modular['model'])
    head.requires_grad_(False)
    head.eval()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(body, phase_val)
    stored = attach_rows(records, cache_records(body, records))
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    single_stored = cache_records(body, single_val)
    del body
    model = NeuralRowPlacement().to(stable.DEVICE)
    saved = torch.load(DX, map_location=stable.DEVICE, weights_only=False)
    model.load_state_dict(saved['model'])
    model.eval()
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    free = evaluate(model, head, val_stored, multi_val, 'free')
    print('RELOAD', json.dumps(dict(oracle=oracle, single=single, teacher=teacher['task'], free=free['task'],
                                   origin_y=teacher['origin_y_mm']['median'],
                                   horizontal=teacher['horizontal_gap_mm']['median'],
                                   vertical_failures=teacher['failures'].get('vertical_gap', 0),
                                   vertical_within=teacher['vertical_within_2mm'])), flush=True)
    reproduced = (teacher['task'] == 86 and free['task'] == 54 and oracle == 60 and single == 62
                  and abs(teacher['origin_mm']['median']-2.5263307094573975) <= 0.01
                  and abs(teacher['horizontal_gap_mm']['median']-0.6673277020454407) <= 0.01)
    diagnosis = teacher_with_optional_gt_y(model, head, val_stored, multi_val) if reproduced else None
    if diagnosis is not None:
        print('DIAGNOSIS', json.dumps(dict(
            before=diagnosis['predicted']['vertical'], after=diagnosis['gt_y']['vertical'],
            fixed=diagnosis['fixed_vertical_failures'], task_before=diagnosis['predicted']['task'],
            task_after=diagnosis['gt_y']['task'], within_before=diagnosis['predicted']['vertical_within_2mm'],
            within_after=diagnosis['gt_y']['vertical_within_2mm'], clear=diagnosis['clear'])), flush=True)
    if not reproduced or not diagnosis['clear']:
        stable.write(ROOT/'result.json', dict(
            status='FAIL', stage='diagnosis', trained=False, reproduced=reproduced,
            reload=dict(teacher=teacher['task'], free=free['task'], oracle=oracle, single=single),
            diagnosis=None if diagnosis is None else {key: diagnosis[key] for key in
                                                      ('fixed_vertical_failures', 'vertical_failures_before', 'clear')},
            next_bottleneck=None if diagnosis is None or diagnosis['clear'] else 'not_origin_y',
            modular_checkpoint_unchanged=stamp(MODULAR) == held,
            dx_checkpoint_unchanged=stamp(DX) == dx_held))
        print('STOP diagnosis is not clear enough to train', flush=True)
        return
    prepare(model)
    x_weight = model.row_layout[-1].weight.detach()[0].clone()
    x_bias = model.row_layout[-1].bias.detach()[0].clone()
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter is not model.row_layout[-1].weight and parameter is not model.row_layout[-1].bias}
    optimizer = torch.optim.AdamW((model.row_layout[-1].weight, model.row_layout[-1].bias),
                                  lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    prepare(copy)
    copy_optimizer = torch.optim.AdamW((copy.row_layout[-1].weight, copy.row_layout[-1].bias),
                                       lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    hidden = hidden_of(head, packed['features'])
    with torch.no_grad():
        original = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                         packed['row'], packed['slot'], False)
        restored = copy(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                        packed['row'], packed['slot'], False)
    reload_ok = torch.equal(original['xy'], restored['xy']) and loaded['update'] == 0 and loaded['config'] == CONFIG
    print('CHECKPOINT', json.dumps(dict(ok=reload_ok)), flush=True)
    del copy, copy_optimizer, original, restored, hidden
    if not reload_ok:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload', diagnosis_clear=True,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              dx_checkpoint_unchanged=stamp(DX) == dx_held))
        print('STOP reload mismatch', flush=True)
        return
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()
    baseline = dict(update=0, oracle=oracle, single=single, teacher=compact(teacher), free=compact(free))

    def record_eval(update, train_loss):
        current_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
        current_free = evaluate(model, head, val_stored, multi_val, 'free')
        current_oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
        current_single = score(head, single_stored, single_val)['task']
        point = dict(update=update, loss=train_loss, oracle=current_oracle, single=current_single,
                     teacher=compact(current_teacher), free=compact(current_free),
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(
            update=update, oracle=current_oracle, single=current_single,
            teacher=current_teacher['task'], free=current_free['task'],
            origin_y=current_teacher['origin_y_mm'], vertical=current_teacher['vertical_within_2mm'],
            horizontal=current_teacher['horizontal_gap_mm']['median'],
            vertical_failures=current_teacher['failures'].get('vertical_gap', 0), loss=train_loss)), flush=True)
        return point

    history.write(json.dumps(baseline)+'\n')
    history.flush()
    best = baseline
    save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, 0, CONFIG)
    reason = 'max_updates'
    stale = 0
    running = 0.
    seen = 0
    for update in range(1, MAX_UPDATES+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
        hidden = hidden_of(head, packed['features'])
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                       packed['row'], packed['slot'], False)
        loss = y_loss(output, packed['target_xy'], packed['valid'], packed['slot'])
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        zero_x_grad(model.row_layout[-1])
        if update == 1:
            y_grad = float(model.row_layout[-1].weight.grad[1].norm())
            x_grad = float(model.row_layout[-1].weight.grad[0].norm())
            leaked = [name for name, parameter in model.named_parameters()
                      if parameter.grad is not None and parameter is not model.row_layout[-1].weight
                      and parameter is not model.row_layout[-1].bias]
            print('PATH', json.dumps(dict(y_grad=y_grad, x_grad=x_grad, leaked=leaked, loss=float(loss.detach()))), flush=True)
            if y_grad == 0 or x_grad != 0 or leaked:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_((model.row_layout[-1].weight, model.row_layout[-1].bias), CONFIG['coordinate_clip'])
        optimizer.step()
        if not torch.equal(model.row_layout[-1].weight[0], x_weight) or not torch.equal(model.row_layout[-1].bias[0], x_bias):
            reason = 'x_moved'
            break
        if any(not torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
               if name in frozen):
            reason = 'frozen_moved'
            break
        running += float(loss.detach())
        seen += 1
        if update % EVAL_EVERY and update != MAX_UPDATES:
            continue
        save_checkpoint(ROOT/f'checkpoint-{update:04d}.pt', model, optimizer, update, CONFIG)
        point = record_eval(update, running/seen)
        running = 0.
        seen = 0
        if point['oracle'] < 60 or point['single'] < 62 or point['teacher']['task'] < 86 or point['free']['task'] < 54:
            reason = 'regression'
            break
        if point['teacher']['horizontal_gap_mm']['median'] > baseline['teacher']['horizontal_gap_mm']['median']+0.02:
            reason = 'horizontal_regression'
            break
        better = (point['teacher']['task'], point['free']['task']) > (best['teacher']['task'], best['free']['task'])
        if better:
            best = point
            stale = 0
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    if reason in ('path_blocked', 'x_moved', 'frozen_moved', 'diverged'):
        stable.write(ROOT/'result.json', dict(status='STOP', stage=reason, trained=False,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              dx_checkpoint_unchanged=stamp(DX) == dx_held))
        print('STOP', reason, flush=True)
        return
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    prepare(reloaded)
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    val_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    val_free = evaluate(model, head, val_stored, multi_val, 'free')
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    status = 'PASS' if passed(val_teacher, val_free, oracle_end, single_end, baseline) else 'FAIL'
    next_limit = None if status == 'PASS' else 'row_order'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, next_bottleneck=next_limit, best_update=best_state['update'],
        best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        reload_matches_logged_success=val_teacher['task'] == best['teacher']['task'] and val_free['task'] == best['free']['task'],
        config=CONFIG, diagnosis=dict(fixed=diagnosis['fixed_vertical_failures'],
                                     before=diagnosis['vertical_failures_before'],
                                     task_before=diagnosis['predicted']['task'],
                                     task_after=diagnosis['gt_y']['task'], clear=True),
        baseline_teacher=baseline['teacher'], baseline_free=baseline['free'],
        validation_teacher=compact(val_teacher), validation_free=compact(val_free),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters()),
        origin_x_unchanged=torch.equal(model.row_layout[-1].weight[0], x_weight) and torch.equal(model.row_layout[-1].bias[0], x_bias),
        modular_checkpoint_unchanged=stamp(MODULAR) == held, dx_checkpoint_unchanged=stamp(DX) == dx_held,
        full_9600=False, joint=False, test_used=False, geometry_solver=False, gt_origin_at_inference=False))
    print('DONE', json.dumps(dict(
        status=status, stop=reason, bottleneck=next_limit, update=best_state['update'],
        teacher_before=86, teacher=val_teacher['task'], free_before=54, free=val_free['task'],
        origin_y=val_teacher['origin_y_mm'], vertical=val_teacher['vertical_within_2mm'],
        horizontal=val_teacher['horizontal_gap_mm']['median'], oracle=oracle_end, single=single_end,
        checkpoint=str(ROOT/'checkpoint-best.pt'))), flush=True)


if __name__ == '__main__':
    main()
