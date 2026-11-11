"""Train only the within-row step, with the improved row origin held fixed.

The loss is the existing direct Δx term: the step output against the GT center
displacement. It is not an edge-gap loss and not absolute X. One run.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement, step_placement_loss
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of
from run_row_layout_step import rejected_offset_matches

ROOT = Path('artifacts/row_layout_dx_fixed_origin')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
ORIGIN = Path('artifacts/row_layout_origin/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 250
MAX_UPDATES = 1500
MIN_UPDATES = 500
PATIENCE = 2
ORACLE_FLOOR = 60
SINGLE_FLOOR = 62
ORIGIN_MEDIAN = 2.5263307094573975
CONFIG = dict(supervision='step_dx_only', lr=LR, per_class=PER_CLASS, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE, seed=1203,
              oracle_floor=ORACLE_FLOOR, single_floor=SINGLE_FLOOR, origin_median_mm=ORIGIN_MEDIAN,
              train_scenes=1273, val_scenes=152, init='origin_best_step_only', coordinate_clip=10.)


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
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('step.'))


def assert_dx_trains_only_the_step():
    if rejected_offset_matches():
        raise RuntimeError('step Δx loss matches the rejected offset loss')
    torch.manual_seed(0)
    model = NeuralRowPlacement()
    prepare(model)
    hidden = torch.randn(1, 3, 128)
    valid = torch.ones(1, 3, dtype=torch.bool)
    goal = torch.zeros(1, 13)
    goal[0, 6] = 1
    widths = torch.tensor([[.04, .05, .06]])
    heights = torch.tensor([[.02, .03, .02]])
    row = torch.zeros(1, 3, dtype=torch.long)
    slot = torch.arange(3)[None]
    target = torch.zeros(1, 3, 2)
    target[0, :, 0] = torch.tensor([.10, .16, .23])
    target[0, :, 1] = .20
    output = model(hidden, valid, goal, widths, heights, row, slot, False)
    losses = step_placement_loss(output, target, torch.zeros(1, 3), valid, row, slot)
    losses['dx'].backward()
    step_grad = sum(parameter.grad.detach().pow(2).sum().item()
                    for parameter in model.step.parameters() if parameter.grad is not None)**.5
    leaked = [name for name, parameter in model.named_parameters()
              if not name.startswith('step.') and parameter.grad is not None]
    if step_grad == 0 or leaked:
        raise RuntimeError(f'Δx supervision does not stay on the step: {step_grad} {leaked}')


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                position_mm=point['position_mm'], origin_mm=point['origin_mm'],
                origin_x_mm=point['origin_x_mm'], origin_y_mm=point['origin_y_mm'],
                horizontal_gap_mm=point['horizontal_gap_mm'], vertical_gap_mm=point['vertical_gap_mm'],
                boundary=point['boundary'], collision=point['collision'], failures=point['failures'])


def heavy(point):
    counts = point['by_count']
    return (counts['8']['success'], counts['6']['success'], counts['4']['success'], point['task'])


def same_pose(left, right):
    return all(torch.equal(left[key], right[key]) for key in ('xy', 'delta_x', 'angle', 'row_logits'))


def gap_of(point):
    return point['horizontal_gap_mm']['median']


def passed(teacher, free, oracle, single, baseline):
    origin = teacher['origin_mm']['median']
    gap_improved = gap_of(teacher) < gap_of(baseline['teacher']) and gap_of(free) < gap_of(baseline['free'])
    heavy_improved = all(free['by_count'][count]['success'] > baseline['free']['by_count'][count]['success']
                         for count in ('4', '6', '8'))
    preserved = (oracle >= ORACLE_FLOOR and single >= SINGLE_FLOOR and origin is not None
                 and origin <= baseline['teacher']['origin_mm']['median']+0.05)
    return gap_improved and heavy_improved and preserved


def bottleneck(teacher, free, baseline):
    teacher_gap = gap_of(teacher) < gap_of(baseline['teacher'])
    teacher_heavy = any(teacher['by_count'][count]['success'] > baseline['teacher']['by_count'][count]['success']
                        for count in ('4', '6', '8'))
    free_heavy = any(free['by_count'][count]['success'] > baseline['free']['by_count'][count]['success']
                     for count in ('4', '6', '8'))
    row_held = abs(free['row_accuracy']-baseline['free']['row_accuracy']) < 1e-6
    slot_held = abs(free['slot_accuracy']-baseline['free']['slot_accuracy']) < 1e-6
    if (teacher_gap or teacher_heavy) and not free_heavy and row_held and slot_held:
        return 'row_order'
    if not teacher_gap:
        return 'within_row_dx'
    return 'free_rollout'


def main():
    assert_dx_trains_only_the_step()
    if (ROOT/'result.json').exists():
        raise RuntimeError('fixed-origin Δx result already exists')
    if not ORIGIN.is_file():
        raise RuntimeError('improved row origin checkpoint is missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    origin_held = stamp(ORIGIN)
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
    saved = torch.load(ORIGIN, map_location=stable.DEVICE, weights_only=False)
    model.load_state_dict(saved['model'])
    prepare(model)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not name.startswith('step.')}
    optimizer = torch.optim.AdamW(model.step.parameters(), lr=LR, weight_decay=0., foreach=False)
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single, train=dict(counts), config=CONFIG)), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              origin_checkpoint_unchanged=stamp(ORIGIN) == origin_held))
        print('STOP initialization mismatch', flush=True)
        return
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    prepare(copy)
    copy_optimizer = torch.optim.AdamW(copy.step.parameters(), lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    hidden = hidden_of(head, packed['features'])
    with torch.no_grad():
        original = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                         packed['row'], packed['slot'], False)
        restored = copy(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                        packed['row'], packed['slot'], False)
    reload_ok = same_pose(original, restored) and loaded['update'] == 0 and loaded['config'] == CONFIG
    print('RELOAD', json.dumps(dict(ok=reload_ok, update=loaded['update'])), flush=True)
    del copy, copy_optimizer, original, restored, hidden
    if not reload_ok:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload',
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              origin_checkpoint_unchanged=stamp(ORIGIN) == origin_held))
        print('STOP reload mismatch', flush=True)
        return
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()

    def evaluate_split(store, rows, mode):
        return evaluate(model, head, store, rows, mode)

    def record_eval(update, train_loss):
        free = evaluate_split(val_stored, multi_val, 'free')
        teacher = evaluate_split(val_stored, multi_val, 'teacher')
        current_oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
        current_single = score(head, single_stored, single_val)['task']
        point = dict(update=update, loss=train_loss, oracle=current_oracle, single=current_single,
                     free=compact(free), teacher=compact(teacher), seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(
            update=update, oracle=current_oracle, single=current_single, free=free['task'],
            free_by={key: value['success'] for key, value in free['by_count'].items()},
            teacher=teacher['task'],
            teacher_by={key: value['success'] for key, value in teacher['by_count'].items()},
            origin=teacher['origin_mm']['median'], horizontal=teacher['horizontal_gap_mm'],
            free_horizontal=free['horizontal_gap_mm'], row=free['row_accuracy'], slot=free['slot_accuracy'],
            loss=train_loss)), flush=True)
        return point

    baseline = record_eval(0, None)
    origin_ok = abs(baseline['teacher']['origin_mm']['median']-ORIGIN_MEDIAN) <= 0.05
    if not origin_ok or baseline['free']['task'] != 16 or baseline['teacher']['task'] != 27:
        history.close()
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', baseline=baseline,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              origin_checkpoint_unchanged=stamp(ORIGIN) == origin_held))
        print('STOP baseline does not match the saved origin model', flush=True)
        return
    best = baseline
    best_gap = baseline
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
        losses = step_placement_loss(output, packed['target_xy'], packed['target_angle'], packed['valid'],
                                     packed['row'], packed['slot'])
        loss = losses['dx']
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        if update == 1:
            moved = [name for name, parameter in model.named_parameters()
                     if not name.startswith('step.') and parameter.grad is not None]
            step_grad = sum(parameter.grad.detach().pow(2).sum().item()
                            for parameter in model.step.parameters() if parameter.grad is not None)**.5
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            print('PATH', json.dumps(dict(step_grad=step_grad, frozen_grads=moved, head_grad=head_grad,
                                         dx=float(loss.detach()))), flush=True)
            if step_grad == 0 or moved or head_grad:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.step.parameters(), CONFIG['coordinate_clip'])
        optimizer.step()
        if update == 1:
            changed = [name for name, parameter in model.named_parameters()
                       if not name.startswith('step.') and not torch.equal(parameter, frozen[name])]
            if changed:
                reason = 'frozen_moved'
                print('STOP frozen weights changed', json.dumps(changed), flush=True)
                break
        running += float(loss.detach())
        seen += 1
        if update % EVAL_EVERY and update != MAX_UPDATES:
            continue
        save_checkpoint(ROOT/f'checkpoint-{update:04d}.pt', model, optimizer, update, CONFIG)
        point = record_eval(update, running/seen)
        running = 0.
        seen = 0
        if point['oracle'] < ORACLE_FLOOR or point['single'] < SINGLE_FLOOR:
            reason = 'regression'
            break
        if point['teacher']['origin_mm']['median'] > baseline['teacher']['origin_mm']['median']+0.05:
            reason = 'origin_regression'
            break
        free_better = heavy(point['free']) > heavy(best['free']) or (
            heavy(point['free']) == heavy(best['free']) and gap_of(point['free']) < gap_of(best['free']))
        gap_moved = gap_of(point['teacher']) <= gap_of(best_gap['teacher'])-0.05
        if free_better:
            best = point
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        if gap_of(point['teacher']) < gap_of(best_gap['teacher']):
            best_gap = point
        stale = 0 if free_better or gap_moved else stale+1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    if reason in ('path_blocked', 'frozen_moved', 'diverged'):
        stable.write(ROOT/'result.json', dict(status='STOP', stage=reason,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              origin_checkpoint_unchanged=stamp(ORIGIN) == origin_held, trained=False))
        print('STOP', reason, flush=True)
        return
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    prepare(reloaded)
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    train_free = evaluate_split(stored, records, 'free')
    train_teacher = evaluate_split(stored, records, 'teacher')
    val_free = evaluate_split(val_stored, multi_val, 'free')
    val_teacher = evaluate_split(val_stored, multi_val, 'teacher')
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    frozen_held = all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                      if not name.startswith('step.'))
    status = 'PASS' if passed(val_teacher, val_free, oracle_end, single_end, baseline) else 'FAIL'
    next_limit = None if status == 'PASS' else bottleneck(val_teacher, val_free, baseline)
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, next_bottleneck=next_limit, best_update=best_state['update'],
        best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        best_teacher_gap_update=best_gap['update'], best_teacher_gap=compact(best_gap['teacher']),
        best_teacher_gap_free=compact(best_gap['free']),
        reload_matches_logged_success=val_free['task'] == best['free']['task'],
        config=CONFIG, baseline_free=compact(baseline['free']), baseline_teacher=compact(baseline['teacher']),
        train_free=compact(train_free), train_teacher=compact(train_teacher),
        validation_free=compact(val_free), validation_teacher=compact(val_teacher),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=unchanged, frozen_origin_and_order_unchanged=frozen_held,
        modular_checkpoint_unchanged=stamp(MODULAR) == held,
        origin_checkpoint_unchanged=stamp(ORIGIN) == origin_held,
        full_9600=False, repeated_run=False, joint=False, test_used=False,
        geometry_solver=False, gt_origin_at_inference=False))
    print('DONE', json.dumps(dict(
        status=status, stop=reason, bottleneck=next_limit, update=best_state['update'],
        teacher=val_teacher['task'], free=val_free['task'],
        teacher_by={key: value['success'] for key, value in val_teacher['by_count'].items()},
        free_by={key: value['success'] for key, value in val_free['by_count'].items()},
        horizontal=val_teacher['horizontal_gap_mm'], free_horizontal=val_free['horizontal_gap_mm'],
        origin=val_teacher['origin_mm']['median'], train_horizontal=train_teacher['horizontal_gap_mm']['median'],
        row=val_free['row_accuracy'], slot=val_free['slot_accuracy'], oracle=oracle_end, single=single_end,
        frozen=frozen_held, checkpoint=str(ROOT/'checkpoint-best.pt'))), flush=True)


if __name__ == '__main__':
    main()
