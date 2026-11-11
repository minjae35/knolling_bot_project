"""Train only the row-origin head so it can see the other rows.

The head reads each row's pooled features and sizes, plus the summed features
of the rows above it and below it. Relation, row/order, angle, and the
within-row step stay at the Δx checkpoint. One run, then stop.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.origin_audit import cross_row_height_reaches_origin
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement, origin_loss, target_step
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate, stats
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of

ROOT = Path('artifacts/row_layout_origin')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DX = Path('artifacts/row_layout_dx_generalize/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 250
MAX_UPDATES = 1500
MIN_UPDATES = 500
PATIENCE = 2
ORACLE_FLOOR = 58
SINGLE_FLOOR = 56
BASELINE_ORIGIN_MM = 22.06244659423828
BASELINE_FREE = 6
BASELINE_TEACHER = 3
CONFIG = dict(supervision='row_origin', aggregation='above_below_sum', lr=LR, per_class=PER_CLASS,
              eval_every=EVAL_EVERY, max_updates=MAX_UPDATES, min_updates=MIN_UPDATES,
              patience_evals=PATIENCE, seed=1203, oracle_floor=ORACLE_FLOOR, single_floor=SINGLE_FLOOR,
              baseline_origin_mm=BASELINE_ORIGIN_MM, baseline_free=BASELINE_FREE,
              baseline_teacher=BASELINE_TEACHER, train_scenes=1273, val_scenes=152,
              init='dx_best_origin_fresh', coordinate_clip=10.)


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
        parameter.requires_grad_(name.startswith('row_layout.'))


def load_dx(model):
    saved = torch.load(DX, map_location=stable.DEVICE, weights_only=False)['model']
    own = model.state_dict()
    if saved['row_layout.0.weight'].shape == own['row_layout.0.weight'].shape:
        raise RuntimeError('the old origin head matches the new input and was not replaced')
    transferred = {key: value for key, value in saved.items()
                   if key in own and own[key].shape == value.shape and not key.startswith('row_layout.')}
    needed = ('step.', 'angle.', 'assign.', 'order_query.', 'order_begin')
    if any(not any(key.startswith(prefix) for key in transferred) for prefix in needed):
        raise RuntimeError('the frozen row module did not load from the Δx checkpoint')
    own.update(transferred)
    model.load_state_dict(own)


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                position_mm=point['position_mm'], origin_mm=point['origin_mm'],
                origin_x_mm=point['origin_x_mm'], origin_y_mm=point['origin_y_mm'],
                horizontal_gap_mm=point['horizontal_gap_mm'], vertical_gap_mm=point['vertical_gap_mm'],
                boundary=point['boundary'], collision=point['collision'], failures=point['failures'])


def same_pose(left, right):
    return all(torch.equal(left[key], right[key]) for key in ('xy', 'delta_x', 'angle', 'row_logits'))


def passed(teacher, free, oracle, single):
    origin = teacher['origin_mm']['median']
    return (origin is not None and origin < 20 and free['task'] > BASELINE_FREE and teacher['task'] > BASELINE_TEACHER
            and oracle >= ORACLE_FLOOR and single >= SINGLE_FLOOR)


def step_error_mm(model, head, stored):
    errors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(stored), 16):
            packed = collate_rows(stored[start:start+16], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                           packed['row'], packed['slot'], False)
            target, covered = target_step(packed['target_xy'], packed['valid'], packed['row'], packed['slot'])
            if bool(covered.any()):
                errors.extend(((output['delta_x']-target).abs()*1000)[covered].detach().cpu().tolist())
    return stats(errors)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row origin result already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    dx_held = stamp(DX)
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    records = [record for record in train_data.records if multi_row(record)]
    multi_val = [record for record in val_data.records if multi_row(record)]
    audit = cross_row_height_reaches_origin(NeuralRowPlacement(), records+multi_val)
    print('AUDIT', json.dumps(audit), flush=True)
    if not audit['reached']:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='origin_input', audit=audit,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              dx_checkpoint_unchanged=stamp(DX) == dx_held, trained=False))
        print('STOP origin input does not carry the other row height', flush=True)
        return
    counts = Counter(len(record['source']['objects']) for record in records)
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    phase_val = subset(val_data, 'validation', 94)
    if sum(counts.values()) != 1273 or len(multi_val) != 152 or len(single_val) != 104:
        raise RuntimeError(f'unexpected pools train={dict(counts)} val={len(multi_val)} single={len(single_val)}')
    groups = {}
    for count in (2, 4, 6, 8):
        groups[count] = [index for index, record in enumerate(records) if len(record['source']['objects']) == count]
        groups[count].sort(key=lambda index: records[index]['sample_id'])
    runner.setup()
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
    load_dx(model)
    prepare(model)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not name.startswith('row_layout.')}
    optimizer = torch.optim.AdamW(model.origin_parameters(), lr=LR, weight_decay=0., foreach=False)
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single, train=dict(counts), config=CONFIG)), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              audit=audit, modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    prepare(copy)
    copy_optimizer = torch.optim.AdamW(copy.origin_parameters(), lr=LR, weight_decay=0., foreach=False)
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
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload', audit=audit,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
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
            by_count={key: value['success'] for key, value in free['by_count'].items()},
            teacher=teacher['task'], origin=teacher['origin_mm']['median'],
            origin_x=teacher['origin_x_mm']['median'], origin_y=teacher['origin_y_mm']['median'],
            horizontal=teacher['horizontal_gap_mm']['median'], vertical=teacher['vertical_gap_mm']['median'],
            loss=train_loss)), flush=True)
        return point

    baseline = record_eval(0, None)
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
        loss = origin_loss(output, packed['target_xy'], packed['valid'], packed['slot'])
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        if update == 1:
            moved = [name for name, parameter in model.named_parameters()
                     if not name.startswith('row_layout.') and parameter.grad is not None]
            origin_grad = sum(parameter.grad.detach().pow(2).sum().item()
                              for parameter in model.origin_parameters() if parameter.grad is not None)**.5
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            print('PATH', json.dumps(dict(origin_grad=origin_grad, frozen_grads=moved, head_grad=head_grad,
                                         loss=float(loss.detach()))), flush=True)
            if origin_grad == 0 or moved or head_grad:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.origin_parameters(), CONFIG['coordinate_clip'])
        optimizer.step()
        if update == 1:
            changed = [name for name, parameter in model.named_parameters()
                       if not name.startswith('row_layout.') and not torch.equal(parameter, frozen[name])]
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
        origin = point['teacher']['origin_mm']['median']
        best_origin = best['teacher']['origin_mm']['median']
        improved = origin is not None and best_origin is not None and origin < best_origin
        if improved:
            best = point
            stale = 0
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    if reason in ('path_blocked', 'frozen_moved', 'diverged'):
        stable.write(ROOT/'result.json', dict(status='STOP', stage=reason, audit=audit,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              dx_checkpoint_unchanged=stamp(DX) == dx_held, trained=False))
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
    dx_error = step_error_mm(model, head, val_stored)
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    frozen_held = all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                      if not name.startswith('row_layout.'))
    status = 'PASS' if passed(val_teacher, val_free, oracle_end, single_end) else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, best_update=best_state['update'], best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        reload_matches_logged_success=val_teacher['origin_mm']['median'] == best['teacher']['origin_mm']['median'],
        config=CONFIG, audit=audit, train_free=compact(train_free), train_teacher=compact(train_teacher),
        validation_free=compact(val_free), validation_teacher=compact(val_teacher),
        remaining_dx_mm=dx_error, oracle_start=oracle, single_start=single,
        oracle_end=oracle_end, single_end=single_end, relation_parameters_unchanged=unchanged,
        frozen_row_module_unchanged=frozen_held, modular_checkpoint_unchanged=stamp(MODULAR) == held,
        dx_checkpoint_unchanged=stamp(DX) == dx_held, full_9600=False, repeated_run=False,
        joint=False, test_used=False, geometry_solver=False, gt_origin_at_inference=False))
    print('DONE', json.dumps(dict(
        status=status, stop=reason, update=best_state['update'], origin=val_teacher['origin_mm']['median'],
        origin_x=val_teacher['origin_x_mm']['median'], origin_y=val_teacher['origin_y_mm']['median'],
        teacher=val_teacher['task'], free=val_free['task'],
        by_count={key: value['success'] for key, value in val_free['by_count'].items()},
        train_origin=train_teacher['origin_mm']['median'], dx=dx_error['median'],
        horizontal=val_teacher['horizontal_gap_mm']['median'], oracle=oracle_end, single=single_end,
        unchanged=unchanged, frozen=frozen_held, checkpoint=str(ROOT/'checkpoint-best.pt'))), flush=True)


if __name__ == '__main__':
    main()
