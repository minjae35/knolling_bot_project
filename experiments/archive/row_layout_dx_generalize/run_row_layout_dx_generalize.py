"""One generalization run of direct Δx supervision on the 1,273 multi-row scenes.

The 40-scene Δx run reached a gap median of 0.45 mm by update 1200 and then
stayed flat through update 1450. This run therefore evaluates every 250
updates, keeps going through 1000 updates before a stall can stop it, and
stops after two evaluations with no free-rollout gain. The cap is 2500
updates, about 45 passes over the smallest class. Checkpoints keep the model,
optimizer, update, and settings. The best file is chosen only by validation
free-rollout success.
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
from knolling_model.experiments.row_layout import (
    NeuralRowPlacement, assignment_loss, order_loss, step_placement_loss)
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of
from run_row_layout_step import rejected_offset_matches
from run_row_transition import choose

ROOT = Path('artifacts/row_layout_dx_generalize')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
PROBE = Path('artifacts/row_layout/checkpoint-probe.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 250
MAX_UPDATES = 2500
MIN_UPDATES = 1000
PATIENCE = 2
ORACLE_FLOOR = 58
SINGLE_FLOOR = 56
CONFIG = dict(supervision='step_dx', lr=LR, per_class=PER_CLASS, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE,
              seed=1203, oracle_floor=ORACLE_FLOOR, single_floor=SINGLE_FLOOR,
              train_scenes=1273, val_scenes=152, coordinate_clip=10., selector_clip=1.,
              init='probe_partial_step_fresh')


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


def load_probe(model):
    saved = torch.load(PROBE, map_location=stable.DEVICE, weights_only=False)['model']
    own = model.state_dict()
    transferred = {key: value for key, value in saved.items() if key in own and own[key].shape == value.shape}
    if any(key.startswith('step.') for key in transferred) or not any(key.startswith('row_layout.') for key in transferred):
        raise RuntimeError('probe transfer does not match the current row module')
    own.update(transferred)
    model.load_state_dict(own)


def success_key(free):
    counts = free['by_count']
    return (free['task'], counts['6']['success'], counts['8']['success'])


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                position_mm=point['position_mm'], horizontal_gap_mm=point['horizontal_gap_mm'],
                vertical_gap_mm=point['vertical_gap_mm'], boundary=point['boundary'],
                collision=point['collision'], failures=point['failures'])


def same_pose(left, right):
    return all(torch.equal(left[key], right[key]) for key in ('xy', 'delta_x', 'angle', 'row_logits'))


def passed(free, oracle, single):
    median = free['position_mm']['median']
    return (free['task'] >= 20 and free['by_count']['6']['success'] >= 4 and free['by_count']['8']['success'] >= 4
            and median is not None and median <= 20 and oracle >= ORACLE_FLOOR and single >= SINGLE_FLOOR)


def main():
    if rejected_offset_matches():
        raise RuntimeError('step Δx loss matches the rejected offset loss')
    if (ROOT/'result.json').exists():
        raise RuntimeError('dx generalization result already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    probe_held = stamp(PROBE)
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
    probe_ids = {record['sample_id'] for record in choose(records)}
    fresh_ids = {records[index]['sample_id'] for count in (2, 4, 6, 8) for index in groups[count][8:16]}
    if probe_ids & fresh_ids:
        raise RuntimeError('the monitored scenes overlap')
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
    load_probe(model)
    optimizer = torch.optim.AdamW([
        dict(params=model.coordinate_parameters()),
        dict(params=model.selector_parameters()),
    ], lr=LR, weight_decay=0., foreach=False)
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single, train=dict(counts), config=CONFIG)), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    copy_optimizer = torch.optim.AdamW([
        dict(params=copy.coordinate_parameters()),
        dict(params=copy.selector_parameters()),
    ], lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    hidden = hidden_of(head, packed['features'])
    with torch.no_grad():
        original = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
        restored = copy(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
    reload_ok = same_pose(original, restored) and loaded['update'] == 0 and loaded['config'] == CONFIG
    print('RELOAD', json.dumps(dict(ok=reload_ok, update=loaded['update'])), flush=True)
    del copy, copy_optimizer, original, restored, hidden
    if not reload_ok:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload', modular_checkpoint_unchanged=stamp(MODULAR) == held))
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
        print('EVAL', json.dumps(dict(update=update, oracle=current_oracle, single=current_single,
                                     val=free['task'], by_count={key: value['success'] for key, value in free['by_count'].items()},
                                     mm=free['position_mm']['median'], gap=free['horizontal_gap_mm']['median'],
                                     teacher=teacher['task'], teacher_mm=teacher['position_mm']['median'],
                                     loss=None if train_loss is None else train_loss.get('dx'))), flush=True)
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
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
        placement = step_placement_loss(output, packed['target_xy'], packed['target_angle'], packed['valid'],
                                        packed['row'], packed['slot'])
        selector = assignment_loss(output, packed['row'], packed['valid'])+order_loss(output)
        if not torch.isfinite(placement['loss']) or not torch.isfinite(selector):
            reason = 'diverged'
            break
        (placement['loss']+selector).backward()
        if update == 1:
            def norm(parameters):
                return sum(parameter.grad.detach().pow(2).sum().item()
                           for parameter in parameters if parameter.grad is not None)**.5
            grad = dict(step=norm(model.step.parameters()), row_layout=norm(model.row_layout.parameters()),
                        selector=norm(model.selector_parameters()),
                        head=any(parameter.grad is not None for parameter in head.parameters()),
                        dx=float(placement['dx'].detach()), x=float(placement['x'].detach()),
                        y=float(placement['y'].detach()), ce=float(selector.detach()))
            print('PATH', json.dumps(grad), flush=True)
            if grad['step'] == 0 or grad['selector'] == 0 or grad['head']:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.coordinate_parameters(), CONFIG['coordinate_clip'])
        torch.nn.utils.clip_grad_norm_(model.selector_parameters(), CONFIG['selector_clip'])
        optimizer.step()
        running += float(placement['dx'].detach())
        seen += 1
        if update % EVAL_EVERY and update != MAX_UPDATES:
            continue
        save_checkpoint(ROOT/f'checkpoint-{update:04d}.pt', model, optimizer, update, CONFIG)
        point = record_eval(update, dict(dx=running/seen))
        running = 0.
        seen = 0
        if point['oracle'] < ORACLE_FLOOR or point['single'] < SINGLE_FLOOR:
            reason = 'regression'
            break
        improved = success_key(point['free']) > success_key(best['free'])
        tied = success_key(point['free']) == success_key(best['free'])
        median = point['free']['position_mm']['median']
        best_median = best['free']['position_mm']['median']
        closer = tied and median is not None and best_median is not None and median < best_median
        if improved or closer:
            best = point
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        stale = 0 if improved else stale+1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    train_free = evaluate_split(stored, records, 'free')
    train_teacher = evaluate_split(stored, records, 'teacher')
    val_free = evaluate_split(val_stored, multi_val, 'free')
    val_teacher = evaluate_split(val_stored, multi_val, 'teacher')
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    status = 'PASS' if passed(val_free, oracle_end, single_end) else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, best_update=best_state['update'], best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        reload_matches_logged_success=val_free['task'] == best['free']['task'],
        config=CONFIG, train_free=compact(train_free), train_teacher=compact(train_teacher),
        validation_free=compact(val_free), validation_teacher=compact(val_teacher),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=unchanged, modular_checkpoint_unchanged=stamp(MODULAR) == held,
        probe_checkpoint_unchanged=stamp(PROBE) == probe_held, full_9600=False, repeated_run=False,
        joint=False, test_used=False, geometry_solver=False))
    print('DONE', json.dumps(dict(status=status, stop=reason, update=best_state['update'],
                                 train=train_free['task'], val=val_free['task'],
                                 by_count={key: value['success'] for key, value in val_free['by_count'].items()},
                                 mm=val_free['position_mm']['median'], teacher=val_teacher['task'],
                                 oracle=oracle_end, single=single_end, unchanged=unchanged,
                                 checkpoint=str(ROOT/'checkpoint-best.pt'))), flush=True)


if __name__ == '__main__':
    main()
