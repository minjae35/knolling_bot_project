"""Generalize the saved row layout on the filtered multi-row training scenes.

One run. The relation head and the encoders stay frozen. Teacher rows and
order are training labels only; the kept score is free rollout.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import NeuralRowPlacement, assignment_loss, order_loss, placement_loss
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of
from run_row_transition import choose

ROOT = Path('artifacts/row_layout_generalize')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
PROBE = Path('artifacts/row_layout/checkpoint-probe.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
PATIENCE = 6
MAX_EPOCHS = 80
ORACLE_FLOOR = 58
SINGLE_FLOOR = 56


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


def successes(free, count):
    value = free['by_count'][str(count)]
    return value['success'] if isinstance(value, dict) else value


def rank_of(free):
    return (free['task'], successes(free, 6)+successes(free, 8), -free['position_mm']['median'])


def passed(free, oracle, single):
    median = free['position_mm']['median']
    return (free['task'] >= 20 and successes(free, 6) >= 4 and successes(free, 8) >= 4
            and median is not None and median <= 20 and oracle >= ORACLE_FLOOR and single >= SINGLE_FLOOR)


def compact(point):
    return dict(task=point['task'], by_count={key: value['success'] for key, value in point['by_count'].items()},
                row=point['row_accuracy'], slot=point['slot_accuracy'], position_mm=point['position_mm'],
                horizontal_gap_mm=point['horizontal_gap_mm'], vertical_gap_mm=point['vertical_gap_mm'],
                boundary=point['boundary'], collision=point['collision'], failures=point['failures'])


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row layout generalization result already exists')
    if not PROBE.is_file():
        raise RuntimeError('the eight-scene row layout checkpoint is missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    probe_stamp = stamp(PROBE)
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
        groups[count] = [index for index, record in enumerate(records)
                         if len(record['source']['objects']) == count]
        groups[count].sort(key=lambda index: records[index]['sample_id'])
        if len(groups[count]) < 16:
            raise RuntimeError('a count pool is too small for a balanced batch and monitor')
    probe_ids = {record['sample_id'] for record in choose(records)}
    fresh_ids = {records[index]['sample_id'] for count in (2, 4, 6, 8) for index in groups[count][8:16]}
    if probe_ids & fresh_ids:
        raise RuntimeError('the fresh training monitor overlaps the memorized eight')
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    head.requires_grad_(False)
    head.eval()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(body, phase_val)
    stored = attach_rows(records, cache_records(body, records))
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    single_stored = cache_records(body, single_val)
    del body
    model = NeuralRowPlacement().to(stable.DEVICE)
    model.load_state_dict(torch.load(PROBE, map_location=stable.DEVICE, weights_only=False)['model'])
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single, train=dict(counts))), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    epoch_steps = min(len(group)//PER_CLASS for group in groups.values())
    fresh_index = [index for count in (2, 4, 6, 8) for index in groups[count][8:16]]
    probe_index = [index for index, record in enumerate(records) if record['sample_id'] in probe_ids]
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()

    def subset_eval(indices, mode):
        return evaluate(model, head, [stored[index] for index in indices], [records[index] for index in indices], mode)

    def measure(step, loss):
        free = evaluate(model, head, val_stored, multi_val, 'free')
        teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
        fresh = subset_eval(fresh_index, 'free')
        probe = subset_eval(probe_index, 'free')
        current_oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
        current_single = score(head, single_stored, single_val)['task']
        point = dict(update=step, loss=loss, oracle=current_oracle, single=current_single,
                     free=compact(free), teacher=compact(teacher), fresh=compact(fresh), probe=compact(probe),
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EPOCH', json.dumps(dict(update=step, oracle=current_oracle, single=current_single,
                                      val=free['task'], by_count=point['free']['by_count'],
                                      mm=free['position_mm']['median'], teacher_task=teacher['task'],
                                      teacher_mm=teacher['position_mm']['median'], fresh=fresh['task'],
                                      probe=probe['task'], loss=loss)), flush=True)
        return point, free, teacher

    baseline, _, baseline_teacher = measure(0, None)
    packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
    hidden = hidden_of(head, packed['features'])
    preview = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
    coordinate = placement_loss(preview, packed['target_xy'], packed['target_angle'], packed['valid'])['loss']
    selector = assignment_loss(preview, packed['row'], packed['valid'])+order_loss(preview)
    weight = float((coordinate.detach()/selector.detach().clamp_min(1e-6)).clamp(.05, 10.))
    print('LOSS', json.dumps(dict(coordinate=float(coordinate.detach()), selector=float(selector.detach()),
                                 weight=weight, epoch_steps=epoch_steps)), flush=True)
    del preview, coordinate, selector, hidden, packed
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0., foreach=False)
    best = baseline
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_teacher_mm = baseline_teacher['position_mm']['median']
    stale = 0
    reason = 'plateau'
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        total = 0.
        stopped = False
        for _ in range(epoch_steps):
            optimizer.zero_grad(set_to_none=True)
            packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
            losses = placement_loss(output, packed['target_xy'], packed['target_angle'], packed['valid'])
            loss = losses['loss']+weight*(assignment_loss(output, packed['row'], packed['valid'])+order_loss(output))
            if not torch.isfinite(loss):
                reason = 'diverged'
                stopped = True
                break
            loss.backward()
            if epoch == 1 and total == 0.:
                head_grad = any(parameter.grad is not None for parameter in head.parameters())
                module_grad = sum(parameter.grad.abs().sum().item() for parameter in model.parameters()
                                  if parameter.grad is not None)
                print('PATH', json.dumps(dict(module=module_grad, head=head_grad)), flush=True)
                if module_grad == 0 or head_grad:
                    reason = 'path_blocked'
                    stopped = True
                    break
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += float(loss.detach())
        if stopped:
            break
        point, _, teacher = measure(epoch*epoch_steps, total/epoch_steps)
        if point['oracle'] < ORACLE_FLOOR or point['single'] < SINGLE_FLOOR:
            reason = 'regression'
            break
        improved = rank_of(point['free']) > rank_of(best['free'])
        teacher_mm = teacher['position_mm']['median']
        if teacher_mm is not None and best_teacher_mm is not None and teacher_mm <= best_teacher_mm-2:
            improved = True
        if teacher_mm is not None and (best_teacher_mm is None or teacher_mm < best_teacher_mm):
            best_teacher_mm = teacher_mm
        if improved and rank_of(point['free']) >= rank_of(best['free']):
            best = point
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(dict(model=best_state, update=point['update']), ROOT/'checkpoint-best.pt')
        stale = 0 if improved else stale+1
        if stale >= PATIENCE:
            reason = 'plateau'
            break
    else:
        reason = 'max_epochs'
    history.close()
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    train_improved = best['fresh']['task'] > baseline['fresh']['task'] or (
        best['fresh']['position_mm']['median'] is not None and baseline['fresh']['position_mm']['median'] is not None
        and best['fresh']['position_mm']['median'] <= baseline['fresh']['position_mm']['median']-1)
    val_improved = best['free']['task'] > baseline['free']['task']
    status = 'PASS' if passed(best['free'], best['oracle'], best['single']) and train_improved and val_improved else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, update=best['update'], lr=LR, epoch_steps=epoch_steps, selector_weight=weight,
        initialized_from='artifacts/row_layout/checkpoint-probe.pt',
        initialization_reason='the eight-scene module already emits row coordinates; this run asks whether more scenes move validation',
        baseline=dict(free=baseline['free'], teacher=baseline['teacher'], fresh=baseline['fresh'], probe=baseline['probe']),
        best=dict(free=best['free'], teacher=best['teacher'], fresh=best['fresh'], probe=best['probe'],
                  oracle=best['oracle'], single=best['single']),
        train_improved=train_improved, validation_improved=val_improved,
        oracle_start=oracle, single_start=single, oracle_end=best['oracle'], single_end=best['single'],
        relation_parameters_unchanged=unchanged, modular_checkpoint_unchanged=stamp(MODULAR) == held,
        probe_checkpoint_unchanged=stamp(PROBE) == probe_stamp, architecture_changed=False,
        full_9600=False, joint=False, test_used=False, adopted=False))
    print('DONE', json.dumps(dict(status=status, stop=reason, update=best['update'], val=best['free']['task'],
                                 by_count=best['free']['by_count'], mm=best['free']['position_mm']['median'],
                                 fresh=best['fresh']['task'], oracle=best['oracle'], single=best['single'],
                                 unchanged=unchanged)), flush=True)


if __name__ == '__main__':
    main()
