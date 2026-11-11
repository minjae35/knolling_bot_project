"""Train only the row-assignment head. Coordinates and order stay at the adopted checkpoint.

The earlier selector run reached its row accuracy by update 250 and did not
move after that, so this run evaluates every 100 updates and stops after two
flat validation checks once 200 updates have passed. The cap is 500.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement, assignment_loss
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of

ROOT = Path('artifacts/row_layout_assign')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 100
MAX_UPDATES = 500
MIN_UPDATES = 200
PATIENCE = 2
CONFIG = dict(supervision='assignment_ce', lr=LR, per_class=PER_CLASS, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE, seed=1203,
              train_scenes=1273, val_scenes=152, init='origin_y_best_assign_only', coordinate_clip=1.)


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
        parameter.requires_grad_(name.startswith('assign.'))


def label_audit(records):
    counts = Counter()
    bad = 0
    for record in records:
        rows = record['supervision_v3']['rows']
        ids = [oid for row in rows for oid in row]
        source = {obj['object_id'] for obj in record['source']['objects']}
        if len(ids) != len(set(ids)) or set(ids) != source or any(len(row) == 0 for row in rows):
            bad += 1
        if max(range(len(rows))) >= 8:
            bad += 1
        for index, row in enumerate(rows):
            counts[index] += len(row)
    return dict(objects=sum(counts.values()), bad=bad, counts={str(key): counts[key] for key in sorted(counts)})


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                row_exact_scenes=point['row_exact_scenes'], horizontal_gap_mm=point['horizontal_gap_mm'],
                failures=point['failures'])


def success_ids(model, head, stored, records):
    found = set()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk = records[start:start+16]
            packed = collate_rows(stored[start:start+len(chunk)], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                           None, None, True)
            for local, record in enumerate(chunk):
                kept = stored[start+local]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                ids, truth = recovery_supervision(record)
                layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                              angle=output['angle'][local:local+1, :kept].cpu(),
                              valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [dict(
                    sample_id=record['sample_id'], object_ids=ids, record=record)]),
                    semantic_check=semantic_errors_v3)[0]
                if judged['task_success']:
                    found.add(record['sample_id'])
    return found


def passed(teacher, free, oracle, single, baseline):
    return (free['row_accuracy'] > baseline['free']['row_accuracy'] and free['task'] > baseline['free']['task']
            and free['by_count']['6']['success'] >= baseline['free']['by_count']['6']['success']
            and free['by_count']['8']['success'] >= baseline['free']['by_count']['8']['success']
            and teacher['task'] >= 95 and oracle >= 60 and single >= 62)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row assignment result already exists')
    if not BEST.is_file():
        raise RuntimeError('adopted row checkpoint is missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    best_held = stamp(BEST)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    records = [record for record in train_data.records if multi_row(record)]
    multi_val = [record for record in val_data.records if multi_row(record)]
    audit = dict(train=label_audit(records), val=label_audit(multi_val))
    print('AUDIT', json.dumps(audit), flush=True)
    if audit['train']['bad'] or audit['val']['bad'] or audit['train']['objects'] != 6782:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='labels', audit=audit, trained=False,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              best_checkpoint_unchanged=stamp(BEST) == best_held))
        print('STOP row labels do not cover every object once', flush=True)
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
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    head = cap.load_head(128, torch.load(MODULAR, map_location='cpu', weights_only=False)['model'])
    head.requires_grad_(False)
    head.eval()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(body, phase_val)
    stored = attach_rows(records, cache_records(body, records))
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    single_stored = cache_records(body, single_val)
    del body
    model = NeuralRowPlacement().to(stable.DEVICE)
    model.load_state_dict(torch.load(BEST, map_location=stable.DEVICE, weights_only=False)['model'])
    model.eval()
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    free = evaluate(model, head, val_stored, multi_val, 'free')
    print('RELOAD', json.dumps(dict(oracle=oracle, single=single, teacher=teacher['task'], free=free['task'],
                                   row=free['row_accuracy'],
                                   by={key: value['success'] for key, value in free['by_count'].items()})), flush=True)
    if (teacher['task'], free['task'], oracle, single) != (95, 66, 60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reproduction', trained=False, audit=audit,
                                              teacher=teacher['task'], free=free['task'], oracle=oracle, single=single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              best_checkpoint_unchanged=stamp(BEST) == best_held))
        print('STOP adopted checkpoint did not reproduce', flush=True)
        return
    baseline_ids = success_ids(model, head, val_stored, multi_val)
    if len(baseline_ids) != 66:
        raise RuntimeError('free success ids do not match the reloaded score')
    prepare(model)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not name.startswith('assign.')}
    optimizer = torch.optim.AdamW(model.assign.parameters(), lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    packed = collate_rows([stored[index] for index in batch_indices(cycles)], stable.DEVICE)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    prepare(copy)
    copy_optimizer = torch.optim.AdamW(copy.assign.parameters(), lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    hidden = hidden_of(head, packed['features'])
    with torch.no_grad():
        original = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1], None, None, True)
        restored = copy(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1], None, None, True)
    reload_ok = torch.equal(original['row_logits'], restored['row_logits']) and loaded['update'] == 0
    print('CHECKPOINT', json.dumps(dict(ok=reload_ok)), flush=True)
    del copy, copy_optimizer, original, restored, hidden
    if not reload_ok:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload', trained=False,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              best_checkpoint_unchanged=stamp(BEST) == best_held))
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
            teacher=current_teacher['task'], free=current_free['task'], row=current_free['row_accuracy'],
            by={key: value['success'] for key, value in current_free['by_count'].items()},
            loss=train_loss)), flush=True)
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
        loss = assignment_loss(output, packed['row'], packed['valid'])
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        if update == 1:
            assign_grad = sum(parameter.grad.detach().pow(2).sum().item()
                              for parameter in model.assign.parameters() if parameter.grad is not None)**.5
            leaked = [name for name, parameter in model.named_parameters()
                      if not name.startswith('assign.') and parameter.grad is not None]
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            print('PATH', json.dumps(dict(assign_grad=assign_grad, leaked=leaked, head_grad=head_grad,
                                         loss=float(loss.detach()))), flush=True)
            if assign_grad == 0 or leaked or head_grad:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.assign.parameters(), CONFIG['coordinate_clip'])
        optimizer.step()
        if any(not torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
               if not name.startswith('assign.')):
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
        if point['oracle'] < 60 or point['single'] < 62 or point['teacher']['task'] < 95:
            reason = 'regression'
            break
        better = (point['free']['task'], point['free']['row_accuracy']) > (best['free']['task'], best['free']['row_accuracy'])
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
    if reason in ('path_blocked', 'frozen_moved', 'diverged'):
        stable.write(ROOT/'result.json', dict(status='STOP', stage=reason, trained=False, audit=audit,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held,
                                              best_checkpoint_unchanged=stamp(BEST) == best_held))
        print('STOP', reason, flush=True)
        return
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    prepare(reloaded)
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    val_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    val_free = evaluate(model, head, val_stored, multi_val, 'free')
    kept_ids = success_ids(model, head, val_stored, multi_val)
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    status = 'PASS' if passed(val_teacher, val_free, oracle_end, single_end, baseline) else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, best_update=best_state['update'], best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        reload_matches_logged_success=val_free['task'] == best['free']['task'],
        preserved_success=len(baseline_ids & kept_ids)/len(baseline_ids),
        config=CONFIG, audit=audit, baseline_teacher=baseline['teacher'], baseline_free=baseline['free'],
        validation_teacher=compact(val_teacher), validation_free=compact(val_free),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters()),
        coordinates_unchanged=all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                                  if not name.startswith('assign.')),
        modular_checkpoint_unchanged=stamp(MODULAR) == held,
        adopted_checkpoint_unchanged=stamp(BEST) == best_held,
        full_9600=False, joint=False, test_used=False, geometry_solver=False))
    print('DONE', json.dumps(dict(
        status=status, stop=reason, update=best_state['update'],
        row_before=baseline['free']['row_accuracy'], row=val_free['row_accuracy'],
        free_before=66, free=val_free['task'], teacher=val_teacher['task'],
        by={key: value['success'] for key, value in val_free['by_count'].items()},
        preserved=len(baseline_ids & kept_ids)/len(baseline_ids),
        oracle=oracle_end, single=single_end, checkpoint=str(ROOT/'checkpoint-best.pt'))), flush=True)


if __name__ == '__main__':
    main()
