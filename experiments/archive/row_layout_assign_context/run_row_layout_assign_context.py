"""Train the set-attention row assignment. Coordinates, order, and encoders stay fixed.

The pointwise head had already plateaued, so this run only learns the assignment
MLP and its object-set attention. Evaluation follows the earlier assignment
schedule: every 100 updates, stop after two flat checks once 200 updates passed.
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
from knolling_model.experiments.row_layout import NeuralRowPlacement, assignment_loss, load_row_weights
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

ROOT = Path('artifacts/row_layout_assign_context')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 100
MAX_UPDATES = 600
MIN_UPDATES = 200
PATIENCE = 2
CONFIG = dict(supervision='assignment_set_attention', lr=LR, per_class=PER_CLASS, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE, seed=1203,
              train_scenes=1273, val_scenes=152, init='origin_y_best_zero_context', coordinate_clip=1.)


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


def assignment_name(name):
    return name.startswith('assign.') or name.startswith('assign_set.')


def prepare(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(assignment_name(name))


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                row_exact_scenes=point['row_exact_scenes'], failures=point['failures'])


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
    return (free['row_accuracy'] >= .74 and free['task'] >= 74
            and free['by_count']['6']['success'] > baseline['free']['by_count']['6']['success']
            and free['by_count']['8']['success'] > baseline['free']['by_count']['8']['success']
            and teacher['task'] >= 95 and oracle >= 60 and single >= 62)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row context result already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    best_held = stamp(BEST)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    records = [record for record in train_data.records if multi_row(record)]
    multi_val = [record for record in val_data.records if multi_row(record)]
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
    load_row_weights(model, torch.load(BEST, map_location=stable.DEVICE, weights_only=False)['model'])
    model.eval()
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    free = evaluate(model, head, val_stored, multi_val, 'free')
    print('RELOAD', json.dumps(dict(oracle=oracle, single=single, teacher=teacher['task'], free=free['task'],
                                   row=free['row_accuracy'],
                                   by={key: value['success'] for key, value in free['by_count'].items()})), flush=True)
    if (teacher['task'], free['task'], oracle, single) != (95, 66, 60, 62):
        raise RuntimeError('zero-context assignment did not reproduce the adopted checkpoint')
    baseline_ids = success_ids(model, head, val_stored, multi_val)
    if len(baseline_ids) != 66:
        raise RuntimeError('free success ids do not match the reloaded score')
    prepare(model)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not assignment_name(name)}
    optimizer = torch.optim.AdamW([parameter for name, parameter in model.named_parameters() if assignment_name(name)],
                                  lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    packed = collate_rows([stored[index] for index in cycles[0].next(1)], stable.DEVICE)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    prepare(copy)
    copy_optimizer = torch.optim.AdamW(
        [parameter for name, parameter in copy.named_parameters() if assignment_name(name)],
        lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    hidden = hidden_of(head, packed['features'])
    with torch.no_grad():
        original = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1], None, None, True)
        restored = copy(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1], None, None, True)
    if not (torch.equal(original['row_logits'], restored['row_logits']) and loaded['update'] == 0):
        raise RuntimeError('assignment checkpoint reload changed the row logits')
    print('CHECKPOINT', json.dumps(dict(ok=True)), flush=True)
    del copy, copy_optimizer, original, restored, hidden
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()
    baseline = dict(update=0, oracle=oracle, single=single, teacher=compact(teacher), free=compact(free))

    def record_eval(update, train_loss):
        current_train = evaluate(model, head, stored, records, 'free')
        current_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
        current_free = evaluate(model, head, val_stored, multi_val, 'free')
        current_oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
        current_single = score(head, single_stored, single_val)['task']
        point = dict(update=update, loss=train_loss, oracle=current_oracle, single=current_single,
                     train=compact(current_train), teacher=compact(current_teacher), free=compact(current_free),
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(
            update=update, oracle=current_oracle, single=current_single, teacher=current_teacher['task'],
            train_row=current_train['row_accuracy'], train_task=current_train['task'],
            free=current_free['task'], row=current_free['row_accuracy'],
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
        packed = collate_rows([stored[index] for index in
                               [index for cycle in cycles for index in cycle.next(PER_CLASS)]], stable.DEVICE)
        hidden = hidden_of(head, packed['features'])
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                       packed['row'], packed['slot'], False)
        loss = assignment_loss(output, packed['row'], packed['valid'])
        if not torch.isfinite(loss):
            raise RuntimeError('assignment loss diverged')
        loss.backward()
        if update == 1:
            context_grad = model.assign[0].weight.grad[:, model.hidden_dim:-13]
            leaked = [name for name, parameter in model.named_parameters()
                      if not assignment_name(name) and parameter.grad is not None]
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            context_norm = float(context_grad.detach().pow(2).sum().sqrt())
            print('PATH', json.dumps(dict(context_grad=context_norm, leaked=leaked, head_grad=head_grad,
                                         loss=float(loss.detach()))), flush=True)
            if context_norm == 0 or leaked or head_grad:
                raise RuntimeError('assignment gradient does not stay inside the new head')
        if update == 2:
            value_grad = model.assign_set.value.weight.grad
            query_grad = model.assign_set.query.weight.grad
            if value_grad is None or query_grad is None or float(value_grad.detach().abs().sum()) == 0 or float(query_grad.detach().abs().sum()) == 0:
                raise RuntimeError('set attention received no gradient')
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad],
                                       CONFIG['coordinate_clip'])
        optimizer.step()
        if any(not torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
               if not assignment_name(name)):
            raise RuntimeError('a frozen row parameter moved')
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
        rank = (point['free']['task'], point['free']['row_accuracy'],
                point['free']['by_count']['6']['success'], point['free']['by_count']['8']['success'])
        held_rank = (best['free']['task'], best['free']['row_accuracy'],
                     best['free']['by_count']['6']['success'], best['free']['by_count']['8']['success'])
        if rank > held_rank:
            best = point
            stale = 0
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    prepare(reloaded)
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    train_free = evaluate(model, head, stored, records, 'free')
    val_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    val_free = evaluate(model, head, val_stored, multi_val, 'free')
    kept_ids = success_ids(model, head, val_stored, multi_val)
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    status = 'PASS' if passed(val_teacher, val_free, oracle_end, single_end, baseline) else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, best_update=best_state['update'], best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        reload_matches_logged_success=val_free['task'] == best['free']['task'] and val_free['row_accuracy'] == best['free']['row_accuracy'],
        preserved_success=len(baseline_ids & kept_ids)/len(baseline_ids),
        lost_success=sorted(baseline_ids-kept_ids),
        config=CONFIG, baseline_teacher=baseline['teacher'], baseline_free=baseline['free'],
        train_free=compact(train_free), validation_teacher=compact(val_teacher), validation_free=compact(val_free),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters()),
        coordinates_unchanged=all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                                  if not assignment_name(name)),
        modular_checkpoint_unchanged=stamp(MODULAR) == held,
        adopted_checkpoint_unchanged=stamp(BEST) == best_held,
        full_9600=False, joint=False, test_used=False, geometry_solver=False))
    print('DONE', json.dumps(dict(
        status=status, stop=reason, update=best_state['update'],
        row_before=baseline['free']['row_accuracy'], row=val_free['row_accuracy'],
        train_row=train_free['row_accuracy'], train_task=train_free['task'],
        free_before=66, free=val_free['task'], teacher=val_teacher['task'],
        by={key: value['success'] for key, value in val_free['by_count'].items()},
        preserved=len(baseline_ids & kept_ids)/len(baseline_ids),
        oracle=oracle_end, single=single_end)), flush=True)


if __name__ == '__main__':
    main()
