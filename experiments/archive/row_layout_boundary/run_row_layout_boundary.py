"""Train the row-boundary head. Order, origin, step, and the encoders stay fixed.

Cuts are supervised on the order head's predicted sequence. GT rows are the
labels for those cuts, not an inference input. The run stops after two flat
validation checks once 200 updates have passed.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.coordinates import wrap_pi
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import (
    NeuralRowPlacement, boundary_loss, load_row_weights, rows_from_boundaries, _ordered_indices)
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

ROOT = Path('artifacts/row_layout_boundary')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PER_CLASS = 4
EVAL_EVERY = 100
MAX_UPDATES = 500
MIN_UPDATES = 200
PATIENCE = 2
CONFIG = dict(supervision='boundary_bce', lr=LR, per_class=PER_CLASS, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE, seed=1203,
              train_scenes=1273, val_scenes=152, init='origin_y_best_boundary_only', coordinate_clip=1.)


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


def compact(point):
    return dict(task=point['task'], total=point['total'], by_count=point['by_count'],
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                row_exact_scenes=point['row_exact_scenes'], failures=point['failures'])


def classifier_pose(model, hidden, valid, goal, widths, heights):
    """The adopted free path, used only to name the 66 scenes being preserved."""
    count = hidden.shape[1]
    logits = model.assign(torch.cat((hidden, goal[:, None].expand(-1, count, -1)), -1))
    row = torch.where(valid, logits.argmax(-1), torch.full(valid.shape, -1, dtype=torch.long, device=hidden.device))
    slot, _, _ = model._order(hidden, valid, goal, row, None)
    origins = model._origins(hidden, valid, goal, row, widths, heights, slot)
    xy, _ = model._positions(valid, goal, row, slot, widths, origins)
    raw = model.angle(torch.cat((hidden, goal[:, None].expand(-1, count, -1)), -1))
    angle = wrap_pi(.5*torch.atan2(raw[..., 1], raw[..., 0])).masked_fill(~valid, 0)
    return xy, angle


def successes(model, head, stored, records, source):
    found = set()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk = records[start:start+16]
            packed = collate_rows(stored[start:start+len(chunk)], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            widths, heights = packed['size'][..., 0], packed['size'][..., 1]
            if source == 'classifier':
                xy, angle = classifier_pose(model, hidden, packed['valid'], packed['goal'], widths, heights)
            else:
                output = model(hidden, packed['valid'], packed['goal'], widths, heights, None, None, True)
                xy, angle = output['xy'], output['angle']
            for local, record in enumerate(chunk):
                kept = stored[start+local]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                ids, truth = recovery_supervision(record)
                layout = dict(xy=xy[local:local+1, :kept].cpu(), angle=angle[local:local+1, :kept].cpu(),
                              valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [dict(
                    sample_id=record['sample_id'], object_ids=ids, record=record)]),
                    semantic_check=semantic_errors_v3)[0]
                if judged['task_success']:
                    found.add(record['sample_id'])
    return found


def boundary_accuracy(model, head, stored, records):
    hits = total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), 16):
            packed = collate_rows(stored[start:start+16], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                           packed['row'], packed['slot'], False)
            mask = output['boundary_mask']
            left = packed['row'].gather(1, output['boundary_left'])
            right = packed['row'].gather(1, output['boundary_right'])
            matched = ((output['boundary_logits'] > 0) == (left != right)) & mask
            hits += int(matched.sum())
            total += int(mask.sum())
    return hits/total


def diagnosis(model, head, stored, records):
    express = ceiling_hit = ceiling_n = 0
    reasons = Counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk = records[start:start+16]
            packed = collate_rows(stored[start:start+len(chunk)], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            order = model._global_order(hidden, packed['valid'], packed['goal'])
            placed, lengths = _ordered_indices(order, packed['valid'])
            cuts = torch.zeros(placed.shape[0], placed.shape[1]-1, dtype=torch.bool, device=placed.device)
            for local, record in enumerate(chunk):
                n = int(lengths[local])
                seq = [int(packed['row'][local, int(placed[local, p])]) for p in range(n)]
                ok = seq[0] == 0 and all(b-a in (0, 1) for a, b in zip(seq, seq[1:]))
                express += ok
                if not ok:
                    reasons['decreases' if any(b < a for a, b in zip(seq, seq[1:])) else 'starts_above_zero'] += 1
                for p in range(n-1):
                    cuts[local, p] = seq[p] != seq[p+1]
            row, _ = rows_from_boundaries(placed, lengths, cuts)
            valid = packed['valid']
            ceiling_hit += int(((row == packed['row']) & valid).sum())
            ceiling_n += int(valid.sum())
    return dict(expressible=express, scenes=len(records), reasons=dict(reasons),
                gt_cut_row_accuracy=ceiling_hit/ceiling_n)


def passed(teacher, free, oracle, single, kept, baseline_ids):
    return (free['row_accuracy'] >= .74 and free['task'] >= 74
            and free['by_count']['4']['success'] > 20 and free['by_count']['6']['success'] > 12
            and free['by_count']['8']['success'] > 11 and len(baseline_ids & kept) >= 60
            and teacher['task'] >= 95 and oracle >= 60 and single >= 62)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row boundary result already exists')
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
    groups = {count: sorted((index for index, record in enumerate(records)
                             if len(record['source']['objects']) == count), key=lambda index: records[index]['sample_id'])
              for count in (2, 4, 6, 8)}
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
    report = diagnosis(model, head, val_stored, multi_val)
    print('DIAGNOSIS', json.dumps(report), flush=True)
    if report['expressible'] != 143 or report['scenes']-report['expressible'] != 9:
        raise RuntimeError('GT rows are not contiguous on the predicted order for 143 scenes')
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    baseline_ids = successes(model, head, val_stored, multi_val, 'classifier')
    print('RELOAD', json.dumps(dict(oracle=oracle, single=single, teacher=teacher['task'],
                                   preserved_pool=len(baseline_ids))), flush=True)
    if (teacher['task'], oracle, single, len(baseline_ids)) != (95, 60, 62, 66):
        raise RuntimeError('adopted checkpoint did not reproduce before boundary training')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('boundary.'))
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not name.startswith('boundary.')}
    optimizer = torch.optim.AdamW(model.boundary.parameters(), lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    cycles = [Cycle(groups[count], generator) for count in (2, 4, 6, 8)]
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    for name, parameter in copy.named_parameters():
        parameter.requires_grad_(name.startswith('boundary.'))
    copy_optimizer = torch.optim.AdamW(copy.boundary.parameters(), lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    probe = collate_rows([stored[index] for index in cycles[0].next(1)], stable.DEVICE)
    hidden = hidden_of(head, probe['features'])
    with torch.no_grad():
        original = model(hidden, probe['valid'], probe['goal'], probe['size'][..., 0], probe['size'][..., 1],
                         probe['row'], probe['slot'], False)
        restored = copy(hidden, probe['valid'], probe['goal'], probe['size'][..., 0], probe['size'][..., 1],
                        probe['row'], probe['slot'], False)
    if not (torch.equal(original['boundary_logits'], restored['boundary_logits']) and loaded['update'] == 0):
        raise RuntimeError('boundary checkpoint reload changed the cut logits')
    print('CHECKPOINT', json.dumps(dict(ok=True)), flush=True)
    del copy, copy_optimizer, original, restored, hidden
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()

    def record_eval(update, train_loss):
        current_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
        current_free = evaluate(model, head, val_stored, multi_val, 'free')
        current_oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
        current_single = score(head, single_stored, single_val)['task']
        cuts = boundary_accuracy(model, head, val_stored, multi_val)
        point = dict(update=update, loss=train_loss, oracle=current_oracle, single=current_single,
                     teacher=compact(current_teacher), free=compact(current_free), boundary_accuracy=cuts,
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(
            update=update, oracle=current_oracle, single=current_single, teacher=current_teacher['task'],
            free=current_free['task'], row=current_free['row_accuracy'], boundary=cuts,
            by={key: value['success'] for key, value in current_free['by_count'].items()},
            loss=train_loss)), flush=True)
        return point

    best = None
    save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, 0, CONFIG)
    reason = 'max_updates'
    stale = 0
    running = 0.
    seen = 0
    for update in range(1, MAX_UPDATES+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        packed = collate_rows([stored[index] for cycle in cycles for index in cycle.next(PER_CLASS)], stable.DEVICE)
        hidden = hidden_of(head, packed['features'])
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                       packed['row'], packed['slot'], False)
        loss = boundary_loss(output, packed['row'])
        if not torch.isfinite(loss):
            raise RuntimeError('boundary loss diverged')
        loss.backward()
        if update == 1:
            leaked = [name for name, parameter in model.named_parameters()
                      if not name.startswith('boundary.') and parameter.grad is not None]
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            grad = model.boundary[-1].weight.grad
            norm = 0. if grad is None else float(grad.detach().pow(2).sum().sqrt())
            print('PATH', json.dumps(dict(boundary_grad=norm, leaked=leaked, head_grad=head_grad,
                                         loss=float(loss.detach()))), flush=True)
            if norm == 0 or leaked or head_grad:
                raise RuntimeError('boundary gradient left the new head')
        torch.nn.utils.clip_grad_norm_(model.boundary.parameters(), CONFIG['coordinate_clip'])
        optimizer.step()
        if any(not torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
               if not name.startswith('boundary.')):
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
        if best is None:
            held_rank = None
        else:
            held_rank = (best['free']['task'], best['free']['row_accuracy'],
                         best['free']['by_count']['6']['success'], best['free']['by_count']['8']['success'])
        if held_rank is None or rank > held_rank:
            best = point
            stale = 0
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    if best is None:
        raise RuntimeError('boundary training produced no validation point')
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    for name, parameter in reloaded.named_parameters():
        parameter.requires_grad_(name.startswith('boundary.'))
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    val_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    val_free = evaluate(model, head, val_stored, multi_val, 'free')
    cuts = boundary_accuracy(model, head, val_stored, multi_val)
    kept = successes(model, head, val_stored, multi_val, 'boundary')
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    status = 'PASS' if passed(val_teacher, val_free, oracle_end, single_end, kept, baseline_ids) else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, best_update=best_state['update'], best_checkpoint=str(ROOT/'checkpoint-best.pt'),
        reload_matches_logged_success=val_free['task'] == best['free']['task'] and val_free['row_accuracy'] == best['free']['row_accuracy'],
        preserved_success=len(baseline_ids & kept)/len(baseline_ids), preserved_count=len(baseline_ids & kept),
        config=CONFIG, diagnosis=report, boundary_accuracy=cuts,
        validation_teacher=compact(val_teacher), validation_free=compact(val_free),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters()),
        coordinates_unchanged=all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                                  if not name.startswith('boundary.')),
        modular_checkpoint_unchanged=stamp(MODULAR) == held,
        adopted_checkpoint_unchanged=stamp(BEST) == best_held,
        full_9600=False, joint=False, test_used=False, geometry_solver=False))
    print('DONE', json.dumps(dict(
        status=status, stop=reason, update=best_state['update'], boundary=cuts,
        row=val_free['row_accuracy'], free=val_free['task'], teacher=val_teacher['task'],
        by={key: value['success'] for key, value in val_free['by_count'].items()},
        preserved=len(baseline_ids & kept), oracle=oracle_end, single=single_end,
        gt_expressible=report['expressible'], gt_cut_row_accuracy=report['gt_cut_row_accuracy'])), flush=True)


if __name__ == '__main__':
    main()
