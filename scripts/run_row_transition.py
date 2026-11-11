"""One pilot that teaches row breaks without replacing the modular best.

The run starts with eight multi-row scenes. It continues to the filtered
2/4/6/8 validation pool only if those scenes show a real placement, and it
stops if Phase 1 oracle or single-row validation falls through the floor.
"""
import json
import math
import time
from pathlib import Path

import torch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import collate, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.relation_pose import relation_loss, transition_loss
from knolling_model.recovery.data import recovery_supervision
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import changed, install_live_text, module_state

ROOT = Path('artifacts/row_transition')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
PEAK = 1e-4
FLOOR = 1e-5
WARMUP = 100
PROBE = 1000
PROBE_EXTEND = 1500
TOTAL = 3000
ORACLE_FLOOR = 58
SINGLE_FLOOR = 56


def rate(step):
    if step <= WARMUP:
        return FLOOR+(PEAK-FLOOR)*step/WARMUP
    progress = min(1., (step-WARMUP)/max(1, TOTAL-WARMUP))
    return FLOOR+0.5*(PEAK-FLOOR)*(1+math.cos(math.pi*progress))


def attach_breaks(records, stored):
    for record, item in zip(records, stored):
        ids, _ = recovery_supervision(record)
        order = item['order'].tolist()
        vertical = {(upper[-1], lower[0]) for upper, lower in record['supervision_v3']['vertical_neighbors']}
        labels = torch.zeros(len(order))
        for step in range(1, len(order)):
            if (ids[order[step-1]], ids[order[step]]) in vertical:
                labels[order[step]] = 1.
        item['break'] = labels
    return stored


def choose(records):
    picked = []
    for count in (2, 4, 6, 8):
        group = sorted((record for record in records if len(record['source']['objects']) == count),
                       key=lambda record: record['sample_id'])
        picked.extend(group[:2])
    if len(picked) != 8:
        raise RuntimeError('the eight-scene probe is incomplete')
    return picked


def sample(pool, size, generator):
    order = torch.randperm(len(pool), generator=generator).tolist()
    return [pool[index] for index in order[:size]]


def break_error(head, stored, records):
    head.eval()
    within, breaks, hits, total = [], [], 0, 0
    with torch.no_grad():
        for start in range(0, len(records), 16):
            indices = list(range(start, min(start+16, len(records))))
            packed = collate([stored[index] for index in indices], stable.DEVICE)
            output = head(packed['features'], packed['valid'], packed['goal'], packed['order'], packed['break'])
            rows = torch.arange(len(indices), device=stable.DEVICE)[:, None]
            current = packed['order'][:, 1:]
            active = current >= 0
            predicted = output['transition_logits'][rows, current.clamp_min(0)] > 0
            truth = packed['break'][rows, current.clamp_min(0)] > .5
            hits += int((predicted == truth)[active].sum())
            total += int(active.sum())
            delta = output['deltas']
            target = packed['target_xy']
            order = packed['order']
            for local in range(len(indices)):
                for step in range(1, order.shape[1]):
                    slot = int(order[local, step])
                    previous = int(order[local, step-1])
                    if slot < 0 or previous < 0:
                        continue
                    error = torch.linalg.vector_norm(delta[local, slot]-(target[local, slot]-target[local, previous]))
                    (breaks if packed['break'][local, slot] > .5 else within).append(float(error)*1000)
    def median(values):
        if not values:
            return None
        return float(torch.tensor(values).median())
    return dict(within_mm=median(within), break_mm=median(breaks), transition_accuracy=None if not total else hits/total,
                breaks=len(breaks))


def main():
    held = stamp(MODULAR)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    phase_train = subset(train_data, 'train', 791)
    phase_val = subset(val_data, 'validation', 94)
    multi_train = [record for record in train_data.records if multi_row(record)]
    multi_val = [record for record in val_data.records if multi_row(record)]
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    probe_records = choose(multi_train)
    if len(single_val) != 104 or len(multi_val) != 152:
        raise RuntimeError(f'unexpected pools single={len(single_val)} multi={len(multi_val)}')
    model = load_integrated(device=stable.DEVICE)
    install_live_text(model)
    model.eval()
    for module in (model.perception, model.arrangement.text_encoder, model.layout.goal_encoder):
        module.requires_grad_(False)
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    head.enable_row_transition()
    phase_cache = stable.cache_oracle(model, phase_val)
    start_oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    print('CACHE', len(phase_train.records), len(multi_train), len(single_val), len(multi_val), flush=True)
    phase_stored = attach_breaks(phase_train.records, cache_records(model, phase_train.records))
    multi_stored = attach_breaks(multi_train, cache_records(model, multi_train))
    single_stored = attach_breaks(single_val, cache_records(model, single_val))
    multi_val_stored = attach_breaks(multi_val, cache_records(model, multi_val))
    probe_ids = {record['sample_id'] for record in probe_records}
    probe_index = [index for index, record in enumerate(multi_train) if record['sample_id'] in probe_ids]
    probe_stored = [multi_stored[index] for index in probe_index]
    probe_rows = [multi_train[index] for index in probe_index]
    start_single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=start_oracle, single=start_single)), flush=True)
    if (start_oracle, start_single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization',
                                              oracle=start_oracle, single=start_single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    frozen = dict(perception=module_state(model.perception), language=module_state(model.arrangement.text_encoder),
                  goal=module_state(model.layout.goal_encoder))
    optimizer = torch.optim.AdamW(head.parameters(), lr=PEAK, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    history = (ROOT/'history.jsonl').open('w')
    best = None
    reason = 'completed'
    pilot = False
    started = time.perf_counter()
    phase_pool = list(range(len(phase_stored)))
    multi_pool = list(range(len(multi_stored)))

    def evaluate(step):
        head.eval()
        oracle = stable.official_metrics(head, phase_cache, phase_val)
        single = score(head, single_stored, single_val)
        multi = score(head, multi_val_stored, multi_val)
        trained = score(head, probe_stored, probe_rows)
        learned = break_error(head, probe_stored, probe_rows)
        point = dict(update=step, oracle=oracle['task'], single=single['task'],
                     multi=multi['task'], multi_total=multi['total'],
                     by_count={key: value['success'] for key, value in multi['by_count'].items()},
                     gap=multi['horizontal_gap_mm'], vertical_gap=multi['vertical_gap_mm'],
                     collision=multi['collision'], boundary=multi['boundary'],
                     row_match=multi['row_match'], order=multi['order'],
                     train8=trained['task'], train8_by_count={key: value['success'] for key, value in trained['by_count'].items()},
                     train_break_mm=learned['break_mm'], train_transition=learned['transition_accuracy'],
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(update=step, oracle=point['oracle'], single=point['single'],
                                     multi=point['multi'], by_count=point['by_count'],
                                     train8=point['train8'], break_mm=point['train_break_mm'])), flush=True)
        return point

    baseline = evaluate(0)
    for step in range(1, TOTAL+1):
        phase_choice = sample(phase_pool, 8, generator)
        if pilot:
            multi_items = [multi_stored[index] for index in sample(multi_pool, 8, generator)]
        else:
            multi_items = [probe_stored[index] for index in sample(list(range(8)), 8, generator)]
        packed = collate([phase_stored[index] for index in phase_choice]+multi_items, stable.DEVICE)
        head.train()
        output = head(packed['features'], packed['valid'], packed['goal'], packed['order'], packed['break'])
        terms = relation_loss(output, packed['target_xy'], packed['target_angle'], packed['order'])
        loss = terms['loss']+transition_loss(output, packed['break'], packed['order'])
        if not torch.isfinite(loss):
            reason = 'nonfinite_loss'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            column = float(head.relation[0].weight.grad[:, -1].abs().sum())
            decision = float(head.transition[-1].weight.grad.abs().sum())
            print('PATH', json.dumps(dict(break_column=column, transition=decision)), flush=True)
            if column == 0 or decision == 0:
                reason = 'transition_path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        for group in optimizer.param_groups:
            group['lr'] = rate(step)
        optimizer.step()
        if step % 250 != 0 and step not in (PROBE, PROBE_EXTEND, TOTAL):
            continue
        point = evaluate(step)
        if point['oracle'] < ORACLE_FLOOR or point['single'] < SINGLE_FLOOR:
            reason = 'regression'
            break
        rank = (point['by_count'].get('6', 0)+point['by_count'].get('8', 0), point['multi'],
                point['by_count'].get('8', 0), -(point['train_break_mm'] or 1e9))
        if rank[0] > 0 and (best is None or rank > best['rank']):
            best = dict(rank=rank, update=step, metrics=point,
                        state={key: value.detach().cpu() for key, value in head.state_dict().items()})
            torch.save(dict(update=step, model=best['state'], source='modular_best', row_transition=True),
                       ROOT/'checkpoint-best.pt')
        if pilot or step < PROBE:
            continue
        learned = point['train8'] >= 4 and point['train8_by_count'].get('6', 0)+point['train8_by_count'].get('8', 0) >= 1
        moving = point['train_break_mm'] is not None and point['train_break_mm'] < .5*baseline['train_break_mm']
        if learned:
            pilot = True
            print('PILOT', step, flush=True)
        elif step >= PROBE_EXTEND or not moving:
            reason = 'break_not_learned'
            break
    history.close()
    movement = dict(perception=changed(frozen['perception'], model.perception),
                    language=changed(frozen['language'], model.arrangement.text_encoder),
                    goal=changed(frozen['goal'], model.layout.goal_encoder))
    status = 'STOP' if reason == 'transition_path_blocked' else 'PASS' if best is not None else 'FAIL'
    report = dict(status=status, stop=reason, pilot=pilot, best_update=None if best is None else best['update'],
                  baseline=baseline, best=None if best is None else best['metrics'],
                  frozen_movement=movement, modular_checkpoint_unchanged=stamp(MODULAR) == held,
                  test_used=False, adopted=status == 'PASS', full_9600=False, joint=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=status, stop=reason, pilot=pilot,
                                 best=None if best is None else best['update'])), flush=True)


if __name__ == '__main__':
    main()
