"""Train only a row-break residual on top of the frozen modular head.

One run. Eight multi-row scenes must be placeable before the filtered
validation pool is used. The relation weights stay at the modular best.
"""
import json
import time
from pathlib import Path

import torch
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, collate, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_residual import FrozenRowModel, RowResidual, residual_loss
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from run_full_arrangement import cache_records, stamp, subset
from run_relation_joint import changed, install_live_text, module_state
from run_row_transition import attach_breaks, choose, sample

ROOT = Path('artifacts/row_residual')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
PROBE = 600
EXTEND = 1000
TOTAL = 2500
ORACLE_FLOOR = 58
SINGLE_FLOOR = 56


def measure(model, stored, records, mode):
    """mode is gt, predicted, or inference. gt and predicted keep the teacher order."""
    model.eval()
    rows = []
    within, breaks, hits, steps, false_breaks = [], [], 0, 0, 0
    with torch.no_grad():
        for start in range(0, len(records), 16):
            indices = list(range(start, min(start+16, len(records))))
            packed = collate([stored[index] for index in indices], stable.DEVICE)
            order = packed['order'] if mode != 'inference' else None
            label = packed['break'] if mode == 'gt' else None
            output = model(packed['features'], packed['valid'], packed['goal'], order, label)
            false_breaks += int(output['break_bit'].sum())
            delta = output['deltas']
            target = packed['target_xy']
            for local, index in enumerate(indices):
                kept = stored[index]['valid'].shape[0]
                ids, _ = recovery_supervision(records[index])
                vertical = {(upper[-1], lower[0]) for upper, lower in records[index]['supervision_v3']['vertical_neighbors']}
                sequence = output['predicted_order'][local].detach().cpu().tolist()
                for step in range(1, kept):
                    slot, previous = sequence[step], sequence[step-1]
                    if slot < 0 or previous < 0:
                        continue
                    steps += 1
                    pair = (ids[previous], ids[slot])
                    truth = pair in vertical
                    hits += int(bool(output['break_bit'][local, slot] > .5) == truth)
                    error = torch.linalg.vector_norm(delta[local, slot]-(target[local, slot]-target[local, previous]))
                    (breaks if truth else within).append(float(error)*1000)
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                              angle=output['angle'][local:local+1, :kept].cpu(),
                              valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                ids, truth = recovery_supervision(records[index])
                meta = dict(sample_id=records[index]['sample_id'], object_ids=ids, record=records[index])
                judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [meta]),
                                        semantic_check=semantic_errors_v3)[0]
                detail = arrangement_errors(records[index], judged['predicted_targets']) if judged.get('predicted_targets') else None
                rows.append(dict(task=bool(judged['task_success']), objects=len(records[index]['source']['objects']),
                                 boundary='boundary' in judged['errors'], collision='collision' in judged['errors'],
                                 row_match=bool(detail and detail['row_match'])))
    def median(values):
        return None if not values else float(torch.tensor(values).median())
    by_count = {str(count): sum(row['task'] for row in rows if row['objects'] == count) for count in (2, 4, 6, 8)}
    return dict(task=sum(row['task'] for row in rows), total=len(rows), by_count=by_count,
                boundary=sum(row['boundary'] for row in rows), collision=sum(row['collision'] for row in rows),
                row_match=sum(row['row_match'] for row in rows), within_mm=median(within), break_mm=median(breaks),
                transition_accuracy=None if not steps else hits/steps, false_breaks=false_breaks)


def brief(point):
    return dict(task=point['task'], by_count=point['by_count'], break_mm=point['break_mm'],
                transition=point['transition_accuracy'], false_breaks=point['false_breaks'])


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row residual result already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
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
    model_body = load_integrated(device=stable.DEVICE)
    install_live_text(model_body)
    model_body.eval()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    head.requires_grad_(False)
    head.eval()
    residual = RowResidual().to(stable.DEVICE)
    model = FrozenRowModel(head, residual)
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(model_body, phase_val)
    start_oracle = stable.official_metrics(model, phase_cache, phase_val)['task']
    print('CACHE', flush=True)
    phase_stored = attach_breaks(phase_train.records, cache_records(model_body, phase_train.records))
    multi_stored = attach_breaks(multi_train, cache_records(model_body, multi_train))
    single_stored = attach_breaks(single_val, cache_records(model_body, single_val))
    multi_val_stored = attach_breaks(multi_val, cache_records(model_body, multi_val))
    probe_ids = {record['sample_id'] for record in probe_records}
    probe_index = [index for index, record in enumerate(multi_train) if record['sample_id'] in probe_ids]
    probe_stored = [multi_stored[index] for index in probe_index]
    probe_rows = [multi_train[index] for index in probe_index]
    start_single = measure(model, single_stored, single_val, 'inference')
    print('START', json.dumps(dict(oracle=start_oracle, single=start_single['task'],
                                 false_breaks=start_single['false_breaks'])), flush=True)
    if (start_oracle, start_single['task'], start_single['false_breaks']) != (60, 62, 0):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=start_oracle,
                                              single=start_single['task'], false_breaks=start_single['false_breaks'],
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    frozen = module_state(head)
    modules = module_state(model_body)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    history = (ROOT/'history.jsonl').open('w')
    best = None
    reason = 'completed'
    pilot = False
    started = time.perf_counter()
    phase_pool = list(range(len(phase_stored)))

    def evaluate(step):
        oracle = stable.official_metrics(model, phase_cache, phase_val)['task']
        single = measure(model, single_stored, single_val, 'inference')
        validation = measure(model, multi_val_stored, multi_val, 'inference')
        gt_validation = measure(model, multi_val_stored, multi_val, 'gt')
        predicted = measure(model, probe_stored, probe_rows, 'predicted')
        granted = measure(model, probe_stored, probe_rows, 'gt')
        point = dict(update=step, oracle=oracle, single=single['task'], false_breaks=single['false_breaks'],
                     multi=validation['task'], by_count=validation['by_count'], break_mm=validation['break_mm'],
                     transition=validation['transition_accuracy'], boundary=validation['boundary'],
                     collision=validation['collision'], row_match=validation['row_match'],
                     gt_multi=gt_validation['task'], gt_by_count=gt_validation['by_count'],
                     gt_break_mm=gt_validation['break_mm'], train_predicted=brief(predicted), train_gt=brief(granted),
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(update=step, oracle=oracle, single=single['task'], false_breaks=single['false_breaks'],
                                     multi=validation['task'], by_count=validation['by_count'],
                                     break_mm=validation['break_mm'], gt_multi=gt_validation['task'],
                                     gt_break_mm=gt_validation['break_mm'],
                                     train_gt=granted['task'], train_predicted=predicted['task'])), flush=True)
        return point

    baseline = evaluate(0)
    for step in range(1, TOTAL+1):
        phase_choice = sample(phase_pool, 8, generator)
        if pilot:
            multi_items = [multi_stored[index] for index in sample(list(range(len(multi_stored))), 8, generator)]
        else:
            multi_items = [probe_stored[index] for index in sample(list(range(8)), 8, generator)]
        packed = collate([phase_stored[index] for index in phase_choice]+multi_items, stable.DEVICE)
        residual.train()
        output = model(packed['features'], packed['valid'], packed['goal'], packed['order'], packed['break'])
        loss = residual_loss(output, packed['target_xy'], packed['order'], packed['break'])['loss']
        if not torch.isfinite(loss):
            reason = 'nonfinite_loss'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            column = float(residual.residual[-1].weight.grad.abs().sum())
            decision = float(residual.classifier[-1].weight.grad.abs().sum())
            print('PATH', json.dumps(dict(residual=column, classifier=decision)), flush=True)
            if column == 0 or decision == 0:
                reason = 'residual_path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.)
        optimizer.step()
        if step % 200 != 0 and step not in (PROBE, EXTEND, TOTAL):
            continue
        point = evaluate(step)
        if point['oracle'] < ORACLE_FLOOR or point['single'] < SINGLE_FLOOR:
            reason = 'regression'
            break
        gained = point['by_count'].get('6', 0)+point['by_count'].get('8', 0)
        if gained > 0 and point['multi'] > 1 and (best is None or gained > best['rank']):
            best = dict(rank=gained, update=step, metrics=point,
                        residual={key: value.detach().cpu() for key, value in residual.state_dict().items()})
            torch.save(dict(update=step, residual=best['residual'], source='modular_best',
                            frozen_relation=True), ROOT/'checkpoint-best.pt')
        if pilot or step < PROBE:
            continue
        gt_ready = point['train_gt']['task'] >= 4 and point['train_gt']['by_count'].get('6', 0)+point['train_gt']['by_count'].get('8', 0) >= 1
        pred_ready = point['train_predicted']['task'] >= 4 and point['train_predicted']['by_count'].get('6', 0)+point['train_predicted']['by_count'].get('8', 0) >= 1
        if gt_ready and pred_ready:
            pilot = True
            print('PILOT', step, flush=True)
        elif step >= EXTEND or (not gt_ready and (point['train_gt']['break_mm'] or 1e9) > .5*baseline['train_gt']['break_mm']):
            reason = 'residual_failed' if not gt_ready else 'classifier_failed'
            break
    history.close()
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    movement = changed(frozen, head)
    body = changed(modules, model_body)
    status = 'STOP' if reason == 'residual_path_blocked' else 'PASS' if best is not None else 'FAIL'
    report = dict(status=status, stop=reason, pilot=pilot, best_update=None if best is None else best['update'],
                  baseline=baseline, best=None if best is None else best['metrics'],
                  relation_parameters_unchanged=unchanged, relation_movement=movement,
                  encoder_movement=body, modular_checkpoint_unchanged=stamp(MODULAR) == held,
                  test_used=False, adopted=status == 'PASS', full_9600=False, joint=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=status, stop=reason, pilot=pilot,
                                 best=None if best is None else best['update'], unchanged=unchanged)), flush=True)


if __name__ == '__main__':
    main()
