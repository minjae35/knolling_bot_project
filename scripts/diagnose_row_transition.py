"""Compare teacher order and free rollout on the modular relation head.

No weights are updated. Row breaks come from supervision_v3.rows and
vertical_neighbors. Target poses are used only to score the diagnosis.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from knolling_model.experiments.full_arrangement import arrangement_errors, collate
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.arrangement.data import recovery_supervision
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from run_full_arrangement import cache_records, subset
from run_relation_joint import install_live_text

MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
ROOT = Path('artifacts/row_transition')


def multi_row(record):
    instruction = record['instruction']
    count = len(record['source']['objects'])
    return (record['mode'] == 'full' and not instruction['keep_all'] and not instruction['locked_ids']
            and count in (2, 4, 6, 8) and len(record['supervision_v3']['rows']) >= 2
            and instruction['group_by'] == 'none' and instruction['region'] == 'full'
            and instruction['axis_alignment'] == 'horizontal')


def labels_for(record, stored):
    ids, _ = recovery_supervision(record)
    order = stored['order'].tolist()
    sequence = [ids[index] for index in order]
    vertical = {(upper[-1], lower[0]) for upper, lower in record['supervision_v3']['vertical_neighbors']}
    horizontal = {tuple(pair) for pair in record['supervision_v3']['horizontal_neighbors']}
    kind = []
    for left, right in zip(sequence, sequence[1:]):
        pair = (left, right)
        if pair in vertical:
            kind.append('break')
        elif pair in horizontal:
            kind.append('within')
        else:
            kind.append('other')
    rows = record['supervision_v3']['rows']
    consistent = (kind.count('break') == len(rows)-1 == len(vertical)
                  and kind.count('within') == len(horizontal) and 'other' not in kind)
    return kind, consistent


def step_errors(output, stored, kind):
    order = stored['order'].tolist()
    target = stored['target_xy']
    got = output['deltas'][0].cpu()
    within, breaks = [], []
    for step, name in enumerate(kind, start=1):
        current = order[step]
        previous = order[step-1]
        truth = target[current]-target[previous]
        error = torch.linalg.vector_norm(got[current]-truth).item()*1000
        (breaks if name == 'break' else within).append(error)
    return within, breaks


def poses_of(output, record, kept):
    ids, _ = recovery_supervision(record)
    xy = output['xy'][0, :kept].cpu()
    angle = output['angle'][0, :kept].cpu()
    return {ids[index]: dict(x_m=float(xy[index, 0]), y_m=float(xy[index, 1]), theta_rad=float(angle[index]))
            for index in range(kept)}


def stats(values):
    if not values:
        return dict(median=None, p95=None, n=0)
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)), n=len(values))


def diagnose(head, stored, records):
    head.eval()
    within_teacher, break_teacher = [], []
    within_free, break_free = [], []
    within_matched, break_matched = [], []
    success = Counter()
    order_scenes = 0
    order_steps = 0
    order_steps_total = 0
    consistent = 0
    by_count = {}
    with torch.no_grad():
        for start in range(0, len(records), 16):
            index_list = list(range(start, min(start+16, len(records))))
            packed = collate([stored[index] for index in index_list], stable.DEVICE)
            teacher = head(packed['features'], packed['valid'], packed['goal'], packed['order'])
            free = head(packed['features'], packed['valid'], packed['goal'])
            for local, index in enumerate(index_list):
                record = records[index]
                kind, aligned = labels_for(record, stored[index])
                consistent += int(aligned)
                count = len(record['source']['objects'])
                kept = stored[index]['valid'].shape[0]
                teacher_item = {key: value[local:local+1] for key, value in teacher.items() if torch.is_tensor(value)}
                free_item = {key: value[local:local+1] for key, value in free.items() if torch.is_tensor(value)}
                within, breaks = step_errors(teacher_item, stored[index], kind)
                within_teacher.extend(within)
                break_teacher.extend(breaks)
                predicted = free['predicted_order'][local, :kept].cpu().tolist()
                truth = stored[index]['order'].tolist()
                matched = predicted == truth
                order_scenes += int(matched)
                order_steps += sum(left == right for left, right in zip(predicted, truth))
                order_steps_total += kept
                if matched:
                    within_matched.extend(within)
                    break_matched.extend(breaks)
                    free_within, free_breaks = step_errors(free_item, stored[index], kind)
                    within_free.extend(free_within)
                    break_free.extend(free_breaks)
                bucket = by_count.setdefault(count, Counter())
                for mode, output in (('teacher', teacher_item), ('free', free_item)):
                    report = arrangement_errors(record, poses_of(output, record, kept))
                    success[(mode, 'task')] += int(report['task'])
                    success[(mode, 'rows')] += int(report['row_match'])
                    bucket[(mode, 'task')] += int(report['task'])
                    bucket[(mode, 'boundary')] += int(report['boundary'])
                    bucket[(mode, 'collision')] += int(report['collision'])
                bucket['total'] += 1
    return dict(scenes=len(records), supervision_consistent=consistent,
                order_scenes=order_scenes, order_steps=order_steps, order_steps_total=order_steps_total,
                teacher_success=success[('teacher', 'task')], teacher_rows=success[('teacher', 'rows')],
                free_success=success[('free', 'task')], free_rows=success[('free', 'rows')],
                within_teacher_mm=stats(within_teacher), break_teacher_mm=stats(break_teacher),
                within_free_on_correct_order_mm=stats(within_free),
                break_free_on_correct_order_mm=stats(break_free),
                within_matched_teacher_mm=stats(within_matched), break_matched_teacher_mm=stats(break_matched),
                by_count={str(count): {f'{key[0]}_{key[1]}' if isinstance(key, tuple) else key: int(value)
                                     for key, value in rows.items()} for count, rows in by_count.items()})


def main():
    runner.setup()
    validation = ExpandedDataset(DATA, 'validation')
    chosen = [record for record in validation.records if multi_row(record)]
    phase = subset(validation, 'validation', 94)
    model = load_integrated(device=stable.DEVICE)
    install_live_text(model)
    model.eval()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    head.eval()
    report = diagnose(head, cache_records(model, chosen), chosen)
    control = diagnose(head, cache_records(model, phase.records), phase.records)
    report['phase1_control'] = dict(scenes=control['scenes'], teacher_success=control['teacher_success'],
                                    free_success=control['free_success'],
                                    within_teacher_mm=control['within_teacher_mm'],
                                    break_teacher_mm=control['break_teacher_mm'])
    ROOT.mkdir(parents=True, exist_ok=True)
    stable.write(ROOT/'diagnosis.json', report)
    print('DIAGNOSIS', json.dumps(dict(
        scenes=report['scenes'], consistent=report['supervision_consistent'],
        order=report['order_scenes'], teacher=report['teacher_success'], free=report['free_success'],
        within=report['within_teacher_mm'], row_break=report['break_teacher_mm'],
        phase1_within=control['within_teacher_mm'], phase1_teacher=control['teacher_success'])), flush=True)


if __name__ == '__main__':
    main()
