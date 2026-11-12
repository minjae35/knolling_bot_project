"""Frozen Origin/Δx on the teacher partition. No weights are updated."""
import json
from collections import Counter

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from counterfactual_color_pose import legal_row_set
from knolling_dataset.schema import instruction_from_dict, scene_from_dict
from knolling_dataset.teacher_balanced import ordered_objects
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import accepted_success
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows, hidden_of
from score_end_to_end import load_adopted
from train_predicted_pose import teacher_pose
from train_set_context import fresh_split


def parsed_rows(row, slot, ids):
    groups = {}
    for object_id, row_id, slot_id in zip(ids, row, slot):
        groups.setdefault(int(row_id), []).append((int(slot_id), object_id))
    contiguous = bool(groups) and sorted(groups) == list(range(len(groups)))
    rows = [[object_id for _, object_id in sorted(groups[key])] for key in sorted(groups)]
    return contiguous, rows


def illegal_kind(record, rows, contiguous):
    legal_rows, legal_sets = legal_row_set(record)
    key = tuple(tuple(row) for row in rows)
    if not contiguous and key in legal_rows:
        return 'gapped_index'
    if contiguous and key in legal_rows:
        return 'template_no_fit'
    if tuple(frozenset(row) for row in rows) in legal_sets:
        return 'order_only'
    scene = scene_from_dict(record['source'])
    instruction = instruction_from_dict(record['instruction'])
    if instruction.group_by == 'color':
        return 'mixed_color'
    canonical = [obj.object_id for obj in ordered_objects(scene, instruction)]
    flat = [object_id for row in rows for object_id in row]
    if flat != canonical:
        return 'order_of_members'
    return 'wrong_cut'


def score_chunk(model, relation, stored, records, forced):
    packed = collate_rows(stored, stable.DEVICE)
    hidden = hidden_of(relation, packed['features'])
    widths, heights = packed['size'][..., 0], packed['size'][..., 1]
    row = packed['row'] if forced else None
    slot = packed['slot'] if forced else None
    output = model(hidden, packed['valid'], packed['goal'], widths, heights, row, slot, True)
    scenes = []
    for local, record in enumerate(records):
        kept = stored[local]['valid'].shape[0]
        item = {key: packed[key][local:local+1, :kept].cpu() for key in
                ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
        objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                          item['valid'], item['query_index'])
        layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                      angle=output['angle'][local:local+1, :kept].cpu(),
                      valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
        ids, truth = recovery_supervision(record)
        judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [dict(
            sample_id=record['sample_id'], object_ids=ids, record=record)]),
            semantic_check=semantic_errors_v3)[0]
        valid = packed['valid'][local, :kept].cpu()
        scenes.append(dict(
            task=bool(accepted_success(record, judged['task_success'])),
            errors=list(judged['errors']),
            row=[int(value) for value in output['row'][local, :kept][valid].cpu()],
            slot=[int(value) for value in output['slot'][local, :kept][valid].cpu()],
            xy=output['xy'][local, :kept][valid].cpu(), ids=ids))
    return scenes


def main():
    runner.setup()
    train = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    _fit, dev, _blocked = fresh_split(train.records)
    system, relation, model, _head, _saved = load_adopted()
    records = [train.records[index] for index in dev]
    stored = attach_rows(records, cache_records(system, records))
    free, forced = [], []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk_s = stored[start:start+16]
            chunk_r = records[start:start+16]
            free.extend(score_chunk(model, relation, chunk_s, chunk_r, False))
            forced.extend(score_chunk(model, relation, chunk_s, chunk_r, True))
    kinds = Counter()
    fixed = Counter()
    forced_errors = Counter()
    gained = lost = preserved = 0
    for record, before, after in zip(records, free, forced):
        if before['task'] and after['task']:
            preserved += 1
        elif before['task'] and not after['task']:
            lost += 1
        elif after['task'] and not before['task']:
            gained += 1
        contiguous, rows = parsed_rows(before['row'], before['slot'], before['ids'])
        exact = contiguous and rows == record['supervision_v3']['rows']
        if before['task']:
            kind = 'success'
        elif exact:
            kind = 'exact_teacher_fail'
        else:
            posed = teacher_pose(record, rows) if contiguous else None
            kind = 'other_legal' if posed is not None else illegal_kind(record, rows, contiguous)
        kinds[kind] += 1
        if kind in ('gapped_index', 'order_only', 'mixed_color', 'order_of_members', 'wrong_cut', 'template_no_fit'):
            if after['task']:
                fixed[kind] += 1
            else:
                for name in after['errors']:
                    forced_errors[name] += 1
        illegal = sum(kinds[name] for name in (
            'gapped_index', 'order_only', 'mixed_color', 'order_of_members', 'wrong_cut', 'template_no_fit'))
    print(json.dumps(dict(
        free_task=sum(scene['task'] for scene in free),
        forced_task=sum(scene['task'] for scene in forced),
        preserved=preserved, lost=lost, gained=gained,
        kinds=dict(kinds), illegal=illegal, fixed_by_teacher_structure=dict(fixed),
        remaining_errors=dict(forced_errors)), indent=2))


if __name__ == '__main__':
    main()
