"""Read-only split of validation failures into frozen and trainable causes.

Teacher row and slot are used only as a diagnostic. They are not model inputs
in the reported free-rollout result.
"""
import json

import torch

from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, poses_from_detection
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from score_end_to_end import (
    JOINT_BEST, SIZE_BEST, labels_of, load_bundle, validation_indices)

DATA = 'artifacts/generalization_20000/dataset'


def judge(system, visual, output, objects, record, identities, truth):
    source = {key: value.detach().cpu() for key, value in visual['source'].items()}
    layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
    meta = dict(sample_id=record['sample_id'], object_ids=identities, record=record)
    judged = evaluate_batch(view, Batch({}, [truth], [meta]), semantic_check=semantic_errors_v3)[0]
    return accepted_success(record, judged['task_success']), list(judged['errors']), bool(judged['count_correct'])


def correspond(output, objects, visual, truth, identities, record, threshold):
    row, slot = labels_of(record, identities)
    chosen = select_queries(visual['source'], threshold)[0]
    distance = torch.cdist(objects.xy[0, :len(chosen)], truth['source_xy'].to(objects.xy.device))
    used_query, used_gt, pairs = set(), set(), {}
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            continue
        used_query.add(query)
        used_gt.add(gt)
        pairs[query] = gt
    teacher_row = torch.full((1, len(chosen)), -1, device=output['row'].device)
    teacher_slot = torch.full((1, len(chosen)), -1, device=output['row'].device)
    row_hits = slot_hits = 0
    for query, gt in pairs.items():
        teacher_row[0, query] = int(row[gt])
        teacher_slot[0, query] = int(slot[gt])
        row_hits += int(int(output['row'][0, query]) == int(row[gt]))
        slot_hits += int(int(output['slot'][0, query]) == int(slot[gt]))
    exact = len(pairs) == len(chosen) == len(identities)
    return dict(row_exact=exact and row_hits == len(chosen), slot_exact=exact and slot_hits == len(chosen),
                labeled=len(pairs), teacher_row=teacher_row, teacher_slot=teacher_slot)


def guided(system, relation, row_model, objects, texts, teacher_row, teacher_slot):
    text, mask = encode_instruction(system, texts, objects.xy.device)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    return row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                     teacher_row, teacher_slot, False)


def one(system, relation, row_model, head, data, index):
    record = data.records[index]
    identities, truth = recovery_supervision(record)
    truth = {key: value.cpu() for key, value in truth.items()}
    image = data[index]['image'][None].to(stable.DEVICE)
    try:
        output, objects, visual = poses_from_detection(
            system, relation, row_model, image, [record['instruction']['text']], head)
    except ValueError:
        return dict(sample_id=record['sample_id'], objects=len(identities), task=False,
                    errors=['count'], row_exact=False, slot_exact=False, guided_task=False,
                    guided_errors=['count'])
    task, errors, count_ok = judge(system, visual, output, objects, record, identities, truth)
    info = correspond(output, objects, visual, truth, identities, record, system.threshold)
    guided_task, guided_errors = False, ['unmatched']
    if count_ok and info['labeled'] == len(identities):
        output_g = guided(system, relation, row_model, objects, [record['instruction']['text']],
                          info['teacher_row'], info['teacher_slot'])
        guided_task, guided_errors, _ = judge(system, visual, output_g, objects, record, identities, truth)
    return dict(sample_id=record['sample_id'], objects=len(identities), task=task, errors=errors,
                row_exact=info['row_exact'], slot_exact=info['slot_exact'],
                guided_task=guided_task, guided_errors=guided_errors)


def cause(scene):
    if 'count' in scene['errors'] or 'source_identity' in scene['errors']:
        return 'detection'
    if not scene['row_exact']:
        return 'row_assignment'
    if not scene['slot_exact']:
        return 'ordering'
    if 'axis_alignment' in scene['errors']:
        return 'angle'
    vertical = 'vertical_gap' in scene['errors'] or 'row_alignment' in scene['errors']
    horizontal = 'horizontal_gap' in scene['errors']
    if vertical and not horizontal:
        return 'origin_y'
    if horizontal and not vertical:
        return 'delta_x'
    if vertical and horizontal:
        return 'origin_and_delta_x'
    if 'size_order' in scene['errors']:
        return 'size_order'
    return 'other:' + ','.join(scene['errors'])


def main():
    runner.setup()
    val = ExpandedDataset(DATA, 'validation')
    train = ExpandedDataset(DATA, 'train')
    indices = validation_indices(val)
    full = [record for record in train.records if is_full(record)]
    families = {}
    for record in full:
        families.setdefault(record['scene_family_id'], set()).add(record['split'])
    old_system, old_relation, old_row, old_head, _ = load_bundle(SIZE_BEST)
    old = [one(old_system, old_relation, old_row, old_head, val, index) for index in indices]
    del old_system, old_relation, old_row, old_head
    torch.cuda.empty_cache()
    system, relation, row_model, head, _ = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    new = [one(system, relation, row_model, head, val, index) for index in indices]
    old_ok = {scene['sample_id'] for scene in old if scene['task']}
    new_ok = {scene['sample_id'] for scene in new if scene['task']}
    lost = [scene for scene in new if scene['sample_id'] in old_ok and scene['sample_id'] not in new_ok]
    failed_eight = [scene for scene in new if scene['objects'] == 8 and not scene['task']]
    sample = failed_eight[:20]

    def tally(scenes, key):
        counts = {}
        for scene in scenes:
            for name in scene[key]:
                counts[name] = counts.get(name, 0)+1
        return counts

    def summarize(scenes):
        counts = {}
        for scene in scenes:
            name = cause(scene)
            counts[name] = counts.get(name, 0)+1
        guided_fix = sum(scene['guided_task'] for scene in scenes)
        vertical = [scene for scene in scenes if 'vertical_gap' in scene['errors'] or 'row_alignment' in scene['errors']]
        still = [scene for scene in scenes if not scene['guided_task']]
        return dict(n=len(scenes), causes=counts, guided_success=guided_fix,
                    vertical_or_alignment=len(vertical),
                    vertical_or_alignment_removed_by_teacher_structure=sum(
                        'vertical_gap' not in scene['guided_errors'] and 'row_alignment' not in scene['guided_errors']
                        for scene in vertical),
                    guided_errors_when_still_failing=tally(still, 'guided_errors'),
                    free_errors=tally(scenes, 'errors'))

    print(json.dumps(dict(
        old_success=len(old_ok), new_success=len(new_ok),
        lost=len(lost), lost_summary=summarize(lost),
        failed_eight=len(failed_eight), eight_summary=summarize(failed_eight),
        eight_sample_20=summarize(sample),
        train_full_records=len(full), train_full_families=len(families),
        family_splits_overlap=sum(len(value) > 1 for value in families.values())), indent=2), flush=True)


if __name__ == '__main__':
    main()
