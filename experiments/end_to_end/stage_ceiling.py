"""How far teacher row, order, and origin move the adopted arrangement.

Read-only. Teacher values are not inference inputs. Count errors stay failures
because a missing detection has no label to force.
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
from score_end_to_end import JOINT_BEST, labels_of, load_bundle, validation_indices

DATA = 'artifacts/generalization_20000/dataset'


def judge(visual, output, objects, record, identities, truth, xy, angle):
    source = {key: value.detach().cpu() for key, value in visual['source'].items()}
    layout = dict(xy=xy.detach().cpu(), angle=angle.detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
    meta = dict(sample_id=record['sample_id'], object_ids=identities, record=record)
    judged = evaluate_batch(view, Batch({}, [truth], [meta]), semantic_check=semantic_errors_v3)[0]
    return accepted_success(record, judged['task_success'])


def align(output, objects, visual, truth, identities, record, threshold):
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
    target = torch.zeros((1, len(chosen), 2), device=output['xy'].device)
    angle = torch.zeros((1, len(chosen)), device=output['xy'].device)
    if len(pairs) != len(chosen) or len(pairs) != len(identities):
        return None
    for query, gt in pairs.items():
        teacher_row[0, query] = int(row[gt])
        teacher_slot[0, query] = int(slot[gt])
        target[0, query] = truth['target_xy'][gt].to(target.device)
        angle[0, query] = truth['target_angle'][gt].to(angle.device)
    return teacher_row, teacher_slot, target, angle


def contiguous(record):
    identities, truth = recovery_supervision(record)
    rows = record['supervision_v3']['rows']
    if len(rows) < 2:
        return None
    location = {object_id: index for index, object_id in enumerate(identities)}
    sign = 1 if record['instruction']['sort_direction'] == 'asc' else -1
    if record['instruction']['group_by'] == 'color':
        palette = ('red', 'green', 'blue')
        color = {object_id: obj['color'] for obj in record['source']['objects'] for object_id in (obj['object_id'],)}
        key = lambda object_id: (palette.index(color[object_id]), sign*float(
            truth['size'][location[object_id]].prod()))
    else:
        key = lambda object_id: sign*float(truth['size'][location[object_id]].prod())
    ordered = sorted((object_id for row in rows for object_id in row), key=key)
    rank = {object_id: index for index, object_id in enumerate(ordered)}
    blocks = []
    for members in rows:
        found = sorted(rank[object_id] for object_id in members)
        blocks.append(found == list(range(found[0], found[0]+len(found))))
    return all(blocks)


def main():
    runner.setup()
    val = ExpandedDataset(DATA, 'validation')
    indices = validation_indices(val)
    labels = [contiguous(val.records[index]) for index in indices]
    labels = [item for item in labels if item is not None]
    system, relation, row_model, head, _ = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    stages = ('free', 'row', 'row_order', 'row_order_origin', 'pose')
    hits = {name: {'all': 0, 'eight': 0} for name in stages}
    counted = {'all': 0, 'eight': 0, 'matched': 0, 'matched_eight': 0}
    with torch.no_grad():
        for index in indices:
            record = val.records[index]
            identities, truth = recovery_supervision(record)
            truth_cpu = {key: value.cpu() for key, value in truth.items()}
            image = val[index]['image'][None].to(stable.DEVICE)
            text = record['instruction']['text']
            eight = len(identities) == 8
            counted['all'] += 1
            counted['eight'] += int(eight)
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, row_model, image, [text], head)
            except ValueError:
                continue
            matched = align(output, objects, visual, truth, identities, record, system.threshold)
            free = judge(visual, output, objects, record, identities, truth_cpu, output['xy'], output['angle'])
            hits['free']['all'] += int(free)
            hits['free']['eight'] += int(free and eight)
            if matched is None:
                continue
            counted['matched'] += 1
            counted['matched_eight'] += int(eight)
            teacher_row, teacher_slot, target, target_angle = matched
            encoded, mask = encode_instruction(system, [text], objects.xy.device)
            goal = system.layout.goal_encoder(objects, encoded, mask)['vector']
            hidden = relation._hidden(objects.features())
            widths, heights = objects.size[..., 0], objects.size[..., 1]
            row_only = row_model(hidden, objects.valid, goal, widths, heights, teacher_row, None, True)
            ordered = row_model(hidden, objects.valid, goal, widths, heights, teacher_row, teacher_slot, False)
            origins = ordered['origins'].clone()
            for query in range(teacher_slot.shape[1]):
                if int(teacher_slot[0, query]) == 0:
                    origins[0, int(teacher_row[0, query])] = target[0, query]
            placed, _ = row_model._positions(objects.valid, goal, teacher_row, teacher_slot, widths, origins)
            forced = (
                ('row', row_only['xy'], row_only['angle']),
                ('row_order', ordered['xy'], ordered['angle']),
                ('row_order_origin', placed, ordered['angle']),
                ('pose', target, target_angle))
            for name, xy, angle in forced:
                ok = judge(visual, output, objects, record, identities, truth_cpu, xy, angle)
                hits[name]['all'] += int(ok)
                hits[name]['eight'] += int(ok and eight)
    print(json.dumps(dict(counted=counted, success=hits,
                          contiguous_multi_row=sum(labels), multi_row=len(labels),
                          full_records=sum(is_full(record) for record in val.records)), indent=2), flush=True)


if __name__ == '__main__':
    main()
