"""Counterfactual: legal coordinates for color-split rows, no weight updates.

The color-loss run kept only update 0, so the rows are the adopted selector's
rows with mixed colors split apart. Teacher poses are diagnostic only.
"""
import json
import math
from collections import Counter

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_dataset.schema import COLORS, Pose, instruction_from_dict, scene_from_dict
from knolling_dataset.teacher_balanced import templates
from knolling_dataset.validator import polygon
from knolling_model.coordinates import angle_delta, to_meters
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import poses_from_detection
from knolling_model.losses import match_source
from knolling_model.recovery.selection import select_queries
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from score_end_to_end import load_adopted
from train_predicted_pose import teacher_pose
from train_set_context import fresh_split


def legal_row_set(record):
    scene = scene_from_dict(record['source'])
    instruction = instruction_from_dict(record['instruction'])
    rows, sets = [], []
    for template in templates(scene, instruction):
        group = tuple(tuple(obj.object_id for obj in row) for row in template['rows'])
        rows.append(group)
        sets.append(tuple(frozenset(row) for row in group))
    return set(rows), set(sets)


def split_colors(rows, color):
    repaired, changed = [], False
    for row in rows:
        buckets, order = {}, []
        for object_id in row:
            name = color[object_id]
            if name not in buckets:
                order.append(name)
                buckets[name] = []
            buckets[name].append(object_id)
        if len(order) == 1:
            repaired.append(list(row))
        else:
            changed = True
            repaired.extend(buckets[name] for name in order)
    return repaired, changed


def extents(obj, theta):
    cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
    return (obj.asset.a_m*cosine+obj.asset.b_m*sine, obj.asset.a_m*sine+obj.asset.b_m*cosine)


def hybrid(record, repaired, original, neural, objects):
    instruction = record['instruction']
    theta = 0. if instruction['axis_alignment'] == 'horizontal' else -math.pi/2
    gap = instruction['gap_m']
    by_id = {obj.object_id: obj for obj in objects}
    original_keys = {tuple(row) for row in original}
    unchanged = [index for index, row in enumerate(repaired) if tuple(row) in original_keys]

    def height(row):
        return max(extents(by_id[object_id], theta)[1] for object_id in row)

    def opening(index):
        left, right = repaired[index], repaired[index+1]
        multiplier = 2 if instruction['group_by'] == 'color' and by_id[left[0]].color != by_id[right[0]].color else 1
        return gap*multiplier

    poses = {object_id: neural[object_id] for row in repaired for object_id in row}
    if not unchanged:
        return None, set()
    center = sum(neural[object_id][0] for index in unchanged for object_id in repaired[index]) / sum(
        len(repaired[index]) for index in unchanged)
    y = [None]*len(repaired)
    for index in unchanged:
        y[index] = sum(neural[object_id][1] for object_id in repaired[index]) / len(repaired[index])
    for index, row in enumerate(repaired):
        if y[index] is not None:
            continue
        previous = next((item for item in range(index-1, -1, -1) if y[item] is not None), None)
        following = next((item for item in range(index+1, len(repaired)) if y[item] is not None), None)
        if previous is not None:
            cursor = y[previous]
            for item in range(previous, index):
                cursor -= height(repaired[item])/2+opening(item)+height(repaired[item+1])/2
            y[index] = cursor
        else:
            cursor = y[following]
            for item in range(following-1, index-1, -1):
                cursor += height(repaired[item+1])/2+opening(item)+height(repaired[item])/2
            y[index] = cursor
    affected = set()
    for index in range(len(repaired)-1):
        deficit = opening(index)-(y[index]-height(repaired[index])/2-y[index+1]-height(repaired[index+1])/2)
        if deficit <= .002:
            continue
        for item in range(index+1, len(repaired)):
            y[item] -= deficit
            if tuple(repaired[item]) in original_keys:
                affected.add(item)
    for index, row in enumerate(repaired):
        if tuple(row) in original_keys and index not in affected:
            continue
        if tuple(row) in original_keys:
            delta = y[index]-sum(neural[object_id][1] for object_id in row)/len(row)
            for object_id in row:
                x_m, y_m, angle = neural[object_id]
                poses[object_id] = (x_m, y_m+delta, angle)
            continue
        widths = [extents(by_id[object_id], theta)[0] for object_id in row]
        left = center-(sum(widths)+gap*(len(row)-1))/2
        for object_id, width in zip(row, widths):
            poses[object_id] = (left+width/2, y[index], theta)
            left += width+gap
    return poses, affected


def judge_poses(record, poses):
    detail = arrangement_errors(record, {key: dict(x_m=value[0], y_m=value[1], theta_rad=value[2])
                                         for key, value in poses.items()})
    return detail


def main():
    runner.setup()
    train = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    _fit, dev, _blocked = fresh_split(train.records)
    system, relation, model, head, _saved = load_adopted()
    counts = Counter()
    gained_errors = Counter()
    base_fail = Counter()
    official = 0
    for index in dev:
        record = train.records[index]
        image = train[index]['image'][None].to(stable.DEVICE)
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
        except ValueError:
            counts['detection'] += 1
            continue
        source = {key: value.detach().cpu() for key, value in visual['source'].items()}
        layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                      valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                      rollout='free_rollout')
        view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
        judged = evaluate_batch(view, Batch({}, [truth], [dict(
            sample_id=record['sample_id'], object_ids=identities, record=record)]),
            semantic_check=semantic_errors_v3)[0]
        success = accepted_success(record, judged['task_success'])
        official += success
        if success:
            counts['baseline_success'] += 1
            continue
        for name in judged['errors']:
            base_fail[name] += 1
        if 'count' in judged['errors'] or 'source_identity' in judged['errors'] or not judged.get('predicted_targets'):
            counts['detection'] += 1
            continue
        active = torch.where(source['objectness_logits'][0].sigmoid() >= .5)[0]
        chosen = select_queries(visual['source'], system.threshold)[0]
        packed = {int(raw): index for index, raw in enumerate(chosen.tolist())}
        limited = {key: value[:, active] for key, value in source.items()}
        queries, gts = match_source(limited, [truth])[0]
        queries = active[queries]
        pairs = []
        scene = scene_from_dict(record['source'])
        catalog = {obj.object_id: obj for obj in scene.objects}
        color = {}
        for query, gt in zip(queries.tolist(), gts.tolist()):
            if int(query) not in packed:
                pairs = None
                break
            original = catalog[identities[gt]]
            xy = to_meters(source['uv'][0, query])
            size = source['size'][0, query]
            angle = source['angle'][0, query]
            a, b = float(size[0])/2, float(size[1])/2
            cosine, sine = math.cos(float(angle)), math.sin(float(angle))
            from shapely.geometry import Polygon
            box = Polygon([(float(xy[0])+cosine*x-sine*y, float(xy[1])+sine*x+cosine*y)
                           for x, y in ((-a, -b), (a, -b), (a, b), (-a, b))])
            ground = polygon(original, original.source_pose)
            iou = box.intersection(ground).area / max(box.union(ground).area, 1e-12)
            distance = float(torch.linalg.vector_norm(xy-truth['source_xy'][gt]))
            if distance > .010 or iou < .5:
                pairs = None
                break
            pairs.append((packed[int(query)], identities[gt]))
            color[identities[gt]] = original.color
        if pairs is None or len(pairs) != len(identities):
            counts['detection'] += 1
            continue
        neural = {}
        buckets = {}
        for query, object_id in pairs:
            neural[object_id] = (float(output['xy'][0, query, 0]), float(output['xy'][0, query, 1]),
                                 float(output['angle'][0, query]))
            buckets.setdefault(int(output['row'][0, query]), []).append((int(output['slot'][0, query]), object_id))
        predicted = [[object_id for _, object_id in sorted(buckets[key])] for key in sorted(buckets)]
        if record['instruction']['group_by'] != 'color':
            counts['not_color'] += 1
            continue
        repaired, changed = split_colors(predicted, color)
        if not changed:
            counts['already_pure'] += 1
            continue
        legal_rows, legal_sets = legal_row_set(record)
        key = tuple(tuple(row) for row in repaired)
        set_key = tuple(frozenset(row) for row in repaired)
        if key not in legal_rows:
            counts['order_only' if set_key in legal_sets else 'illegal_row'] += 1
            continue
        counts['legal_color_split'] += 1
        posed = teacher_pose(record, repaired)
        if posed is None:
            counts['legal_but_no_teacher_fit'] += 1
            continue
        full = {object_id: (pose.x_m, pose.y_m, pose.theta_rad) for object_id, pose in posed.items()}
        if judge_poses(record, full)['task']:
            counts['full_teacher_would_succeed'] += 1
        placed, affected = hybrid(record, repaired, predicted, neural, scene.objects)
        if placed is None:
            counts['no_unchanged_row'] += 1
            if judge_poses(record, full)['task']:
                counts['gain_only_if_all_rows_move'] += 1
            continue
        detail = judge_poses(record, placed)
        counts['affected_rows'] += len(affected)
        if detail['task']:
            counts['hybrid_gain'] += 1
        else:
            counts['hybrid_still_fails'] += 1
            for name in detail['errors']:
                gained_errors[name] += 1
    print(json.dumps(dict(official_success=official, counts=counts, hybrid_fail_errors=gained_errors,
                          baseline_fail=base_fail), default=lambda value: dict(value)))


if __name__ == '__main__':
    main()
