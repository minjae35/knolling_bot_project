"""Why the eight-scene residual scored 0/8. No checkpoint is written."""
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from counterfactual_color_pose import hybrid, judge_poses, legal_row_set, split_colors
from knolling_dataset.schema import scene_from_dict
from knolling_dataset.validator import polygon
from knolling_model.coordinates import to_meters
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import poses_from_detection
from knolling_model.losses import match_source
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
from score_end_to_end import load_adopted
from shapely.geometry import Polygon
from train_predicted_pose import teacher_pose
from train_set_context import fresh_split
from verify_row_correction import RowCorrection, features_of


def official(record, identities, truth, source, valid, query_index, xy, angle):
    layout = dict(xy=xy.clone(), angle=angle.clone(), valid=valid.clone(),
                  query_index=query_index.clone(), rollout='free_rollout')
    view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
    judged = evaluate_batch(view, Batch({}, [truth], [dict(
        sample_id=record['sample_id'], object_ids=identities, record=record)]),
        semantic_check=semantic_errors_v3)[0]
    return dict(task=bool(accepted_success(record, judged['task_success'])), errors=judged['errors'])


def write_pose(xy, angle, order, poses):
    xy, angle = xy.clone(), angle.clone()
    for object_id, pose in poses.items():
        slot = order[object_id]
        xy[0, slot, 0], xy[0, slot, 1], angle[0, slot] = pose[0], pose[1], pose[2]
    return xy, angle


def main():
    runner.setup()
    train = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    _fit, dev, _blocked = fresh_split(train.records)
    system, relation, model, head, _saved = load_adopted()
    model.eval()
    gains = []
    for index in dev:
        if len(gains) >= 8:
            break
        record = train.records[index]
        if record['instruction']['group_by'] != 'color':
            continue
        image = train[index]['image'][None].to(stable.DEVICE)
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
        except ValueError:
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
        if 'count' in judged['errors'] or 'source_identity' in judged['errors']:
            continue
        active = torch.where(source['objectness_logits'][0].sigmoid() >= .5)[0]
        chosen = select_queries(visual['source'], system.threshold)[0]
        packed = {int(raw): slot for slot, raw in enumerate(chosen.tolist())}
        queries, gts = match_source({key: value[:, active] for key, value in source.items()}, [truth])[0]
        pairs, ok = [], True
        scene = scene_from_dict(record['source'])
        catalog = {obj.object_id: obj for obj in scene.objects}
        color = {}
        for query, gt in zip(active[queries].tolist(), gts.tolist()):
            if int(query) not in packed:
                ok = False
                break
            original = catalog[identities[gt]]
            xy = to_meters(source['uv'][0, query])
            size, angle = source['size'][0, query], source['angle'][0, query]
            a, b = float(size[0])/2, float(size[1])/2
            cosine, sine = math.cos(float(angle)), math.sin(float(angle))
            box = Polygon([(float(xy[0])+cosine*x-sine*y, float(xy[1])+sine*x+cosine*y)
                           for x, y in ((-a, -b), (a, -b), (a, b), (-a, b))])
            ground = polygon(original, original.source_pose)
            iou = box.intersection(ground).area / max(box.union(ground).area, 1e-12)
            if float(torch.linalg.vector_norm(xy-truth['source_xy'][gt])) > .010 or iou < .5:
                ok = False
                break
            pairs.append((packed[int(query)], identities[gt]))
            color[identities[gt]] = original.color
        if not ok or len(pairs) != len(identities) or success:
            continue
        ids = [object_id for _, object_id in sorted(pairs)]
        order = {object_id: query for query, object_id in pairs}
        neural = {object_id: (float(output['xy'][0, order[object_id], 0]), float(output['xy'][0, order[object_id], 1]),
                              float(output['angle'][0, order[object_id]])) for object_id in ids}
        buckets = {}
        for object_id in ids:
            query = order[object_id]
            buckets.setdefault(int(output['row'][0, query]), []).append((int(output['slot'][0, query]), object_id))
        predicted = [[object_id for _, object_id in sorted(items)] for _, items in sorted(buckets.items())]
        repaired, changed = split_colors(predicted, color)
        if not changed:
            continue
        legal_rows, _sets = legal_row_set(record)
        if tuple(tuple(row) for row in repaired) not in legal_rows:
            continue
        placed, _affected = hybrid(record, repaired, predicted, neural, scene.objects)
        if placed is None or not judge_poses(record, placed)['task']:
            continue
        widths, heights = {}, {}
        theta = 0. if record['instruction']['axis_alignment'] == 'horizontal' else -math.pi/2
        cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
        for object_id in ids:
            asset = catalog[object_id].asset
            widths[object_id] = asset.a_m*cosine+asset.b_m*sine
            heights[object_id] = asset.a_m*sine+asset.b_m*cosine
        base = torch.tensor([[neural[object_id][0], neural[object_id][1]] for object_id in ids])
        goal = torch.tensor([[placed[object_id][0], placed[object_id][1]] for object_id in ids])
        features = features_of(record, ids, repaired, widths, heights)
        gains.append(dict(
            record=record, identities=identities, truth=truth, source=source, ids=ids, order=order,
            neural=neural, placed=placed, predicted=predicted, repaired=repaired,
            valid=layout['valid'], query_index=layout['query_index'],
            neural_xy=layout['xy'], neural_angle=layout['angle'],
            base=base, goal=goal, features=features, widths=widths, heights=heights,
            gt_rows=record['supervision_v3']['rows']))
    if len(gains) < 8:
        print(json.dumps(dict(found=len(gains))))
        return

    def both(scene, poses):
        detail = judge_poses(scene['record'], poses)
        xy, angle = write_pose(scene['neural_xy'], scene['neural_angle'], scene['order'], poses)
        scored = official(scene['record'], scene['identities'], scene['truth'], scene['source'],
                          scene['valid'], scene['query_index'], xy, angle)
        return dict(arrangement=bool(detail['task']), arrangement_errors=detail['errors'],
                    official=scored['task'], official_errors=scored['errors'])

    rows = []
    pooled = []
    for scene in gains:
        posed = teacher_pose(scene['record'], scene['repaired'])
        teacher = None if posed is None else {
            object_id: (pose.x_m, pose.y_m, pose.theta_rad) for object_id, pose in posed.items()}
        neural_angle = {object_id: (scene['placed'][object_id][0], scene['placed'][object_id][1], scene['neural'][object_id][2])
                        for object_id in scene['ids']}
        feature_ok = all(abs(float(scene['features'][index, 0])-scene['widths'][object_id]) < 1e-6
                         for index, object_id in enumerate(scene['ids']))
        goal_ok = all(abs(float(scene['goal'][index, 0])-scene['placed'][object_id][0]) < 1e-6
                      and abs(float(scene['goal'][index, 1])-scene['placed'][object_id][1]) < 1e-6
                      for index, object_id in enumerate(scene['ids']))
        moved = [object_id for object_id in scene['ids']
                 if abs(scene['placed'][object_id][0]-scene['neural'][object_id][0]) > 1e-4
                 or abs(scene['placed'][object_id][1]-scene['neural'][object_id][1]) > 1e-4
                 or abs(scene['placed'][object_id][2]-scene['neural'][object_id][2]) > 1e-4]
        angle_deg = [abs(math.degrees(scene['placed'][object_id][2]-scene['neural'][object_id][2])) for object_id in moved]
        rows.append(dict(
            sample_id=scene['record']['sample_id'], objects=len(scene['ids']),
            repaired_equals_gt=scene['repaired'] == scene['gt_rows'],
            teacher_available=teacher is not None,
            teacher=None if teacher is None else both(scene, teacher),
            hybrid=both(scene, scene['placed']),
            hybrid_xy_neural_angle=both(scene, neural_angle),
            correspondence=feature_ok and goal_ok,
            moved=len(moved), angle_delta_deg_max=max(angle_deg) if angle_deg else 0.,
            delta_mm_max=float((scene['goal']-scene['base']).norm(dim=-1).max())*1000))
        for index, object_id in enumerate(scene['ids']):
            pooled.append((tuple(round(float(value), 6) for value in scene['features'][index]),
                           (scene['goal']-scene['base'])[index]))

    groups = {}
    for key, delta in pooled:
        groups.setdefault(key, []).append(delta)
    spreads = []
    for deltas in groups.values():
        stacked = torch.stack(deltas)
        spreads.append(float((stacked.max(0).values-stacked.min(0).values).max())*1000)
    features = torch.cat([scene['features'] for scene in gains])
    deltas = torch.cat([scene['goal']-scene['base'] for scene in gains])
    distance = torch.cdist(features, features)
    distance.fill_diagonal_(float('inf'))
    neighbor = distance.min(1).values
    delta_gap = torch.cdist(deltas, deltas)
    delta_gap.fill_diagonal_(0)
    nearest = distance.argmin(1)
    nearest_delta = delta_gap.gather(1, nearest[:, None]).squeeze(1)
    module = RowCorrection()
    loss = sum(F.smooth_l1_loss(scene['base']+module(scene['features']), scene['goal']) for scene in gains)
    loss.backward()
    grad = {name: None if value.grad is None else float(value.grad.norm())
            for name, value in module.named_parameters()}
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    optimizer.zero_grad()
    stepped = sum(F.smooth_l1_loss(scene['base']+module(scene['features']), scene['goal']) for scene in gains)
    stepped.backward()
    optimizer.step()
    optimizer.zero_grad()
    follow = sum(F.smooth_l1_loss(scene['base']+module(scene['features']), scene['goal']) for scene in gains)
    follow.backward()
    grad_after_one = {name: None if value.grad is None else float(value.grad.norm())
                      for name, value in module.named_parameters()}

    def run_fit(build, steps):
        probe = nn.Sequential(nn.Linear(build(gains[0]).shape[-1], 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(probe[-1].weight)
        nn.init.zeros_(probe[-1].bias)
        updater = torch.optim.Adam(probe.parameters(), lr=1e-3)
        curve = []
        for step in range(steps):
            current = sum(F.smooth_l1_loss(scene['base']+probe(build(scene)), scene['goal']) for scene in gains)
            if step in (0, 99, 399, steps-1):
                curve.append(round(float(current.detach()), 6))
            updater.zero_grad()
            current.backward()
            updater.step()
        errors, passed = [], 0
        with torch.no_grad():
            for scene in gains:
                pred = scene['base']+probe(build(scene))
                errors.append(round(float((pred-scene['goal']).norm(dim=-1).max())*1000, 1))
                poses = {object_id: (float(pred[index, 0]), float(pred[index, 1]), scene['placed'][object_id][2])
                         for index, object_id in enumerate(scene['ids'])}
                passed += judge_poses(scene['record'], poses)['task']
        return dict(loss_curve=curve, xy_error_mm_max=errors, task=passed)

    located = lambda scene: torch.cat((scene['features'], scene['base']), -1)
    print(json.dumps(dict(
        scenes=rows,
        feature_objects=len(pooled), unique_features=len(groups),
        colliding_features=sum(len(item) > 1 for item in groups.values()),
        max_delta_spread_mm=max(spreads),
        feature_min=float(features.min()), feature_max=float(features.max()),
        nearest_feature_l2_min=float(neighbor.min()), nearest_feature_l2_median=float(neighbor.median()),
        nearest_delta_mm_max=float(nearest_delta.max())*1000,
        delta_min_mm=float(deltas.min())*1000, delta_max_mm=float(deltas.max())*1000,
        grad_norm_at_init=grad, grad_norm_after_one_step=grad_after_one,
        shape_only_400=run_fit(lambda scene: scene['features'], 400),
        shape_only_2000=run_fit(lambda scene: scene['features'], 2000),
        shape_plus_frozen_xy_400=run_fit(located, 400)), indent=2))


if __name__ == '__main__':
    main()
