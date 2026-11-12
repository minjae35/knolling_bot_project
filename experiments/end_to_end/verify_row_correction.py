"""Fit a residual pose on eight color-split scenes. Free rollout decides a pilot."""
import json

import torch
from torch import nn
from torch.nn import functional as F

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from counterfactual_color_pose import hybrid, judge_poses, legal_row_set, split_colors
from knolling_dataset.schema import scene_from_dict
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
from knolling_dataset.validator import polygon
from score_end_to_end import load_adopted
from train_set_context import fresh_split
import math
from shapely.geometry import Polygon


class RowCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


def features_of(record, ids, rows, widths, heights):
    instruction = record['instruction']
    gap = instruction['gap_m']
    located = {}
    summary = {}
    for row in rows:
        located.update({object_id: (index, len(row)) for index, object_id in enumerate(row)})
        summary[tuple(row)] = (len(row), sum(widths[object_id] for object_id in row),
                               max(heights[object_id] for object_id in row))
    rows_of = []
    for object_id in ids:
        count, total, tall = summary[next(tuple(row) for row in rows if object_id in row)]
        slot, length = located[object_id]
        rows_of.append((widths[object_id], heights[object_id], gap, count, total, tall, slot/max(length-1, 1)))
    return torch.tensor(rows_of, dtype=torch.float32)


def main():
    runner.setup()
    train = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    _fit, dev, _blocked = fresh_split(train.records)
    system, relation, model, head, _saved = load_adopted()
    model.eval()
    gains, anchors = [], []
    for index in dev:
        if len(gains) >= 8 and len(anchors) >= 2:
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
        if not ok or len(pairs) != len(identities):
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
        if success and len(anchors) < 2:
            anchors.append(dict(record=record, ids=ids, rows=predicted, neural=neural, objects=scene.objects))
            continue
        repaired, changed = split_colors(predicted, color)
        if success or not changed:
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
        gains.append(dict(record=record, ids=ids, rows=repaired, free_rows=predicted, neural=neural, target=placed,
                          objects=scene.objects, widths=widths, heights=heights,
                          base=torch.tensor([[neural[object_id][0], neural[object_id][1]] for object_id in ids]),
                          goal=torch.tensor([[placed[object_id][0], placed[object_id][1]] for object_id in ids]),
                          features=features_of(record, ids, repaired, widths, heights)))
    if len(gains) < 8:
        print(json.dumps(dict(status='NO-GO', reason='not enough placeable scenes', found=len(gains))))
        return
    module = RowCorrection()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    forced = 0
    for _step in range(400):
        loss = 0
        for scene in gains:
            pred = scene['base']+module(scene['features'])
            loss = loss+F.smooth_l1_loss(pred, scene['goal'])
        for scene in anchors:
            widths = {object_id: 0. for object_id in scene['ids']}
            heights = dict(widths)
            theta = 0. if scene['record']['instruction']['axis_alignment'] == 'horizontal' else -math.pi/2
            cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
            catalog = {obj.object_id: obj for obj in scene['objects']}
            for object_id in scene['ids']:
                asset = catalog[object_id].asset
                widths[object_id] = asset.a_m*cosine+asset.b_m*sine
                heights[object_id] = asset.a_m*sine+asset.b_m*cosine
            base = torch.tensor([[scene['neural'][object_id][0], scene['neural'][object_id][1]] for object_id in scene['ids']])
            pred = base+module(features_of(scene['record'], scene['ids'], scene['rows'], widths, heights))
            loss = loss+F.smooth_l1_loss(pred, base)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    forced = 0
    shifts = []
    for scene in gains:
        pred = (scene['base']+module(scene['features'])).detach()
        poses = {object_id: (float(pred[index, 0]), float(pred[index, 1]), scene['neural'][object_id][2])
                 for index, object_id in enumerate(scene['ids'])}
        forced += judge_poses(scene['record'], poses)['task']
    for scene in anchors:
        widths = {}
        heights = {}
        theta = 0. if scene['record']['instruction']['axis_alignment'] == 'horizontal' else -math.pi/2
        cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
        catalog = {obj.object_id: obj for obj in scene['objects']}
        for object_id in scene['ids']:
            asset = catalog[object_id].asset
            widths[object_id] = asset.a_m*cosine+asset.b_m*sine
            heights[object_id] = asset.a_m*sine+asset.b_m*cosine
        base = torch.tensor([[scene['neural'][object_id][0], scene['neural'][object_id][1]] for object_id in scene['ids']])
        pred = (base+module(features_of(scene['record'], scene['ids'], scene['rows'], widths, heights))).detach()
        shifts.append(float((pred-base).norm(dim=-1).max())*1000)
    free = 0
    for scene in gains:
        pred = (scene['base']+module(features_of(
            scene['record'], scene['ids'], scene['free_rows'], scene['widths'], scene['heights']))).detach()
        poses = {object_id: (float(pred[index, 0]), float(pred[index, 1]), scene['neural'][object_id][2])
                 for index, object_id in enumerate(scene['ids'])}
        free += judge_poses(scene['record'], poses)['task']
    print(json.dumps(dict(forced_success=forced, anchor_shift_mm=shifts, free_on_mixed_rows=free,
                          pilot='GO' if forced == 8 and free > 0 and max(shifts) < 2 else 'NO-GO')))


if __name__ == '__main__':
    main()
