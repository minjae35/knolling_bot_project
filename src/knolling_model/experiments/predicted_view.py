"""Place source assets at predicted poses for a visualization render.

Source identity uses the detector's source center, size, and angle. The pose
written for Blender is the predicted target pose. The dataset target pose is
not read, and an unmatched asset is left out instead of being borrowed.
"""
import math
from dataclasses import replace

import torch
from shapely.geometry import Polygon

from knolling_dataset.schema import Pose, scene_from_dict
from knolling_dataset.validator import polygon
from knolling_model.losses import match_source
from knolling_model.recovery.selection import select_queries


def associate_source_assets(record, visual, output, objects, threshold=.5):
    """One source asset per accepted detection. Unmatched assets are omitted."""
    source = scene_from_dict(record['source'])
    by_id = {obj.object_id: obj for obj in source.objects}
    identities = sorted(by_id)
    device = visual['source']['uv'].device
    truth = {
        'source_xy': torch.tensor(
            [[by_id[name].source_pose.x_m, by_id[name].source_pose.y_m] for name in identities],
            dtype=torch.float32, device=device),
        'source_angle': torch.tensor(
            [by_id[name].source_pose.theta_rad for name in identities],
            dtype=torch.float32, device=device),
        'size': torch.tensor(
            [[by_id[name].asset.a_m, by_id[name].asset.b_m] for name in identities],
            dtype=torch.float32, device=device),
    }
    chosen = select_queries(visual['source'], threshold)[0]
    if len(chosen) == 0 or len(truth['source_xy']) > len(chosen):
        return _empty(source, identities, len(chosen))
    limited = {key: value[:, chosen] for key, value in visual['source'].items()}
    local, gt_index = match_source(limited, [truth])[0]
    accepted, rejected = [], set()
    for query_local, asset_index in zip(local.tolist(), gt_index.tolist()):
        query = int(chosen[query_local])
        slot = int((objects.query_index[0] == query).nonzero()[0])
        asset = by_id[identities[asset_index]]
        center = objects.xy[0, slot]
        size = objects.size[0, slot]
        angle = objects.angle[0, slot]
        distance = float(torch.linalg.vector_norm(center-truth['source_xy'][asset_index]))
        if distance <= .010 and _iou(center, size, angle, asset) >= .5:
            accepted.append(dict(
                object_id=asset.object_id,
                query_index=query,
                pose=Pose(float(output['xy'][0, slot, 0]), float(output['xy'][0, slot, 1]),
                          float(output['angle'][0, slot]))))
        else:
            rejected.add(asset.object_id)
    matched_ids = {item['object_id'] for item in accepted}
    omitted = [name for name in identities if name not in matched_ids]
    return dict(scene=source, matched=accepted, omitted=omitted,
                rejected=sorted(rejected), predictions=len(chosen), assets=len(identities))


def scene_at_predicted_poses(association):
    """Copy matched assets. Their rendered pose is the prediction."""
    matched = association['matched']
    if not 2 <= len(matched) <= 8:
        raise ValueError('a visualization needs 2 to 8 corresponded assets')
    poses = {item['object_id']: item['pose'] for item in matched}
    objects = tuple(
        replace(obj, target_pose=poses[obj.object_id])
        for obj in association['scene'].objects if obj.object_id in poses)
    return replace(association['scene'], objects=objects)


def _empty(source, identities, predictions):
    return dict(scene=source, matched=[], omitted=list(identities), rejected=[],
                predictions=predictions, assets=len(identities))


def _iou(center, size, angle, asset):
    half_a, half_b = float(size[0])/2, float(size[1])/2
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    outline = Polygon([
        (float(center[0])+cosine*x-sine*y, float(center[1])+sine*x+cosine*y)
        for x, y in ((-half_a, -half_b), (half_a, -half_b), (half_a, half_b), (-half_a, half_b))])
    actual = polygon(asset, asset.source_pose)
    union = outline.union(actual).area
    return outline.intersection(actual).area/union if union else 0.
