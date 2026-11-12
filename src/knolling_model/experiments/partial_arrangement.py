"""Partial and no-op routing on top of the adopted full arrangement.

A full command never reaches this module. No-op copies the detected source
pose. Partial runs the existing selector and geometry decoder on the objects
the goal encoder selected, and leaves the others on their detected source
pose. Those fixed objects are obstacles: the moved group is translated only
when a shift keeps it inside the requested region and at least 10 mm away.
"""
import math

import torch
from shapely.geometry import Polygon

from knolling_model.experiments.constraint_decode import select_layout
from knolling_model.experiments.geometry_decoder import _instruction, apply_geometry
from knolling_model.recovery.contracts import Objects

CLEARANCE_M = 0.010
_STEP_M = 0.005
_REACH_M = 0.12


def command_is_full(keep_logit, selected, valid):
    """True when keep is off and every valid object is selected."""
    if keep_logit.shape[0] != valid.shape[0] or selected.shape != valid.shape:
        raise ValueError('keep and selection do not match the objects')
    keep_off = keep_logit.sigmoid() <= 0.5
    covered = (selected >= 0.5) | ~valid
    return bool((keep_off & covered.all(-1)).all())


def _shifts():
    count = int(round(_REACH_M / _STEP_M))
    values = [index * _STEP_M for index in range(-count, count + 1)]
    pairs = [(dx, dy) for dx in values for dy in values if dx or dy]
    pairs.sort(key=lambda item: item[0] * item[0] + item[1] * item[1])
    return pairs


_SHIFTS = _shifts()


def _polygon(center, angle, major, minor):
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    half_x, half_y = float(major) / 2, float(minor) / 2
    points = []
    for along, across in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)):
        points.append((float(center[0]) + cosine * along - sine * across,
                       float(center[1]) + sine * along + cosine * across))
    return Polygon(points)


def _fits(movers, limits):
    left, bottom, right, top = (float(item) for item in limits)
    for poly in movers:
        xmin, ymin, xmax, ymax = poly.bounds
        if xmin < left - 1e-6 or ymin < bottom - 1e-6 or xmax > right + 1e-6 or ymax > top + 1e-6:
            return False
    return True


def _clear(movers, held):
    for mover in movers:
        for obstacle in held:
            if mover.intersection(obstacle).area > 1e-12 or mover.distance(obstacle) < CLEARANCE_M - 1e-4:
                return False
    return True


def clear_held_objects(xy, angle, size, moving, goal):
    """Translate the moved group off fixed objects. Fixed poses are not edited."""
    if xy.ndim != 3 or xy.shape[0] != 1 or moving.ndim != 1:
        raise ValueError('obstacle clearance expects one scene')
    count = xy.shape[1]
    held = [index for index in range(count) if index not in set(int(item) for item in moving.tolist())]
    if not held or int(moving.numel()) == 0:
        return xy
    _gap, _grouped, limits = _instruction(goal[0] if goal.ndim == 2 else goal)

    def placed(dx, dy):
        movers, obstacles = [], []
        for index in moving.tolist():
            center = (float(xy[0, index, 0]) + dx, float(xy[0, index, 1]) + dy)
            movers.append(_polygon(center, angle[0, index], size[0, index, 0], size[0, index, 1]))
        for index in held:
            obstacles.append(_polygon(xy[0, index], angle[0, index], size[0, index, 0], size[0, index, 1]))
        return movers, obstacles

    movers, obstacles = placed(0., 0.)
    if _fits(movers, limits) and _clear(movers, obstacles):
        return xy
    for dx, dy in _SHIFTS:
        movers, obstacles = placed(dx, dy)
        if _fits(movers, limits) and _clear(movers, obstacles):
            shifted = xy.clone()
            shifted[0, moving, 0] = shifted[0, moving, 0] + dx
            shifted[0, moving, 1] = shifted[0, moving, 1] + dy
            return shifted
    return xy


def _take(objects, index, chosen):
    def column(value):
        return value[index, chosen][None]

    return Objects(
        xy=column(objects.xy), size=column(objects.size), angle=column(objects.angle),
        color=column(objects.color), shape=column(objects.shape),
        valid=torch.ones(1, int(chosen.numel()), dtype=torch.bool, device=objects.xy.device),
        query_index=column(objects.query_index)).validate()


def _source_pose(objects, index, route):
    count = objects.valid.shape[1]
    row = torch.full((1, count), -1, dtype=torch.long, device=objects.xy.device)
    slot = row.clone()
    return dict(
        xy=objects.xy[index:index + 1].clone(), angle=objects.angle[index:index + 1].clone(),
        row=row, slot=slot, geometry_status=['source'], route=route)


def _place_selected(relation, row_model, objects, index, goal, chosen):
    subset = _take(objects, index, chosen)
    hidden = relation._hidden(subset.features())
    output = row_model(
        hidden, subset.valid, goal, subset.size[..., 0], subset.size[..., 1],
        None, None, True, colors=subset.color)
    output = select_layout(row_model, hidden, subset, goal, output)
    output = apply_geometry(output, subset, goal)
    placed = _source_pose(objects, index, 'partial')
    placed['xy'][0, chosen] = output['xy'][0]
    placed['angle'][0, chosen] = output['angle'][0]
    placed['xy'] = clear_held_objects(placed['xy'], placed['angle'], objects.size[index:index + 1], chosen, goal)
    placed['row'][0, chosen] = output['row'][0]
    placed['slot'][0, chosen] = output['slot'][0]
    placed['geometry_status'] = list(output['geometry_status'])
    return placed


def arrange_held(relation, row_model, objects, encoded):
    """Place one batch. Keep copies the detection. Partial arranges the selection."""
    goal = encoded['vector']
    keep = encoded['keep_logit'].sigmoid()
    selected = encoded['selected']
    scenes = []
    for index in range(objects.valid.shape[0]):
        count = int(objects.valid[index].sum())
        moving = ((selected[index, :count] >= 0.5) & (keep[index] <= 0.5)).nonzero().flatten()
        if float(keep[index]) > 0.5 or int(moving.numel()) == 0:
            scene = _source_pose(objects, index, 'noop')
        else:
            scene = _place_selected(relation, row_model, objects, index, goal[index:index + 1], moving)
        scene['selected'] = selected[index:index + 1]
        scene['keep_probability'] = keep[index:index + 1]
        scenes.append(scene)
    if len(scenes) == 1:
        return scenes[0]
    merged = {}
    for key, value in scenes[0].items():
        if torch.is_tensor(value):
            merged[key] = torch.cat([scene[key] for scene in scenes], 0)
        else:
            merged[key] = [item for scene in scenes for item in (
                scene[key] if isinstance(scene[key], list) else [scene[key]])]
    return merged
