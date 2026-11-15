"""Partial and no-op routing on top of the adopted full arrangement.

A full command never reaches this module. No-op copies the detected source
pose. Partial runs the selector and geometry decoder on the objects that move,
and leaves the others on their detected source pose. A command that names
one color moves the objects of that color. Those fixed objects are obstacles:
the moved group is translated only
when a shift keeps it inside the requested region and at least 10 mm away.
"""
import math

import torch
from shapely.geometry import Polygon

from knolling_dataset.schema import COLORS
from knolling_model.experiments.constraint_decode import repair_rows, select_layout
from knolling_model.experiments.geometry_decoder import _instruction, align_goal_axis, apply_geometry
from knolling_model.arrangement.contracts import Objects

CLEARANCE_M = 0.010
_STEP_M = 0.005
_REACH_M = 0.12
# A score just under one half is not a confident reject. Treating it as held
# sent a full command through the partial solver.
HELD_BELOW = 0.45


def text_requests_every_object(texts):
    """A full command names every object. A low selection score must not reroute it."""
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError('one instruction text per scene required')
    return all(text.startswith('Arrange all objects') for text in texts)


def named_color(text):
    """The one color a partial command names. Anything ambiguous stays unset."""
    if not isinstance(text, str):
        return None
    found = [color for color in COLORS if f'only the {color} objects' in text]
    return found[0] if len(found) == 1 else None


def supplement_moving(moving, color, text):
    """Use the named color as the moving set when that color is unambiguous.

    A partial command names exactly one color. Objects whose color is a unique
    match are the movers, and a selector object of another color is not moved.
    A tied color, a sentence that names no single color, or a scene with no
    sure match leaves the selector set unchanged.
    """
    if moving.ndim != 1 or color.ndim != 2 or color.shape[-1] != len(COLORS):
        raise ValueError('color supplement expects one scene')
    name = named_color(text)
    if name is None:
        return moving
    probability = color.detach()
    best = probability.argmax(-1)
    top = probability.max(-1).values
    second = probability.topk(2, dim=-1).values[:, 1]
    sure = (best == COLORS.index(name)) & (top > second)
    extra = sure.nonzero().flatten()
    if int(extra.numel()) == 0:
        return moving
    return extra


def command_is_full(keep_logit, selected, valid):
    """True when keep is off and no valid object is confidently rejected."""
    if keep_logit.shape[0] != valid.shape[0] or selected.shape != valid.shape:
        raise ValueError('keep and selection do not match the objects')
    keep_off = keep_logit.sigmoid() <= 0.5
    held = valid & (selected < HELD_BELOW)
    return bool((keep_off & ~held.any(-1)).all())


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


def group_is_clear(xy, angle, size, moving, goal):
    """True when the moved group is inside the region and clear of fixed objects."""
    if xy.ndim != 3 or xy.shape[0] != 1 or moving.ndim != 1:
        raise ValueError('obstacle clearance expects one scene')
    held = [index for index in range(xy.shape[1]) if index not in set(int(item) for item in moving.tolist())]
    _gap, _grouped, limits = _instruction(goal[0] if goal.ndim == 2 else goal)
    movers = [_polygon(
        (float(xy[0, index, 0]), float(xy[0, index, 1])), angle[0, index], size[0, index, 0], size[0, index, 1])
        for index in moving.tolist()]
    if not _fits(movers, limits):
        return False
    obstacles = [_polygon(xy[0, index], angle[0, index], size[0, index, 0], size[0, index, 1]) for index in held]
    return _clear(movers, obstacles)


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


def _source_pose(objects, index, route, keep_angle=None):
    count = objects.valid.shape[1]
    row = torch.full((1, count), -1, dtype=torch.long, device=objects.xy.device)
    slot = row.clone()
    angle = objects.angle if keep_angle is None else keep_angle
    return dict(
        xy=objects.xy[index:index + 1].clone(), angle=angle[index:index + 1].clone(),
        row=row, slot=slot, geometry_status=['source'], route=route)


def _place_selected(relation, row_model, objects, index, goal, chosen, keep_angle=None, text=None):
    subset = _take(objects, index, chosen)
    hidden = relation._hidden(subset.features())
    output = row_model(
        hidden, subset.valid, goal, subset.size[..., 0], subset.size[..., 1],
        None, None, True, colors=subset.color)
    output = select_layout(row_model, hidden, subset, goal, output)
    geometry_goal = align_goal_axis(goal, [text]) if text else goal
    output = apply_geometry(output, subset, geometry_goal)
    if str(output['geometry_status'][0]).startswith('infeasible'):
        count = int(subset.valid[0].sum())
        note = 'unresolved'
        for new_row, new_slot in repair_rows(
                output['row_logits'][0, :count], output['angle'][0, :count],
                subset.size[0, :count, 0], subset.size[0, :count, 1],
                subset.color[0, :count].argmax(-1), geometry_goal[0]):
            trial_row, trial_slot = output['row'].clone(), output['slot'].clone()
            trial_row[0, :count] = new_row
            trial_slot[0, :count] = new_slot
            candidate = apply_geometry(dict(output, row=trial_row, slot=trial_slot), subset, geometry_goal)
            if str(candidate['geometry_status'][0]).startswith('infeasible'):
                continue
            preview_xy = objects.xy[index:index + 1].clone()
            preview_xy[0, chosen] = candidate['xy'][0]
            preview_angle = (objects.angle if keep_angle is None else keep_angle)[index:index + 1].clone()
            preview_angle[0, chosen] = candidate['angle'][0]
            cleared = clear_held_objects(
                preview_xy, preview_angle, objects.size[index:index + 1], chosen, goal)
            if not group_is_clear(cleared, preview_angle, objects.size[index:index + 1], chosen, goal):
                continue
            output = dict(candidate, row_repair=['replaced'])
            note = 'replaced'
            break
        else:
            output = dict(output, row_repair=[note])
    placed = _source_pose(objects, index, 'partial', keep_angle)
    placed['xy'][0, chosen] = output['xy'][0]
    placed['angle'][0, chosen] = output['angle'][0]
    placed['xy'] = clear_held_objects(placed['xy'], placed['angle'], objects.size[index:index + 1], chosen, goal)
    placed['row'][0, chosen] = output['row'][0]
    placed['slot'][0, chosen] = output['slot'][0]
    placed['geometry_status'] = list(output['geometry_status'])
    return placed


def arrange_held(relation, row_model, objects, encoded, keep_angle=None, texts=None):
    """Place one batch. Keep copies the detection. Partial arranges the selection."""
    if texts is not None and len(texts) != objects.valid.shape[0]:
        raise ValueError('one instruction text per scene required')
    goal = encoded['vector']
    keep = encoded['keep_logit'].sigmoid()
    selected = encoded['selected']
    scenes = []
    for index in range(objects.valid.shape[0]):
        count = int(objects.valid[index].sum())
        text = None if texts is None else texts[index]
        moving = ((selected[index, :count] >= HELD_BELOW) & (keep[index] <= 0.5)).nonzero().flatten()
        moving = supplement_moving(moving, objects.color[index, :count], text)
        if int(moving.numel()) == 0:
            scene = _source_pose(objects, index, 'noop', keep_angle)
        else:
            scene = _place_selected(
                relation, row_model, objects, index, goal[index:index + 1], moving, keep_angle, text)
        mask = torch.zeros(1, objects.valid.shape[1], dtype=torch.bool, device=objects.xy.device)
        mask[0, moving] = True
        scene['moving_mask'] = mask
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
