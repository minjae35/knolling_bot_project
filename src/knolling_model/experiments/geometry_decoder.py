"""Coordinates from a predicted row and order.

The selector chooses the row, the order, and the goal. This layer only solves
X and Y. A pose already inside the evaluator bands is left where it is. A
vertical opening tighter than the request is opened to that request when the
rows still fit, and a horizontal opening is placed at the request when the row
still fits. Otherwise the band placement is kept. When the predicted boxes
fit 1 mm inside the requested region, the layout is shifted there. A span
that does not fit is left where it is. If the row cannot fit the requested
region, the original coordinates are returned with an explicit failure
instead of a layout that still breaks the constraints.
"""
import math

import torch

from knolling_dataset.schema import HEIGHT_M, REGIONS, WIDTH_M

GAP_VALUES = (0.015, 0.020, 0.030)
REGION_NAMES = ('full', 'left', 'right')
BAND_M = 0.002
WORK_LOW = 0.010 - 0.0001
WORK_X_HIGH = WIDTH_M - 0.010 + 0.0001
WORK_Y_HIGH = HEIGHT_M - 0.010 + 0.0001
REGION_SLACK = 0.0001
# Pull a fitted layout this far inside the region when the slack exists.
REGION_INSET = 0.001
# Same band the axis check uses. An angle already inside it is left untouched.
AXIS_TOL = math.radians(2.)


def extent_offsets(angle, major, minor):
    """Corner offsets of the rotated envelope, relative to the object center."""
    cosine, sine = torch.cos(angle), torch.sin(angle)
    xs, ys = [], []
    for sign_x, sign_y in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
        local_x, local_y = sign_x*major/2, sign_y*minor/2
        xs.append(cosine*local_x-sine*local_y)
        ys.append(sine*local_x+cosine*local_y)
    offset_x, offset_y = torch.stack(xs), torch.stack(ys)
    return offset_x.min(), offset_x.max(), offset_y.min(), offset_y.max()


def _instruction(goal):
    if goal.shape[-1] != 13:
        raise ValueError('geometry decoder expects the 13-d goal')
    gap = (goal[6:9]*goal.new_tensor(GAP_VALUES)).sum()
    grouped = bool(goal[10] >= goal[9])
    region = REGIONS[REGION_NAMES[int(goal[3:6].argmax())]]
    return gap, grouped, (
        max(region[0]-REGION_SLACK, WORK_LOW), max(region[1]-REGION_SLACK, WORK_LOW),
        min(region[2]+REGION_SLACK, WORK_X_HIGH), min(region[3]+REGION_SLACK, WORK_Y_HIGH))


def _bands(actual, required):
    low, high = required-BAND_M, required+BAND_M
    return torch.minimum(torch.maximum(actual, low), high)


def _members(row, slot):
    groups = {}
    for index in range(len(row)):
        groups.setdefault(int(row[index]), []).append(index)
    ordered = []
    for row_id in sorted(groups):
        members = groups[row_id]
        slots = [int(slot[index]) for index in members]
        if len(slots) != len(set(slots)):
            return None
        members.sort(key=lambda index: int(slot[index]))
        ordered.append(members)
    return ordered


def _openings(xy, groups, offsets, gap, colors, grouped):
    horizontal, vertical = [], []
    for members in groups:
        for left, right in zip(members, members[1:]):
            opening = (xy[right, 0]+offsets[right][0])-(xy[left, 0]+offsets[left][1])
            horizontal.append((opening, gap))
    for upper, lower in zip(groups, groups[1:]):
        top = torch.stack([xy[index, 1]+offsets[index][2] for index in upper]).min()
        bottom = torch.stack([xy[index, 1]+offsets[index][3] for index in lower]).max()
        factor = 2. if grouped and int(colors[upper[0]]) != int(colors[lower[0]]) else 1.
        vertical.append((top-bottom, gap*factor))
    return horizontal, vertical


def _inside(xy, groups, offsets, limits):
    left, bottom, right, top = limits
    for members in groups:
        ys = torch.stack([xy[index, 1] for index in members])
        if float((ys.max()-ys.min()).detach()) > BAND_M+1e-6:
            return False
    for index, (xmin, xmax, ymin, ymax) in enumerate(offsets):
        x, y = xy[index, 0], xy[index, 1]
        if float((x+xmin).detach()) < left-1e-6 or float((y+ymin).detach()) < bottom-1e-6:
            return False
        if float((x+xmax).detach()) > right+1e-6 or float((y+ymax).detach()) > top+1e-6:
            return False
    return True


def _feasible(xy, groups, offsets, gap, colors, grouped, limits):
    if not _inside(xy, groups, offsets, limits):
        return False
    horizontal, vertical = _openings(xy, groups, offsets, gap, colors, grouped)
    for opening, required in horizontal+vertical:
        if float((opening-required).abs().detach()) > BAND_M+1e-6:
            return False
    return True


def _anchor(prefer, deltas, low_offset, high_offset, low_limit, high_limit):
    low = torch.stack([low_limit-delta-offset for delta, offset in zip(deltas, low_offset)]).max()
    high = torch.stack([high_limit-delta-offset for delta, offset in zip(deltas, high_offset)]).min()
    if float((low-high).detach()) > 1e-6:
        return None
    return torch.minimum(torch.maximum(prefer, low), high)


def decode_scene(xy, angle, row, slot, major, minor, color, goal, *, tighten_vertical=False,
                 exact_horizontal=False):
    """One unpadded scene. `color` is the predicted color index of each object."""
    if xy.ndim != 2 or xy.shape[-1] != 2 or len(xy) != len(angle) or len(xy) != len(row):
        raise ValueError('geometry decoder scene shapes do not match')
    if len(xy) == 0:
        return xy, 'empty'
    gap, grouped, limits = _instruction(goal)
    offsets = [extent_offsets(angle[index], major[index], minor[index]) for index in range(len(xy))]
    groups = _members(row, slot)
    if groups is None:
        return xy, 'infeasible:duplicate_slot'
    colors = [int(color[index]) for index in range(len(color))]
    horizontal, vertical = _openings(xy, groups, offsets, gap, colors, grouped)
    far = [abs(float((opening-required).detach())) > .0015 for opening, required in horizontal]
    needs_horizontal = exact_horizontal and any(far)
    if _feasible(xy, groups, offsets, gap, colors, grouped, limits) and not tighten_vertical and not needs_horizontal:
        return xy, 'kept'
    if needs_horizontal:
        desired_h = [required if flag else _bands(opening, required)
                     for (opening, required), flag in zip(horizontal, far)]
    else:
        desired_h = [_bands(opening, required) for opening, required in horizontal]
    if tighten_vertical:
        desired_v = [required if float((opening-required).detach()) < -1e-6 else _bands(opening, required)
                     for opening, required in vertical]
    else:
        desired_v = [_bands(opening, required) for opening, required in vertical]
    placed = xy.clone()
    cursor = 0
    for members in groups:
        deltas = [xy.new_zeros(())]
        for left, right, desired in zip(members, members[1:], desired_h[cursor:cursor+len(members)-1]):
            deltas.append(deltas[-1]+offsets[left][1]-offsets[right][0]+desired)
        cursor += max(len(members)-1, 0)
        prefer = torch.stack([xy[index, 0]-delta for index, delta in zip(members, deltas)]).mean()
        anchor = _anchor(prefer, deltas, [offsets[index][0] for index in members],
                         [offsets[index][1] for index in members], limits[0], limits[2])
        if anchor is None:
            return xy, 'infeasible:horizontal_room'
        for index, delta in zip(members, deltas):
            placed[index, 0] = anchor+delta
    deltas = [xy.new_zeros(())]
    for upper, lower, desired in zip(groups, groups[1:], desired_v):
        upper_drop = torch.stack([offsets[index][2] for index in upper]).min()
        lower_rise = torch.stack([offsets[index][3] for index in lower]).max()
        deltas.append(deltas[-1]+upper_drop-lower_rise-desired)
    prefer = []
    for members, delta in zip(groups, deltas):
        prefer.extend(xy[index, 1]-delta for index in members)
    anchor = _anchor(torch.stack(prefer).mean(), [delta for members, delta in zip(groups, deltas) for _ in members],
                     [offsets[index][2] for members in groups for index in members],
                     [offsets[index][3] for members in groups for index in members], limits[1], limits[3])
    if anchor is None:
        return xy, 'infeasible:vertical_room'
    for members, delta in zip(groups, deltas):
        for index in members:
            placed[index, 1] = anchor+delta
    if not _feasible(placed, groups, offsets, gap, colors, grouped, limits):
        return xy, 'infeasible:constraint'
    return placed, 'placed'


def align_goal_instruction(goal, texts):
    """Use the named region and gap when the goal picked a different one.

    The mass already on that field stays, so only the chosen bin changes.
    A goal that already agrees with the text is returned as the same tensor.
    No-op text names neither field.
    """
    if goal.ndim != 2 or goal.shape[-1] != 13 or len(texts) != goal.shape[0]:
        raise ValueError('instruction command does not match the goal batch')
    replaced = None
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError('instruction command text required')
        region = 1 if 'left region' in text else 2 if 'right region' in text else 0 if 'full workspace' in text else None
        gap = 0 if 'Leave 15 mm' in text else 1 if 'Leave 20 mm' in text else 2 if 'Leave 30 mm' in text else None
        chosen = []
        if region is not None and int(goal[index, 3:6].argmax()) != region:
            chosen.append((3, region))
        if gap is not None and int(goal[index, 6:9].argmax()) != gap:
            chosen.append((6, gap))
        if not chosen:
            continue
        if replaced is None:
            replaced = goal.clone()
        for start, choice in chosen:
            mass = replaced[index, start:start + 3].sum()
            replaced[index, start:start + 3] = 0
            replaced[index, start + choice] = mass
    return goal if replaced is None else replaced


def align_goal_axis(goal, texts):
    """Use the requested long-axis words for geometry. The selector goal stays put.

    A near tie in the goal can snap a moved object to the other axis. The
    instruction text names the axis directly. No-op text has no axis phrase,
    and that goal is returned unchanged.
    """
    if goal.ndim != 2 or goal.shape[-1] != 13 or len(texts) != goal.shape[0]:
        raise ValueError('axis command does not match the goal batch')
    replaced = goal.clone()
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError('axis command text required')
        vertical = 'long axes vertical' in text
        horizontal = 'long axes horizontal' in text
        if vertical == horizontal:
            continue
        mass = replaced[index, 1] + replaced[index, 2]
        replaced[index, 1] = mass if horizontal else replaced.new_zeros(())
        replaced[index, 2] = mass if vertical else replaced.new_zeros(())
    return replaced


def commanded_axis(goal):
    """Predicted axis for each scene. Inactive when keep wins or the axis is tied."""
    if goal.ndim != 2 or goal.shape[-1] != 13:
        raise ValueError('geometry decoder expects a batched 13-d goal')
    keep, horizontal, vertical = goal[:, 0], goal[:, 1], goal[:, 2]
    vertical_wins = vertical > horizontal
    horizontal_wins = horizontal > vertical
    active = (vertical_wins | horizontal_wins) & (torch.maximum(horizontal, vertical) > keep)
    target = torch.where(
        vertical_wins, goal.new_full(goal.shape[:1], math.pi / 2), goal.new_zeros(goal.shape[:1]))
    return active, target


def normalize_axis_angle(angle, valid, goal):
    """Turn a requested axis to 0 or pi/2 without swapping major and minor.

    The asset's long and short sides stay on their own axes. The pose angle is
    what makes that side horizontal or vertical, and it is the same angle the
    envelope and the returned pose both use. A keep command, a tied axis, or an
    angle already inside the axis band is not changed.
    """
    active, target = commanded_axis(goal)
    delta = torch.remainder(angle - target[:, None] + math.pi / 2, math.pi) - math.pi / 2
    replace = valid & active[:, None] & (delta.abs() > AXIS_TOL)
    return torch.where(replace, target[:, None].expand_as(angle), angle)


def _span_shift(xy, offsets, limits, inset):
    left, bottom, right, top = (limits[0]+inset, limits[1]+inset, limits[2]-inset, limits[3]-inset)
    dx_low = torch.stack([left-(xy[index, 0]+offsets[index][0]) for index in range(len(xy))]).max()
    dx_high = torch.stack([right-(xy[index, 0]+offsets[index][1]) for index in range(len(xy))]).min()
    dy_low = torch.stack([bottom-(xy[index, 1]+offsets[index][2]) for index in range(len(xy))]).max()
    dy_high = torch.stack([top-(xy[index, 1]+offsets[index][3]) for index in range(len(xy))]).min()
    return dx_low, dx_high, dy_low, dy_high


def _pick_shift(low, high):
    if float((low-high).detach()) > 1e-6:
        return None
    if float(low.detach()) <= 0. <= float(high.detach()):
        return low.new_zeros(())
    return low if float(low.detach()) > 0. else high


def nudge_scene(xy, angle, major, minor, goal):
    """Shift a layout up to the 1 mm region inset when its predicted span fits."""
    if len(xy) == 0:
        return xy, 'empty'
    _gap, _grouped, limits = _instruction(goal)
    offsets = [extent_offsets(angle[index], major[index], minor[index]) for index in range(len(xy))]
    inset = _span_shift(xy, offsets, limits, REGION_INSET)
    dx, dy = _pick_shift(inset[0], inset[1]), _pick_shift(inset[2], inset[3])
    if dx is None or dy is None or (abs(float(dx.detach())) < 1e-6 and abs(float(dy.detach())) < 1e-6):
        return xy, 'kept'
    moved = xy.clone()
    moved[:, 0] = moved[:, 0]+dx
    moved[:, 1] = moved[:, 1]+dy
    return moved, 'placed'


def nudge_batch(xy, angle, major, minor, goal, valid):
    """Apply the region inset to each valid prefix. Padding stays put."""
    if xy.shape[:2] != valid.shape or goal.ndim != 2 or len(goal) != len(xy):
        raise ValueError('geometry decoder batch shapes do not match')
    nudged = xy.clone()
    status = []
    for index in range(len(xy)):
        count = int(valid[index].sum())
        if count == 0 or not bool(valid[index, :count].all()):
            status.append('empty')
            continue
        pose, name = nudge_scene(
            xy[index, :count], angle[index, :count], major[index, :count], minor[index, :count], goal[index])
        nudged[index, :count] = pose
        status.append(name)
    return nudged, status


def _accept(base, base_status, proposal, proposal_status, valid):
    chosen = base.clone()
    status = []
    for index, (name, new_name) in enumerate(zip(base_status, proposal_status)):
        count = int(valid[index].sum())
        accepted = not str(new_name).startswith('infeasible') and new_name != 'empty'
        changed = accepted and count > 0 and not torch.allclose(
            proposal[index, :count], base[index, :count], atol=1e-6, rtol=0)
        if changed:
            chosen[index, :count] = proposal[index, :count]
            status.append(new_name)
        else:
            status.append(name)
    return chosen, status


def apply_geometry(output, objects, goal):
    """Replace target X/Y with the decoded coordinates. Row stays neural."""
    angle = normalize_axis_angle(output['angle'], objects.valid, goal)
    shared = (angle, output['row'], output['slot'], objects.size[..., 0], objects.size[..., 1],
              objects.color, goal, objects.valid)
    decoded, status = decode_batch(output['xy'], *shared)
    tightened, tight_status = decode_batch(decoded, *shared, tighten_vertical=True)
    chosen, chosen_status = _accept(decoded, status, tightened, tight_status, objects.valid)
    closed, closed_status = decode_batch(chosen, *shared, exact_horizontal=True)
    chosen, chosen_status = _accept(chosen, chosen_status, closed, closed_status, objects.valid)
    nudged, nudge_status = nudge_batch(
        chosen, angle, objects.size[..., 0], objects.size[..., 1], goal, objects.valid)
    chosen, chosen_status = _accept(chosen, chosen_status, nudged, nudge_status, objects.valid)
    replaced = dict(output)
    replaced['xy'] = chosen.to(dtype=output['xy'].dtype, device=output['xy'].device)
    replaced['angle'] = angle.to(dtype=output['angle'].dtype, device=output['angle'].device)
    replaced['geometry_status'] = chosen_status
    return replaced


def decode_batch(xy, angle, row, slot, major, minor, color, goal, valid, *, tighten_vertical=False,
                 exact_horizontal=False):
    """Apply the scene decoder to each valid prefix. Padding stays at zero."""
    if xy.shape[:2] != valid.shape or goal.ndim != 2 or len(goal) != len(xy):
        raise ValueError('geometry decoder batch shapes do not match')
    decoded = torch.zeros_like(xy)
    status = []
    for index in range(len(xy)):
        count = int(valid[index].sum())
        if count == 0 or not bool(valid[index, :count].all()):
            status.append('empty')
            continue
        pose, name = decode_scene(
            xy[index, :count], angle[index, :count], row[index, :count], slot[index, :count],
            major[index, :count], minor[index, :count], color[index, :count].argmax(-1), goal[index],
            tighten_vertical=tighten_vertical, exact_horizontal=exact_horizontal)
        decoded[index, :count] = pose
        status.append(name)
    return decoded, status
