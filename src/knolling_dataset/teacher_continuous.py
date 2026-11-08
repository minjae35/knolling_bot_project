"""Offline-only experimental teacher; never registered as a dataset default.

For each balanced row template, project its preferred translation onto free
axis-aligned envelope space. This removes the old 5mm translation lattice, not
discrete template changes or obstacle-side ties. Envelope constraints are
conservative: failure is NOT proof that the physical task is infeasible.
"""
import math

from .schema import Pose, REGIONS, check_instruction, with_targets
from .teacher_balanced import BalancedResult, templates, movement_cost
from .validator import polygon, validate_layout, validate_source, geometry

POLICY = 'continuous_envelope_v0_review'


def project_translation(preferred, bounds, forbidden):
    """Closest point in a rectangle outside a union of open rectangles.

    Rectangle-edge coordinates plus the preferred coordinates contain a nearest
    feasible point: within any rectangular free cell the solution is a coordinate
    clamp. No sampled grid or numerical optimizer is involved.
    """
    l, b, r, t = bounds
    if l > r or b > t:
        return None
    if not all(math.isfinite(x) for x in (*preferred, *bounds, *(v for rect in forbidden for v in rect))):
        raise ValueError('nonfinite translation geometry')
    clamp_x = lambda x: min(r, max(l, x))
    clamp_y = lambda y: min(t, max(b, y))
    xs = {l, r, clamp_x(preferred[0])}
    ys = {b, t, clamp_y(preferred[1])}
    for x0, y0, x1, y1 in forbidden:
        xs.update((clamp_x(x0), clamp_x(x1)))
        ys.update((clamp_y(y0), clamp_y(y1)))
    candidates = [(x, y) for x in sorted(xs) for y in sorted(ys)
                  if not any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in forbidden)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: ((p[0]-preferred[0])**2+(p[1]-preferred[1])**2, -p[1], p[0]))


def template_translation(scene, ins, template):
    l, b, r, t = REGIONS[ins.region]
    half_w, half_h = template['width']/2, template['height']/2
    bounds = l+half_w, b+half_h, r-half_w, t-half_h
    locked = [o for o in scene.objects if o.object_id in ins.locked_ids]
    selected = [o for o in scene.objects if o.object_id in ins.selected_ids]
    if locked:
        preferred = tuple(math.fsum(getattr(o.source_pose, axis)-getattr(template['relative'][o.object_id], axis)
                                    for o in selected)/len(selected) for axis in ('x_m', 'y_m'))
    else:
        preferred = ((l+r)/2, (b+t)/2)
    forbidden = []
    # Slightly expand to avoid floating-point underflow at the 10mm safety gap.
    gap = .0100001
    for o in selected:
        relative = polygon(o, template['relative'][o.object_id]).bounds
        for fixed in locked:
            obstacle = polygon(fixed, fixed.source_pose).bounds
            forbidden.append((obstacle[0]-relative[2]-gap, obstacle[1]-relative[3]-gap,
                              obstacle[2]-relative[0]+gap, obstacle[3]-relative[1]+gap))
    return project_translation(preferred, bounds, forbidden)


def solve(scene, ins):
    """Produce an in-memory review candidate, retaining the existing semantics."""
    check_instruction(scene, ins)
    mode = 'partial' if ins.locked_ids else 'full'
    def result(status, reason=None, target=None, **audit):
        return BalancedResult(status, mode, target, reason,
                              audit=dict(teacher_policy=POLICY, review_only=True, **audit))
    if ins.expected_status in ('unsupported', 'ambiguous'):
        return result(ins.expected_status, ins.reason)
    if ins.expected_status != 'ok':
        raise ValueError('feasibility cannot be supplied as a label')
    current = with_targets(scene, {o.object_id: o.source_pose for o in scene.objects},
                           [o.object_id for o in scene.objects])
    if validate_layout(scene, current, ins, 'no_op').valid:
        out = result('ok', target=current, translation='identity')
        out.mode = 'no_op'
        return out
    if ins.keep_all:
        return result('infeasible', 'keep_all_violates_final_geometry')
    if not validate_source(scene, 'full').valid:
        return result('generation_failure', 'invalid_source_scene')
    locked = [o for o in scene.objects if o.object_id in ins.locked_ids]
    if not geometry(locked, 'source_pose', .010, .010).valid:
        return result('infeasible', 'immutable_objects_violate_final_geometry')
    candidates = []
    # Existing-arrangement detection uses tolerance boundaries; exclude it from
    # this candidate policy so the template family depends on dimensions/order.
    for template in templates(scene, ins):
        if template['kind'] == 'observed':
            continue
        center = template_translation(scene, ins, template)
        if center is None:
            continue
        x, y = center
        poses = {o.object_id: o.source_pose for o in locked}
        poses.update({oid: Pose(x+p.x_m, y+p.y_m, p.theta_rad)
                      for oid, p in template['relative'].items()})
        target = with_targets(scene, poses, ins.locked_ids)
        if not validate_layout(scene, target, ins, mode).valid:
            continue
        shape_key = (template['J_shape'], template['width']*template['height'],
                     len(template['rows']), template['key'])
        move = movement_cost(scene, ins, poses)
        key = (move, *shape_key, -y, x) if locked else shape_key
        candidates.append((key, target, template, center, move))
    if not candidates:
        return result('generation_failure', 'conservative_candidate_family_exhausted')
    _, target, template, center, move = min(candidates, key=lambda item: item[0])
    out = result('ok', target=target, center_m=center, J_move=move, template_key=template['key'],
                 valid_candidates=len(candidates), translation='continuous_envelope_projection')
    out.rows = [[o.object_id for o in row] for row in template['rows']]
    return out
