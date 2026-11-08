"""Versioned offline canonical teacher; no inference caller imports this solver."""
import hashlib
from pathlib import Path

from .schema import Pose, check_instruction, with_targets, digest
from .teacher_balanced import BalancedResult, templates, movement_cost
from .teacher_continuous import template_translation
from .validator import validate_layout as base_validation, validate_source, geometry

POLICY = 'continuous_rows_v3'
CONFIG = dict(policy=POLICY, tie_decimals=12, observed_templates=False,
              translation='continuous_envelope_projection', identity_first=True,
              axis_deg=2, gap_tolerance_m=.002, row_tolerance_m=.002)


def validate_layout(source, target, instruction, mode):
    return base_validation(source,target,instruction,mode,axis_deg=2,gap_tolerance=.002,row_tolerance=.002)


def provenance():
    names = ('teacher_v3.py', 'teacher_continuous.py', 'teacher_balanced.py',
             'teacher_legacy.py', 'schema.py', 'validator.py')
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    return dict(config=CONFIG, config_sha256=digest(CONFIG), code_sha256=digest(hashes), files=hashes)


def solve(scene, ins):
    check_instruction(scene, ins)
    mode = 'partial' if ins.locked_ids else 'full'
    def result(status, reason=None, target=None, **audit):
        return BalancedResult(status, mode, target, reason, audit=dict(teacher_policy=POLICY, **audit))
    if ins.expected_status != 'ok':
        raise ValueError('v3 verification accepts positive instructions only')
    current = with_targets(scene, {o.object_id:o.source_pose for o in scene.objects},
                           [o.object_id for o in scene.objects])
    if validate_layout(scene, current, ins, 'no_op').valid:
        out = result('ok', target=current, translation='identity')
        out.mode = 'no_op'
        return out
    if ins.keep_all:
        return result('infeasible', 'keep_all_violates_final_geometry')
    if not validate_source(scene, 'full').valid:
        return result('generation_failure', 'invalid_source')
    locked = [o for o in scene.objects if o.object_id in ins.locked_ids]
    if not geometry(locked, 'source_pose', .010, .010).valid:
        return result('infeasible', 'immutable_objects_violate_final_geometry')
    candidates = []
    for template in templates(scene, ins):
        if template['kind'] == 'observed':
            continue
        center = template_translation(scene, ins, template, decimals=12)
        if center is None:
            continue
        x, y = center
        poses = {o.object_id:o.source_pose for o in locked}
        poses.update({oid:Pose(x+p.x_m, y+p.y_m, p.theta_rad) for oid,p in template['relative'].items()})
        target = with_targets(scene, poses, ins.locked_ids)
        if not validate_layout(scene, target, ins, mode).valid:
            continue
        move = movement_cost(scene, ins, poses)
        shape = (round(template['J_shape'],12), round(template['width']*template['height'],12),
                 len(template['rows']), template['key'])
        key = (round(move,12), *shape, -y, x) if locked else shape
        candidates.append((key,target,template,center,move))
    if not candidates:
        return result('generation_failure', 'conservative_candidate_family_exhausted')
    _, target, template, center, move = min(candidates, key=lambda item:item[0])
    out = result('ok', target=target, center_m=center, J_move=move, template_key=template['key'],
                 translation='continuous_envelope_projection', valid_candidates=len(candidates))
    out.rows = [[o.object_id for o in row] for row in template['rows']]
    return out
