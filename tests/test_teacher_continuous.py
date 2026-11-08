from dataclasses import replace

import pytest

from knolling_dataset.schema import Pose
from knolling_dataset.teacher_continuous import project_translation, solve, POLICY
from knolling_dataset.validator import validate_layout


def test_continuous_projection_and_obstacle_boundaries():
    assert project_translation((.123456, .234567), (0, 0, 1, 1), []) == (.123456, .234567)
    assert project_translation((.1, .2), (1, 0, 0, 1), []) is None
    assert project_translation((.5, .5), (0, 0, 1, 1), [(-1, -1, 2, 2)]) is None
    x, y = project_translation((.21, .5), (0, 0, 1, 1), [(.2, .4, .6, .7)])
    assert x == .2 and y == .5
    assert project_translation((.21, .50001), (0, 0, 1, 1), [(.2, .4, .6, .7)]) == (.2, .50001)
    with pytest.raises(ValueError, match='nonfinite'):
        project_translation((float('nan'), 0), (0, 0, 1, 1), [])


def test_teacher_validity_locked_identity_and_local_perturbation(simple_scene, simple_instruction):
    # One selected object changes orientation away from an unobstructed source.
    scene = replace(simple_scene, objects=(replace(simple_scene.objects[0], source_pose=Pose(.123456, .2, .2)),
                                         replace(simple_scene.objects[1], source_pose=Pose(.45, .3, 0))))
    ins = replace(simple_instruction, selected_ids=('o1',), locked_ids=('o2',), axis_alignment='vertical')
    original = solve(scene, ins)
    assert original.status == 'ok' and original.audit['teacher_policy'] == POLICY
    assert validate_layout(scene, original.target, ins, 'partial').valid
    assert original.target.objects[1].target_pose == scene.objects[1].source_pose
    x = original.target.objects[0].target_pose.x_m
    assert x == pytest.approx(.123456)
    perturbed = replace(scene, objects=(replace(scene.objects[0], source_pose=Pose(.123466, .2, .2)), scene.objects[1]))
    changed = solve(perturbed, ins)
    assert changed.status == 'ok'
    assert changed.target.objects[0].target_pose.x_m-x == pytest.approx(.000010)
    assert solve(scene, ins).target == original.target
    assert all(o.target_pose is None for o in scene.objects)


def test_teacher_keeps_valid_noop_and_full_geometry(simple_scene, simple_instruction):
    noop = solve(simple_scene, simple_instruction)
    assert noop.status == 'ok' and noop.mode == 'no_op'
    vertical = replace(simple_instruction, axis_alignment='vertical')
    full = solve(simple_scene, vertical)
    assert full.status == 'ok' and validate_layout(simple_scene, full.target, vertical, 'full').valid
