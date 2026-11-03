from dataclasses import replace
import pytest
from knolling_dataset.assets import build_registry
from knolling_dataset.scenes import generate_scene
from knolling_dataset.instructions import generate_instruction, negative_instruction
from knolling_dataset.schema import Pose
from knolling_dataset.teacher import solve
from knolling_dataset.validator import validate_layout


def test_teacher_many_seeds_and_repeated_noop():
    pool = build_registry()["train"]
    successes = {"full":0,"partial":0}
    for seed in range(100):
        mode = "full" if seed%2 == 0 else "partial"
        scene = generate_scene(pool,seed,"train",2+seed%7,mode)
        ins = generate_instruction(scene,seed,mode)
        result = solve(scene,ins)
        assert result == solve(scene,ins)
        if result.status == "generation_failure":
            assert result.target is None
            continue
        assert result.status == "ok"
        assert validate_layout(scene,result.target,ins,result.mode).valid
        successes[mode] += 1
        current = replace(scene,objects=tuple(replace(o,source_pose=o.target_pose,target_pose=None,
                          locked=False,move_label=False) for o in result.target.objects))
        repeated = solve(current,ins)
        assert repeated.mode == "no_op" and repeated.status == "ok"
        assert not repeated.rows and not repeated.relations
        assert all(o.source_pose == o.target_pose and not o.move_label and o.locked for o in repeated.target.objects)
    assert successes["full"] >= 35 and successes["partial"] >= 35


def test_keep_all_invalid_is_proven_infeasible(simple_scene):
    bad = replace(simple_scene,objects=(simple_scene.objects[0],replace(simple_scene.objects[1],source_pose=Pose(.265,.2,0))))
    ins = generate_instruction(bad,1,"keep_all")
    result = solve(bad,ins)
    assert result.status == "infeasible" and result.target is None
    assert result.reason == "keep_all_violates_final_geometry"


def test_search_budget_is_not_infeasibility(simple_scene):
    ins = generate_instruction(simple_scene,1,"full",axis_alignment="vertical")
    result = solve(simple_scene,ins,search_budget=0)
    assert result.status == "generation_failure" and result.reason == "search_budget"
    for status in ("ambiguous","unsupported"):
        assert solve(simple_scene,negative_instruction(simple_scene,status)).status == status


def test_pure_color_rows_and_locked_identity():
    pool = build_registry()["train"]
    scene = generate_scene(pool,41,"train",8)
    ins = generate_instruction(scene,2,"full",group_by="color",region="full",gap_m=.015)
    result = solve(scene,ins)
    assert result.status == "ok"
    byid = {o.object_id:o for o in scene.objects}
    assert all(len({byid[oid].color for oid in row}) == 1 for row in result.rows)
    partial = generate_instruction(scene,3,"partial",region="full")
    result = solve(scene,partial)
    assert result.status == "ok"
    assert all(o.target_pose == o.source_pose for o in result.target.objects if o.locked)
