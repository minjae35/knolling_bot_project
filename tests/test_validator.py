from dataclasses import replace
import pytest
from knolling_dataset.schema import Pose, with_targets
from knolling_dataset.validator import validate_layout, validate_source, geometry


def same(scene, locks=()):
    return with_targets(scene,{o.object_id:o.source_pose for o in scene.objects},locks)


def test_valid_and_order_independent_identity(simple_scene,simple_instruction):
    result = same(simple_scene)
    assert validate_layout(simple_scene,result,simple_instruction,"full").valid
    result = replace(result,objects=tuple(reversed(result.objects)))
    assert validate_layout(simple_scene,result,simple_instruction,"full").valid


@pytest.mark.parametrize("x,code",[(.21,"collision"),(.265,"clearance"),(.59,"boundary"),(.37,"horizontal_gap")])
def test_bad_geometry(simple_scene,simple_instruction,x,code):
    result = with_targets(simple_scene,{"o1":Pose(.20,.20,0),"o2":Pose(x,.20,0)})
    assert code in validate_layout(simple_scene,result,simple_instruction,"full").errors


def test_noop_final_gap_and_exactness(simple_scene,simple_instruction):
    noop = same(simple_scene,("o1","o2"))
    assert validate_layout(simple_scene,noop,simple_instruction,"no_op").valid
    bad = replace(simple_scene,objects=(simple_scene.objects[0],replace(simple_scene.objects[1],source_pose=Pose(.265,.2,0))))
    assert validate_source(bad,"full").valid
    assert not validate_source(bad,"no_op").valid
    moved = with_targets(simple_scene,{"o1":Pose(.201,.2,0),"o2":Pose(.28,.2,0)},("o1","o2"))
    assert "locked_pose_changed" in validate_layout(simple_scene,moved,simple_instruction,"no_op").errors


def test_identity_corruption(simple_scene,simple_instruction):
    target = same(simple_scene)
    corrupted = replace(target,objects=(replace(target.objects[0],color="green"),target.objects[1]))
    assert "source_target_identity" in validate_layout(simple_scene,corrupted,simple_instruction,"full").errors
    wrong_id = replace(target,objects=(replace(target.objects[0],object_id="extra"),target.objects[1]))
    assert "object_preservation" in validate_layout(simple_scene,wrong_id,simple_instruction,"full").errors


def test_color_row_semantics(simple_scene,simple_instruction):
    ins = replace(simple_instruction,group_by="color")
    assert "mixed_color_row" in validate_layout(simple_scene,same(simple_scene),ins,"full").errors
    grouped = with_targets(simple_scene,{"o1":Pose(.3,.235,0),"o2":Pose(.3,.165,0)})
    assert validate_layout(simple_scene,grouped,ins,"full").valid


def test_locked_partial_and_region(simple_scene,simple_instruction):
    ins = replace(simple_instruction,selected_ids=("o1",),locked_ids=("o2",),region="left")
    assert validate_layout(simple_scene,same(simple_scene,("o2",)),ins,"partial").valid
    wrong = with_targets(simple_scene,{"o1":Pose(.45,.2,0),"o2":Pose(.28,.2,0)},("o2",))
    assert "requested_region" in validate_layout(simple_scene,wrong,ins,"partial").errors


def test_rotated_envelope_independent_geometry(simple_scene):
    import math
    rotated = replace(simple_scene.objects[0],source_pose=Pose(.025,.2,math.pi/4))
    assert "boundary" in geometry((rotated,),"source_pose",.01,.01).errors
