from dataclasses import replace
import math
import pytest
from knolling_dataset.assets import make_asset,build_registry
from knolling_dataset.schema import Scene,ObjectState,Pose,Instruction,with_targets,angle_error
from knolling_dataset.scenes import generate_scene
from knolling_dataset.instructions import generate_instruction
from knolling_dataset.teacher import solve
from knolling_dataset.teacher_balanced import templates,partition_sizes,observed_rows
from knolling_dataset.validator import validate_layout
from knolling_dataset.policies import BALANCED,LEGACY,require_single_policy,policy_metadata


def poses(result):
    return {o.object_id:o.target_pose for o in result.target.objects}


def test_partitions_dedup_and_eight_vertical_unbalanced_colors():
    assert partition_sizes(5,4,False)==[4,1]
    assert partition_sizes(5,4,True)==[3,2]
    scene=generate_scene(build_registry()["train"],42,"train",8)
    scene=replace(scene,objects=tuple(replace(o,color=c) for o,c in zip(scene.objects,["red"]*5+["green"]*2+["blue"])))
    ins=generate_instruction(scene,2,"full",group_by="color",axis_alignment="vertical",region="full",gap_m=.015)
    ts=templates(scene,ins)
    sizes=[[len(r) for r in t["rows"]] for t in ts]
    assert [4,1,2,1] in sizes and [3,2,2,1] in sizes and [5,2,1] in sizes
    assert len(ts)<=16 and len({t["key"] for t in ts})==len(ts)
    result=solve(scene,ins,policy=BALANCED)
    assert result.status=="ok"
    assert validate_layout(scene,result.target,ins,result.mode).valid
    for t in ts:
        for row in t["rows"]:
            assert len({o.color for o in row})==1


def test_full_shape_selection_and_invariance():
    scene=generate_scene(build_registry()["train"],44,"train",5)
    ins=generate_instruction(scene,2,"full",group_by="none",region="full")
    result=solve(scene,ins,policy=BALANCED)
    assert result.status=="ok"
    feasible=[]
    for t in templates(scene,ins):
        target=with_targets(scene,{oid:Pose(.3+p.x_m,.2+p.y_m,p.theta_rad) for oid,p in t["relative"].items()})
        if validate_layout(scene,target,ins,"full").valid:
            feasible.append((round(t["J_shape"],12),round(t["width"]*t["height"],12),len(t["rows"]),t["key"]))
    assert (round(result.audit["J_shape"],12),round(result.audit["block_width_m"]*result.audit["block_height_m"],12),len(result.rows),result.audit["template_key"])==min(feasible)
    mutated=replace(scene,objects=tuple(replace(o,asset=replace(o.asset,asset_id="alias_"+o.object_id)) for o in reversed(scene.objects)))
    paraphrase=replace(ins,text="동일한 의미의 한국어 문장",language="ko",selected_ids=tuple(reversed(ins.selected_ids)))
    assert poses(solve(mutated,paraphrase,policy=BALANCED))==poses(result)
    current=replace(scene,objects=tuple(replace(o,source_pose=o.target_pose,target_pose=None,locked=False,move_label=False) for o in result.target.objects))
    noop=solve(current,ins,policy=BALANCED)
    assert noop.mode=="no_op" and noop.rows==[] and noop.relations=={}
    assert all(o.source_pose==o.target_pose and o.locked and not o.move_label for o in noop.target.objects)


def test_partial_global_minimum_with_blocked_center(simple_scene):
    scene=replace(simple_scene,objects=(replace(simple_scene.objects[0],source_pose=Pose(.30,.12,.2)),
                                     replace(simple_scene.objects[1],source_pose=Pose(.30,.20,0))))
    ins=Instruction("Move the red object vertically; keep blue fixed.","en",("o1",),("o2",),axis_alignment="vertical")
    result=solve(scene,ins,policy=BALANCED)
    assert result.status=="ok" and poses(result)["o2"]==scene.objects[1].source_pose
    # Independently enumerate the entire lattice, including corner separations.
    valid=[]
    for ix in range(120):
        for iy in range(80):
            p=Pose(ix*.005,iy*.005,-math.pi/2)
            target=with_targets(scene,{"o1":p,"o2":scene.objects[1].source_pose},("o2",))
            if validate_layout(scene,target,ins,"partial").valid:
                cost=(p.x_m-.30)**2+(p.y_m-.12)**2
                valid.append((round(cost,12),-p.y_m,p.x_m))
    chosen=poses(result)["o1"]
    assert (round(result.audit["J_move"],12),-chosen.y_m,chosen.x_m)==min(valid)
    fail=solve(scene,ins,search_budget=0,policy=BALANCED)
    assert fail.status=="generation_failure" and fail.reason=="search_budget"


def test_existing_rows_candidate_and_scatter_rejection():
    objects=[]
    for row,y in ((range(2),.28),(range(2,5),.20)):
        widths=[(.060+i*.001) for i in row]; left=.20-(sum(widths)+(len(widths)-1)*.02)/2
        for i,w in zip(row,widths):
            objects.append(ObjectState(str(i),make_asset("box",60+i,30,10),"red",Pose(left+w/2,y,0))); left+=w+.02
    objects.append(ObjectState("lock",make_asset("box",70,30,10),"blue",Pose(.5,.3,0)))
    scene=Scene("observed",1,"train",tuple(objects))
    ins=Instruction("Rotate red objects; keep blue fixed.","en",tuple(str(i) for i in range(5)),("lock",),axis_alignment="vertical",gap_m=.015)
    assert [len(r) for r in observed_rows(scene,ins)]==[2,3]
    assert any(t["kind"]=="observed" and [len(r) for r in t["rows"]]==[2,3] for t in templates(scene,ins))
    scatter=replace(scene,objects=tuple(replace(o,source_pose=replace(o.source_pose,theta_rad=.25)) for o in scene.objects))
    assert observed_rows(scatter,ins)==[]


def test_policy_rejection_and_legacy_exact_source():
    import hashlib,subprocess
    from pathlib import Path
    old=subprocess.check_output(["git","show","69cf565:src/knolling_dataset/teacher.py"])
    assert hashlib.sha256(old).digest()==hashlib.sha256(Path("src/knolling_dataset/teacher_legacy.py").read_bytes()).digest()
    assert require_single_policy([BALANCED,BALANCED])==BALANCED
    with pytest.raises(ValueError): require_single_policy([LEGACY,BALANCED])
    assert policy_metadata(LEGACY)["teacher_config_hash"]!=policy_metadata(BALANCED)["teacher_config_hash"]
