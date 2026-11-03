from dataclasses import asdict,replace
import pytest
from knolling_dataset.instructions import generate_legacy_instruction as generate_instruction,clarify_instruction
from knolling_dataset.builder import _object_annotations
from knolling_dataset.schema import with_targets,Pose,scene_dict


@pytest.mark.parametrize("language",["ko","en"])
@pytest.mark.parametrize("group",["none","color"])
@pytest.mark.parametrize("gap",[.015,.020,.030])
def test_clarification_is_semantically_identical(simple_scene,language,group,gap):
    legacy=generate_instruction(simple_scene,3,"partial",language=language,group_by=group,gap_m=gap)
    new=clarify_instruction(legacy)
    assert new==clarify_instruction(new)
    assert {k:v for k,v in asdict(legacy).items() if k!="text"}=={k:v for k,v in asdict(new).items() if k!="text"}
    assert ("행 띠" if language=="ko" else "row band") in new.text
    assert str(round(gap*1000)) in new.text
    assert new==generate_instruction(simple_scene,3,"partial",language=language,group_by=group,gap_m=gap,contract="row_edges_v1")


def test_keep_all_and_unknown_text(simple_scene):
    ins=generate_instruction(simple_scene,3,"keep_all")
    assert clarify_instruction(ins)==ins
    with pytest.raises(ValueError):
        clarify_instruction(replace(generate_instruction(simple_scene,2),text="unrecognized phrasing"))


def test_annotation_joins_ids_not_array_positions(simple_scene):
    target=with_targets(simple_scene,{"o1":Pose(.1,.2,0),"o2":Pose(.4,.2,0)})
    record={"source":scene_dict(simple_scene),"target":scene_dict(replace(target,objects=tuple(reversed(target.objects))))}
    objects=[{"object_id":oid,"instance_id":i,"bbox_xyxy_px":[i,0,i+2,2]} for i,oid in enumerate(("o1","o2"),1)]
    labels=_object_annotations(record,{"objects":objects},{"objects":list(reversed(objects))})
    assert labels[0]["target_center_uv"][0]==pytest.approx(.1/.6)
    assert labels[1]["target_center_uv"][0]==pytest.approx(.4/.6)
