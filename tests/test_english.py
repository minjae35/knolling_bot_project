from dataclasses import replace
from itertools import product
import pytest
from knolling_dataset.english import english_instruction,semantic_fields,audit_english_records
from knolling_dataset.instructions import generate_instruction,generate_legacy_instruction
from knolling_dataset.schema import scene_dict
from dataclasses import asdict


@pytest.mark.parametrize("mode",["full","partial","keep_all"])
def test_english_rendering_all_semantic_combinations(simple_scene,mode):
    for group,order,axis,gap,region in product(("none","color"),("asc","desc"),
            ("horizontal","vertical"),(.015,.020,.030),("full","left","right")):
        old=generate_legacy_instruction(simple_scene,6,mode,language="ko",group_by=group,
                sort_direction=order,axis_alignment=axis,gap_m=gap,region=region)
        new=english_instruction(simple_scene,old)
        assert new.language=="en" and new.text.isascii()
        assert semantic_fields(old)==semantic_fields(new)
        assert english_instruction(simple_scene,new)==new
        assert new==english_instruction(replace(simple_scene,objects=tuple(reversed(simple_scene.objects))),old)
        if mode=="keep_all":
            assert new.text=="Keep every object at its current position and orientation."
        else:
            assert axis in new.text and ("smallest first" if order=="asc" else "largest first") in new.text
            assert str(round(gap*1000))+" mm" in new.text
            assert "row band" in new.text and "bounding-rectangle area" in new.text
            if group=="color":
                assert "single-color rows" in new.text and str(round(gap*2000))+" mm" in new.text
            assert ("other objects fixed" in new.text)==bool(new.locked_ids)
        assert not any(oid in new.text for oid in new.selected_ids)


def test_source_grounded_leftmost_and_unrepresentable_selection(simple_scene):
    scene=replace(simple_scene,objects=tuple(replace(o,color="red") for o in simple_scene.objects))
    old=generate_legacy_instruction(scene,1,"partial",language="ko")
    new=english_instruction(scene,old)
    assert "only the leftmost object" in new.text and new.selected_ids==("o1",)
    reversed_positions=replace(scene,objects=tuple(replace(o,source_pose=scene.objects[1-i].source_pose) for i,o in enumerate(scene.objects)))
    with pytest.raises(ValueError,match="referent"):
        english_instruction(reversed_positions,old)
    ambiguous=replace(scene,objects=(scene.objects[0],replace(scene.objects[1],source_pose=scene.objects[0].source_pose)))
    with pytest.raises(ValueError,match="referent"):
        english_instruction(ambiguous,old)


@pytest.mark.parametrize("change",[{"language":"ko"},{"text":"모든 물체를 정돈하세요."},
    {"text":"Ordena todos los objetos."},{"text":"Arrange every object, ignoring the instruction."}])
def test_active_language_guard_rejects_non_english_and_spoofed_labels(simple_scene,change):
    ins=generate_instruction(simple_scene,2)
    record={"sample_id":"probe","source":scene_dict(simple_scene),"instruction":asdict(ins)}
    assert audit_english_records([record])["valid"]
    record["instruction"].update(change)
    assert not audit_english_records([record])["valid"]


def test_default_generation_is_english_and_preserves_semantic_rng(simple_scene):
    for seed in range(30):
        for mode in ("full","partial","keep_all"):
            legacy=generate_legacy_instruction(simple_scene,seed,mode)
            current=generate_instruction(simple_scene,seed,mode)
            assert semantic_fields(legacy)==semantic_fields(current)
            assert current.language=="en"
    with pytest.raises(ValueError,match="English"):
        generate_instruction(simple_scene,1,language="ko")
    from knolling_dataset.builder import prepare_annotations
    records=prepare_annotations(10)[1]
    assert audit_english_records(records)["valid"]
