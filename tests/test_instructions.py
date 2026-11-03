from dataclasses import replace
import pytest
from knolling_dataset.instructions import generate_legacy_instruction as generate_instruction, negative_instruction
from knolling_dataset.schema import check_instruction


@pytest.mark.parametrize("language",["ko","en"])
@pytest.mark.parametrize("mode",["full","partial","keep_all"])
def test_grammar_and_grounding(simple_scene,language,mode):
    ins = generate_instruction(simple_scene,6,mode,language=language,group_by="color",gap_m=.030)
    assert ins == generate_instruction(simple_scene,6,mode,language=language,group_by="color",gap_m=.030)
    check_instruction(simple_scene,ins)
    assert not any(oid in ins.text for oid in ins.selected_ids)
    if language == "ko":
        assert "만를" not in ins.text
        if mode == "full":
            assert ins.text.startswith("모든 물체를 ")
    if mode == "partial":
        assert len(ins.selected_ids) == len(ins.locked_ids) == 1
    elif mode == "keep_all":
        assert ins.keep_all and not ins.selected_ids
    else:
        assert "60" in ins.text and "30" in ins.text


def test_monochrome_leftmost_and_negative(simple_scene):
    scene = replace(simple_scene,objects=tuple(replace(o,color="red") for o in simple_scene.objects))
    ins = generate_instruction(scene,1,"partial",language="en")
    assert ins.selected_ids == ("o1",) and "leftmost" in ins.text
    for status in ("unsupported","ambiguous"):
        assert negative_instruction(scene,status).expected_status == status
