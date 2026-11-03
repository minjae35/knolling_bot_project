import os
from pathlib import Path
import pytest
from knolling_dataset.assets import build_registry
from knolling_dataset.scenes import generate_scene
from knolling_dataset.rendering import make_job,render_batch,validate_render


@pytest.mark.skipif(os.environ.get("RUN_BLENDER_TESTS") != "1",reason="set RUN_BLENDER_TESTS=1 for real Blender integration")
def test_actual_blender_masks_and_coordinates(tmp_path):
    scene = generate_scene(build_registry()["train"],179,"train",8)
    out = tmp_path/"source"
    repeat = tmp_path/"repeat"
    render_batch([make_job(scene,"source_pose",out,"integration"),
                  make_job(scene,"source_pose",repeat,"repeat")],tmp_path/"work")
    report = validate_render(scene,"source_pose",out)
    assert report["valid"],report
    for name in ("rgb.png","instance_mask.png"):
        assert (out/name).read_bytes() == (repeat/name).read_bytes()


def test_missing_target_rejected(simple_scene,tmp_path):
    with pytest.raises(ValueError):
        make_job(simple_scene,"target_pose",tmp_path,"bad")
