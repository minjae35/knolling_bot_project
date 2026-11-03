from knolling_dataset.assets import build_registry
from knolling_dataset.scenes import generate_scene
from knolling_dataset.validator import validate_source


def test_seeded_scenes_independent_validation():
    assets = build_registry()["train"]
    scenes = []
    for seed in range(100):
        mode = ("full","partial","no_op")[seed%3]
        scene = generate_scene(assets,seed,"train",2+seed%7,mode)
        assert validate_source(scene,mode).valid
        assert scene == generate_scene(assets,seed,"train",2+seed%7,mode)
        assert len(scene.objects) == 2+seed%7
        if mode == "partial":
            assert validate_source(scene,"no_op").valid
        scenes.append(scene)
    assert len({s.objects[0].source_pose for s in scenes}) == 100
