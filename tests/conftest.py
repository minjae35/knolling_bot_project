import pytest
from knolling_dataset.schema import Asset, ObjectState, Pose, Scene, Instruction


@pytest.fixture
def simple_scene():
    a = Asset("a", "af", "box", .06, .03, .01,
              ((-.03,-.015),(.03,-.015),(.03,.015),(-.03,.015)))
    return Scene("s", 1, "train", (
        ObjectState("o1", a, "red", Pose(.20,.20,0)),
        ObjectState("o2", a, "blue", Pose(.28,.20,0))))


@pytest.fixture
def simple_instruction():
    return Instruction("Arrange horizontally, ascending size, 20 mm gap.","en",("o1","o2"),())
