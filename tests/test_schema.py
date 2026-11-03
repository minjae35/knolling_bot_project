import math
import pytest
from knolling_dataset.schema import *


def test_coordinate_round_trips_and_axes():
    for p in ((0,0),(.6,.4),(.3,.2),(.17,.33)):
        assert to_meters(*to_normalized(*p)) == pytest.approx(p)
        assert from_pixel(*to_pixel(*p)) == pytest.approx(p)
    assert to_pixel(0, .4) == (-.5, -.5)
    assert to_pixel(.6, 0) == (575.5, 383.5)


def test_finite_and_periodic_angle():
    with pytest.raises(ValueError):
        Pose(float('nan'), 0, 0)
    assert angle_error(0, math.pi) == pytest.approx(0)
    assert angle_error(math.pi/2-.01, -math.pi/2+.01) == pytest.approx(.02)


def test_json_hash_stable_and_strict():
    assert digest({'a':1,'b':2}) == digest({'b':2,'a':1})
    with pytest.raises(ValueError):
        canonical_json({'x': float('nan')})
