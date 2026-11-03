import pytest
from shapely.geometry import Polygon
from knolling_dataset.assets import make_asset, build_registry, geometry_fingerprint, asset_split
from knolling_dataset.schema import SHAPES


@pytest.mark.parametrize("shape",SHAPES)
def test_solid_footprints_and_uniform_scale_family(shape):
    a,b = make_asset(shape,60,30,10),make_asset(shape,72,36,12)
    p = Polygon(a.footprint)
    assert p.is_valid and p.exterior.is_ccw and p.area > 0
    assert p.bounds == pytest.approx((-.03,-.015,.03,.015))
    assert geometry_fingerprint(a) == geometry_fingerprint(b)
    assert a.family_id == b.family_id and a.asset_id != b.asset_id
    assert a.family_id == make_asset(shape,60,30,15).family_id


def test_registry_reproducible_disjoint_and_diverse():
    first = build_registry()
    assert first == build_registry()
    all_families = []
    for split, assets in first.items():
        assert len(assets) == 32
        assert {a.shape for a in assets} == set(SHAPES)
        assert all(asset_split(a.family_id) == split for a in assets)
        all_families.extend(a.family_id for a in assets)
    assert len(set(all_families)) == len(all_families)


def test_reject_unsupported_geometry():
    with pytest.raises(ValueError):
        make_asset("box",40,40,10)
