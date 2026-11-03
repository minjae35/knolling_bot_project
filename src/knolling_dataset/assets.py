"""M1: original procedural solids, extruded from canonical CCW footprints.

Asset families ignore color and uniform physical scale. A new aspect/height
ratio is a new geometry family; the three shape CLASSES are not family IDs.
"""
from dataclasses import asdict
import math
import random
from .schema import Asset, SHAPES, SPLITS, digest


def geometry_fingerprint(asset):
    return digest({"shape":asset.shape, "height_ratio":round(asset.h_m/asset.a_m,9),
                   "footprint":[[round(x/asset.a_m,9),round(y/asset.a_m,9)]
                                for x,y in asset.footprint]})


def asset_split(family_id):
    bucket = int(digest(family_id)[:8],16) % 100
    return SPLITS[0 if bucket < 70 else 1 if bucket < 80 else 2 if bucket < 90 else 3]


def make_asset(shape, a_mm, b_mm, h_mm):
    a,b,h = a_mm/1000,b_mm/1000,h_mm/1000
    x,y = a/2,b/2
    if shape == "box":
        points = ((-x,-y),(x,-y),(x,y),(-x,y))
    elif shape == "chamfer":
        d = .18*b
        points = ((-x+d,-y),(x-d,-y),(x,-y+d),(x,y-d),
                  (x-d,y),(-x+d,y),(-x,y-d),(-x,-y+d))
    elif shape == "capsule":
        points = tuple((side*(x-y)+y*math.cos(t),y*math.sin(t))
                       for side,start in ((1,-math.pi/2),(-1,math.pi/2))
                       for t in [start+k*math.pi/16 for k in range(17)])
    else:
        raise ValueError("unknown shape")
    provisional = Asset("pending","pending",shape,a,b,h,points)
    family = "af_"+geometry_fingerprint(provisional)[:20]
    asset_id = "asset_"+digest([shape,a_mm,b_mm,h_mm])[:20]
    from dataclasses import replace
    area = abs(sum(px*qy-qx*py for (px,py),(qx,qy) in zip(points,points[1:]+points[:1])))/2
    return replace(provisional,asset_id=asset_id,family_id=family,mass_kg=area*h*625)


def build_registry(seed=1201, families_per_split=32):
    if families_per_split < 8:
        raise ValueError("at least eight families per split required")
    rng = random.Random(seed)
    pools = {split:[] for split in SPLITS}
    seen = set()
    for _ in range(200000):
        a,b = rng.randint(40,80),rng.randint(25,50)
        if a/b < 1.3:
            continue
        asset = make_asset(rng.choice(SHAPES),a,b,rng.randint(5,20))
        split = asset_split(asset.family_id)
        if len(pools[split]) == families_per_split or asset.family_id in seen:
            continue
        pools[split].append(asset)
        seen.add(asset.family_id)
        if all(len(p) == families_per_split for p in pools.values()):
            return {split:tuple(sorted(p,key=lambda a:a.asset_id)) for split,p in pools.items()}
    raise RuntimeError("registry sampling budget exhausted")


def registry_dict(registry):
    return {split:[asdict(a) for a in assets] for split,assets in registry.items()}
