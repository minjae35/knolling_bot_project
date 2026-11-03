"""M2: seeded placement; RNG is local and independent from rendering."""
import math
import random
from .schema import Scene, ObjectState, Pose, COLORS, digest
from .placement_geometry import safe_placement


class SamplingFailure(RuntimeError):
    pass


def generate_scene(assets, seed, split, count=4, mode="full", family_id=None):
    if not 2 <= count <= 8 or len(assets) < count:
        raise ValueError("unsupported object count / insufficient assets")
    if mode not in ("full","partial","no_op"):
        raise ValueError("unknown mode")
    rng = random.Random(seed)
    candidates = rng.sample(list(assets),len(assets))
    objects,placed,appearance = [],[],set()
    for asset in candidates:
        color = rng.choice(COLORS)
        signature = (asset.shape,asset.a_m,asset.b_m,color)
        if signature in appearance:
            continue
        for _ in range(2000):
            pose = Pose(rng.uniform(.04,.56),rng.uniform(.04,.36),rng.uniform(-math.pi/2,math.pi/2))
            if safe_placement(asset,pose,placed,gap=.005 if mode == "full" else .012,margin=.012):
                objects.append(ObjectState(f"o{len(objects)+1:02d}",asset,color,pose))
                placed.append((asset,pose))
                appearance.add(signature)
                break
        else:
            raise SamplingFailure("source placement budget exhausted")
        if len(objects) == count:
            return Scene(family_id or "sf_"+digest([seed,split])[:20],seed,split,tuple(objects))
    raise SamplingFailure("not enough visually distinct objects")
