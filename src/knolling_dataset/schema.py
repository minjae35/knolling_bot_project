"""M0: immutable contracts; JSON is data, never executable instructions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

WIDTH_M, HEIGHT_M = 0.600, 0.400
IMAGE_WIDTH, IMAGE_HEIGHT = 576, 384
FINAL_GAP_M = FINAL_MARGIN_M = 0.010
TEACHER_MARGIN_M = 0.020
GEOMETRY_TOL_M = 0.0001
COLORS = ("red", "green", "blue")
SHAPES = ("box", "chamfer", "capsule")
SPLITS = ("train", "validation", "development", "test")
REGIONS = {
    "full": (0.02, 0.02, 0.58, 0.38),
    "left": (0.02, 0.02, 0.28, 0.38),
    "right": (0.32, 0.02, 0.58, 0.38),
}


def finite(*values: float) -> None:
    if not all(math.isfinite(x) for x in values):
        raise ValueError("non-finite numeric value")


def wrap_angle(theta_rad: float) -> float:
    finite(theta_rad)
    return (theta_rad + math.pi / 2) % math.pi - math.pi / 2


def angle_error(a_rad: float, b_rad: float) -> float:
    return abs(wrap_angle(a_rad - b_rad))


def to_normalized(x_m: float, y_m: float) -> tuple[float, float]:
    finite(x_m, y_m)
    return x_m / WIDTH_M, y_m / HEIGHT_M


def to_meters(u: float, v: float) -> tuple[float, float]:
    finite(u, v)
    return u * WIDTH_M, v * HEIGHT_M


def to_pixel(x_m: float, y_m: float) -> tuple[float, float]:
    u, v = to_normalized(x_m, y_m)
    return IMAGE_WIDTH * u - 0.5, IMAGE_HEIGHT * (1 - v) - 0.5


def from_pixel(x_px: float, y_px: float) -> tuple[float, float]:
    return to_meters((x_px + 0.5) / IMAGE_WIDTH, 1 - (y_px + 0.5) / IMAGE_HEIGHT)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    theta_rad: float

    def __post_init__(self):
        finite(self.x_m, self.y_m, self.theta_rad)


@dataclass(frozen=True)
class Asset:
    asset_id: str
    family_id: str
    shape: str
    a_m: float
    b_m: float
    h_m: float
    footprint: tuple[tuple[float, float], ...]
    mass_kg: float = 0.030
    friction: float = 0.6
    provenance: str = "procedural-original"
    license: str = "CC0-1.0"

    def __post_init__(self):
        finite(self.a_m, self.b_m, self.h_m, self.mass_kg, self.friction)
        if self.shape not in SHAPES or not (0.040 <= self.a_m <= 0.080):
            raise ValueError("invalid shape or major dimension")
        # Unit conversion can put the exact 52/40 boundary one ULP below 1.3.
        # This tolerance is dimensionless and applies only to this comparison.
        if not (0.025 <= self.b_m <= 0.050 and (
            self.a_m / self.b_m >= 1.3 or math.isclose(
                self.a_m / self.b_m, 1.3, rel_tol=0.0, abs_tol=4 * math.ulp(1.3)
            )
        )):
            raise ValueError("invalid minor dimension or aspect ratio")
        if not (0.005 <= self.h_m <= 0.020) or len(self.footprint) < 4:
            raise ValueError("invalid height or footprint")
        if not self.asset_id or not self.family_id or self.mass_kg <= 0 or self.friction < 0:
            raise ValueError("invalid asset metadata")
        for x, y in self.footprint:
            finite(x, y)
            if abs(x) > self.a_m / 2 + 1e-9 or abs(y) > self.b_m / 2 + 1e-9:
                raise ValueError("footprint outside standard envelope")


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    asset: Asset
    color: str
    source_pose: Pose
    target_pose: Pose | None = None
    locked: bool = False
    move_label: bool = False

    def __post_init__(self):
        if not self.object_id or self.color not in COLORS:
            raise ValueError("invalid object identity/color")


@dataclass(frozen=True)
class Scene:
    family_id: str
    seed: int
    split: str
    objects: tuple[ObjectState, ...]

    def __post_init__(self):
        ids = [o.object_id for o in self.objects]
        if not (2 <= len(ids) <= 8) or len(set(ids)) != len(ids):
            raise ValueError("scene needs 2–8 uniquely identified objects")
        if self.split not in SPLITS or not self.family_id:
            raise ValueError("invalid scene split/family")


@dataclass(frozen=True)
class Instruction:
    text: str
    language: str
    selected_ids: tuple[str, ...]
    locked_ids: tuple[str, ...]
    group_by: str = "none"
    sort_direction: str = "asc"
    axis_alignment: str = "horizontal"
    gap_m: float = 0.020
    region: str = "full"
    keep_all: bool = False
    preference_text: str = ""
    expected_status: str = "ok"
    reason: str | None = None

    def __post_init__(self):
        if self.language not in ("ko", "en") or not self.text.strip():
            raise ValueError("invalid language/text")
        if self.group_by not in ("none", "color") or self.sort_direction not in ("asc", "desc"):
            raise ValueError("unsupported ordering")
        if self.axis_alignment not in ("horizontal", "vertical") or self.region not in REGIONS:
            raise ValueError("unsupported axis/region")
        if self.gap_m not in (0.015, 0.020, 0.030):
            raise ValueError("unsupported gap")
        if self.expected_status not in ("ok", "ambiguous", "infeasible", "unsupported"):
            raise ValueError("unsupported status")
        if len(set(self.selected_ids)) != len(self.selected_ids) or len(set(self.locked_ids)) != len(self.locked_ids):
            raise ValueError("duplicate instruction IDs")
        if set(self.selected_ids) & set(self.locked_ids):
            raise ValueError("selected and locked sets overlap")


def check_instruction(scene: Scene, instruction: Instruction) -> None:
    ids = {o.object_id for o in scene.objects}
    if set(instruction.selected_ids) | set(instruction.locked_ids) != ids:
        raise ValueError("instruction must partition the observed objects")
    if instruction.keep_all and (instruction.selected_ids or set(instruction.locked_ids) != ids):
        raise ValueError("keep-all must lock every object")


def scene_dict(scene: Scene) -> dict:
    return asdict(scene)


def scene_from_dict(data: dict) -> Scene:
    objects = []
    for raw in data["objects"]:
        a = dict(raw["asset"])
        a["footprint"] = tuple(tuple(p) for p in a["footprint"])
        objects.append(ObjectState(
            raw["object_id"], Asset(**a), raw["color"], Pose(**raw["source_pose"]),
            Pose(**raw["target_pose"]) if raw.get("target_pose") else None,
            raw.get("locked", False), raw.get("move_label", False)))
    return Scene(data["family_id"], data["seed"], data["split"], tuple(objects))


def instruction_from_dict(data: dict) -> Instruction:
    values = dict(data)
    for key in ("selected_ids", "locked_ids"):
        values[key] = tuple(values[key])
    return Instruction(**values)


def with_targets(scene: Scene, targets: dict[str, Pose], locked_ids=()) -> Scene:
    if set(targets) != {o.object_id for o in scene.objects}:
        raise ValueError("target IDs must equal source IDs")
    locked = set(locked_ids)
    return replace(scene, objects=tuple(replace(
        o, target_pose=targets[o.object_id], locked=o.object_id in locked,
        move_label=(math.hypot(targets[o.object_id].x_m-o.source_pose.x_m,
                               targets[o.object_id].y_m-o.source_pose.y_m) > 0.003
                    or angle_error(targets[o.object_id].theta_rad, o.source_pose.theta_rad) > math.radians(3)))
        for o in scene.objects))
