"""M5. Independent Shapely oracle. Never imports scene sampling or the teacher.

Dataset validation is stricter than prediction evaluation: locked poses and
source metadata must be EXACT copies. All geometry here is measured in metres.
"""
from dataclasses import dataclass, field
import math
from shapely.geometry import Polygon, box
from .schema import (Scene, Instruction, Pose, REGIONS, WIDTH_M, HEIGHT_M,
                     GEOMETRY_TOL_M as EPS, angle_error, check_instruction)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def valid(self):
        return not self.errors

    def fail(self, code):
        if code not in self.errors:
            self.errors.append(code)

    def as_dict(self):
        return {"valid": self.valid, "errors": self.errors, "metrics": self.metrics}


def polygon(obj, pose: Pose, envelope=True):
    a, b = obj.asset.a_m / 2, obj.asset.b_m / 2
    points = ((-a,-b),(a,-b),(a,b),(-a,b)) if envelope else obj.asset.footprint
    c, s = math.cos(pose.theta_rad), math.sin(pose.theta_rad)
    return Polygon([(pose.x_m+c*x-s*y, pose.y_m+s*x+c*y) for x,y in points])


def geometry(objects, pose_field, gap, margin):
    report = ValidationReport()
    polys = []
    clearances, margins = [], []
    inner = box(margin-EPS, margin-EPS, WIDTH_M-margin+EPS, HEIGHT_M-margin+EPS)
    for obj in objects:
        pose = getattr(obj, pose_field)
        if pose is None:
            report.fail("missing_pose")
            continue
        footprint = Polygon(obj.asset.footprint)
        if not footprint.is_valid or footprint.area <= 0:
            report.fail("invalid_footprint")
        p = polygon(obj, pose)
        polys.append(p)
        xmin,ymin,xmax,ymax = p.bounds
        margins.append(min(xmin,ymin,WIDTH_M-xmax,HEIGHT_M-ymax))
        if not inner.covers(p):
            report.fail("boundary")
    for i,p in enumerate(polys):
        for q in polys[i+1:]:
            distance = p.distance(q)
            clearances.append(distance)
            if p.intersection(q).area > 1e-12:
                report.fail("collision")
            if distance < gap-EPS:
                report.fail("clearance")
    report.metrics.update(min_clearance_m=min(clearances, default=None),
                          min_boundary_margin_m=min(margins, default=None))
    return report


def validate_source(scene: Scene, mode="full"):
    if mode not in ("full", "partial", "no_op"):
        raise ValueError("unknown mode")
    return geometry(scene.objects, "source_pose", .010 if mode == "no_op" else .005, .010)


def inferred_rows(objects):
    """Row membership uses coordinates, not teacher-supplied row labels."""
    rows = []
    for obj in sorted(objects, key=lambda o: (-o.target_pose.y_m, o.target_pose.x_m)):
        if not rows or rows[-1][0].target_pose.y_m-obj.target_pose.y_m > .010 + EPS:
            rows.append([])
        rows[-1].append(obj)
    return [sorted(row, key=lambda o: o.target_pose.x_m) for row in rows]


def validate_layout(source: Scene, target: Scene, instruction: Instruction, mode: str):
    if mode not in ("full", "partial", "no_op"):
        raise ValueError("unknown mode")
    report = ValidationReport()
    try:
        check_instruction(source, instruction)
    except ValueError:
        report.fail("instruction_partition")
        return report
    if instruction.expected_status != "ok":
        report.fail("unexpected_positive_target")
    original = {o.object_id:o for o in source.objects}
    output = {o.object_id:o for o in target.objects}
    if original.keys() != output.keys():
        report.fail("object_preservation")
        return report
    if (source.family_id,source.seed,source.split) != (target.family_id,target.seed,target.split):
        report.fail("scene_identity")
    expected_locks = set(original) if mode == "no_op" else set(instruction.locked_ids)
    for oid, obj in output.items():
        src = original[oid]
        if (obj.asset, obj.color, obj.source_pose) != (src.asset, src.color, src.source_pose):
            report.fail("source_target_identity")
        if obj.locked != (oid in expected_locks):
            report.fail("lock_label")
        if obj.target_pose is None:
            report.fail("missing_pose")
            continue
        if oid in expected_locks and obj.target_pose != src.source_pose:
            report.fail("locked_pose_changed")
        moved = (math.hypot(obj.target_pose.x_m-src.source_pose.x_m,
                            obj.target_pose.y_m-src.source_pose.y_m) > .003 or
                 angle_error(obj.target_pose.theta_rad,src.source_pose.theta_rad) > math.radians(3))
        if obj.move_label != moved:
            report.fail("move_label")
    if "missing_pose" in report.errors:
        return report
    g = geometry(target.objects,"target_pose",.010,.010)
    report.errors.extend(e for e in g.errors if e not in report.errors)
    report.metrics.update(g.metrics)
    if instruction.keep_all:
        if mode != "no_op":
            report.fail("keep_all_not_no_op")
        return report
    selected = [output[oid] for oid in instruction.selected_ids]
    region = box(*REGIONS[instruction.region]).buffer(EPS, join_style=2)
    expected_angle = 0 if instruction.axis_alignment == "horizontal" else math.pi/2
    for obj in selected:
        if angle_error(obj.target_pose.theta_rad,expected_angle) > math.radians(5):
            report.fail("axis_alignment")
        if not region.covers(polygon(obj,obj.target_pose)):
            report.fail("requested_region")
    rows = inferred_rows(selected)
    report.metrics["row_object_ids"] = [[o.object_id for o in row] for row in rows]
    row_colors = []
    for row in rows:
        if instruction.group_by == "color":
            colors = {o.color for o in row}
            if len(colors) != 1:
                report.fail("mixed_color_row")
            row_colors.append(row[0].color)
        for left,right in zip(row,row[1:]):
            lp,rp = polygon(left,left.target_pose), polygon(right,right.target_pose)
            gap = rp.bounds[0]-lp.bounds[2]
            if abs(gap-instruction.gap_m) > .005+EPS:
                report.fail("horizontal_gap")
    if instruction.group_by == "color":
        runs = [c for i,c in enumerate(row_colors) if i == 0 or c != row_colors[i-1]]
        if len(runs) != len(set(runs)):
            report.fail("noncontiguous_color_group")
    for upper,lower in zip(rows,rows[1:]):
        upper_bottom = min(polygon(o,o.target_pose).bounds[1] for o in upper)
        lower_top = max(polygon(o,o.target_pose).bounds[3] for o in lower)
        multiplier = 2 if instruction.group_by == "color" and upper[0].color != lower[0].color else 1
        if abs(upper_bottom-lower_top-multiplier*instruction.gap_m) > .005+EPS:
            report.fail("vertical_gap")
    ordered = [o for row in rows for o in row]
    groups = [[o for o in ordered if o.color == c] for c in ("red","green","blue")] if instruction.group_by == "color" else [ordered]
    for group in groups:
        areas = [o.asset.a_m*o.asset.b_m for o in group]
        for a,b in zip(areas,areas[1:]):
            if (instruction.sort_direction == "asc" and a > 1.05*b+1e-12 or
                    instruction.sort_direction == "desc" and b > 1.05*a+1e-12):
                report.fail("size_order")
    return report
