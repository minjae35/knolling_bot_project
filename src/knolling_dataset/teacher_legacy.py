"""M4 deterministic supervision. This planner is never a neural inference head."""
from dataclasses import dataclass, field
import math
from .schema import Scene, Pose, COLORS, REGIONS, check_instruction, with_targets
from .placement_geometry import inside, safe_placement
from .validator import validate_layout, validate_source, geometry


@dataclass
class TeacherResult:
    status: str
    mode: str
    target: Scene | None = None
    reason: str | None = None
    rows: list[list[str]] = field(default_factory=list)
    relations: dict = field(default_factory=dict)
    candidates_checked: int = 0


def _templates(scene, instruction):
    selected = [o for o in scene.objects if o.object_id in instruction.selected_ids]
    sign = 1 if instruction.sort_direction == "asc" else -1
    ordered = sorted(selected,key=lambda o:(sign*o.asset.a_m*o.asset.b_m,o.asset.a_m,
                     o.asset.b_m,o.color,o.source_pose.y_m,o.source_pose.x_m))
    groups = [[o for o in ordered if o.color == c] for c in COLORS] if instruction.group_by == "color" else [ordered]
    vertical = instruction.axis_alignment == "vertical"
    dims = lambda o: (o.asset.b_m,o.asset.a_m) if vertical else (o.asset.a_m,o.asset.b_m)
    templates = []
    for k in range(1,5):
        rows = [group[i:i+k] for group in groups for i in range(0,len(group),k)]
        if not rows:
            continue
        widths = [sum(dims(o)[0] for o in row)+(len(row)-1)*instruction.gap_m for row in rows]
        heights = [max(dims(o)[1] for o in row) for row in rows]
        gaps = [instruction.gap_m*(2 if instruction.group_by == "color" and up[0].color != down[0].color else 1)
                for up,down in zip(rows,rows[1:])]
        height,width = sum(heights)+sum(gaps),max(widths)
        positions = {}
        top = height/2
        for r,row in enumerate(rows):
            left = -widths[r]/2
            for obj in row:
                w,h = dims(obj)
                positions[obj.object_id] = Pose(left+w/2,top-heights[r]/2,math.pi/2 if vertical else 0)
                left += w+instruction.gap_m
            top -= heights[r]+(gaps[r] if r < len(gaps) else 0)
        score = (len(rows),round(width*height,12),round(width,9),
                 tuple(sorted((p.x_m,p.y_m) for p in positions.values())))
        templates.append((score,width,height,positions,rows,gaps))
    return sorted(templates,key=lambda t:t[0])


def _translations(width,height,region,partial):
    l,b,r,t = region
    if width > r-l+1e-9 or height > t-b+1e-9:
        return []
    center = ((l+r)/2,(b+t)/2)
    if not partial:
        return [center]
    # Centers on a global 5 mm lattice, deterministic distance / upper / left ties.
    xs = range(math.ceil((l+width/2-1e-9)/.005),math.floor((r-width/2+1e-9)/.005)+1)
    ys = range(math.ceil((b+height/2-1e-9)/.005),math.floor((t-height/2+1e-9)/.005)+1)
    return sorted(((x*.005,y*.005) for x in xs for y in ys),
                  key=lambda p:(round((p[0]-center[0])**2+(p[1]-center[1])**2,12),-p[1],p[0]))


def solve(scene, instruction, search_budget=20000):
    check_instruction(scene,instruction)
    mode = "partial" if instruction.locked_ids else "full"
    if instruction.expected_status in ("unsupported","ambiguous"):
        return TeacherResult(instruction.expected_status,mode,reason=instruction.reason)
    # Metadata must not manufacture infeasibility: require geometric proof below.
    if instruction.expected_status == "infeasible":
        raise ValueError("request feasibility must be established by geometry, not a label")
    current = with_targets(scene,{o.object_id:o.source_pose for o in scene.objects},
                           [o.object_id for o in scene.objects])
    if validate_layout(scene,current,instruction,"no_op").valid:
        return TeacherResult("ok","no_op",current)
    if instruction.keep_all:
        return TeacherResult("infeasible","no_op",reason="keep_all_violates_final_geometry")
    locked = [o for o in scene.objects if o.object_id in instruction.locked_ids]
    if not geometry(locked,"source_pose",.010,.010).valid:
        return TeacherResult("infeasible",mode,reason="immutable_objects_violate_final_geometry")
    if not validate_source(scene,"full").valid:
        return TeacherResult("generation_failure",mode,reason="invalid_source_scene")
    region = REGIONS[instruction.region]
    vertical = instruction.axis_alignment == "vertical"
    selected = [o for o in scene.objects if o.object_id in instruction.selected_ids]
    if any((o.asset.b_m if vertical else o.asset.a_m) > region[2]-region[0] or
           (o.asset.a_m if vertical else o.asset.b_m) > region[3]-region[1] for o in selected):
        return TeacherResult("infeasible",mode,reason="object_larger_than_requested_region")
    checked = 0
    for _,width,height,relative,rows,gaps in _templates(scene,instruction):
        for cx,cy in _translations(width,height,region,bool(locked)):
            if checked >= search_budget:
                return TeacherResult("generation_failure",mode,reason="search_budget",candidates_checked=checked)
            checked += 1
            targets = {o.object_id:o.source_pose for o in locked}
            placed = [(o.asset,o.source_pose) for o in locked]
            for obj in selected:
                p = relative[obj.object_id]
                pose = Pose(cx+p.x_m,cy+p.y_m,p.theta_rad)
                if not inside(obj.asset,pose,region) or not safe_placement(obj.asset,pose,placed):
                    break
                targets[obj.object_id] = pose
                placed.append((obj.asset,pose))
            else:
                target = with_targets(scene,targets,instruction.locked_ids)
                validation = validate_layout(scene,target,instruction,mode)
                if not validation.valid:
                    # A teacher/validator disagreement is a bug, never silently discarded.
                    raise RuntimeError(f"teacher contract violated: {validation.errors}")
                row_ids = [[o.object_id for o in row] for row in rows]
                relations = {
                    "horizontal_neighbors":[{"ids":[a,b],"gap_m":instruction.gap_m}
                                            for row in row_ids for a,b in zip(row,row[1:])],
                    "vertical_row_neighbors":[{"upper_ids":a,"lower_ids":b,"gap_m":g}
                                              for a,b,g in zip(row_ids,row_ids[1:],gaps)]}
                return TeacherResult("ok",mode,target,rows=row_ids,relations=relations,candidates_checked=checked)
    return TeacherResult("generation_failure",mode,reason="candidate_family_exhausted",candidates_checked=checked)
