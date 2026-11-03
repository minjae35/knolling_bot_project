"""v1.3 finite-candidate optimization; no stochastic layout tie-breaking."""
from dataclasses import dataclass,field,replace
import math
from .schema import Pose,COLORS,REGIONS,check_instruction,with_targets,angle_error
from .teacher_legacy import TeacherResult,_translations
from .validator import validate_layout,validate_source,geometry,inferred_rows,polygon
from .policies import BALANCED,policy_metadata


@dataclass
class BalancedResult(TeacherResult):
    audit: dict = field(default_factory=dict)


def ordered_objects(scene,ins):
    sign = 1 if ins.sort_direction == "asc" else -1
    return sorted((o for o in scene.objects if o.object_id in ins.selected_ids),
                  key=lambda o:(sign*o.asset.a_m*o.asset.b_m,o.asset.a_m,o.asset.b_m,
                                o.color,o.source_pose.y_m,o.source_pose.x_m))


def observed_rows(scene,ins):
    """Conservative structural test, not a claim about unobserved history.

Require aligned axes, common row-block center and consistent horizontal gaps.
Random singleton rows at different x do not establish an existing arrangement.
New axis/gap/region are deliberately NOT required of the old arrangement.
"""
    selected = [replace(o,target_pose=o.source_pose) for o in ordered_objects(scene,ins)]
    if len(selected) < 2:
        return []
    if not any(all(angle_error(o.source_pose.theta_rad,a) <= math.radians(5) for o in selected)
               for a in (0,math.pi/2)):
        return []
    rows = inferred_rows(selected)
    centers=[]; gaps=[]
    for row in rows:
        boxes = [polygon(o,o.source_pose).bounds for o in row]
        centers.append((min(b[0] for b in boxes)+max(b[2] for b in boxes))/2)
        gaps += [b[0]-a[2] for a,b in zip(boxes,boxes[1:])]
    if max(centers)-min(centers) > .0051 or (gaps and (min(gaps)<.0099 or max(gaps)-min(gaps)>.0051)):
        return []
    for up,down in zip(rows,rows[1:]):
        if min(polygon(o,o.source_pose).bounds[1] for o in up)-max(polygon(o,o.source_pose).bounds[3] for o in down) < .0099:
            return []
    flat = [o.object_id for row in rows for o in row]
    ordered = ordered_objects(scene,ins)
    if ins.group_by == "color":
        if any(len({o.color for o in row}) != 1 for row in rows):
            return []
        colors=[row[0].color for row in rows]
        runs=[c for i,c in enumerate(colors) if i==0 or c != colors[i-1]]
        if len(runs) != len(set(runs)):
            return []
        for color in COLORS:
            if [o.object_id for row in rows for o in row if o.color==color] != [o.object_id for o in ordered if o.color==color]:
                return []
    elif flat != [o.object_id for o in ordered]:
        return []
    return rows


def partition_sizes(n,k,balanced):
    r = math.ceil(n/k)
    if not balanced:
        return [min(k,n-i) for i in range(0,n,k)]
    q,s = divmod(n,r)
    return [q+1]*s+[q]*(r-s)


def templates(scene,ins):
    ordered = ordered_objects(scene,ins)
    indices = {o.object_id:i for i,o in enumerate(ordered)}
    groups = [[o for o in ordered if o.color==c] for c in COLORS] if ins.group_by=="color" else [ordered]
    groups = [g for g in groups if g]
    seen=set(); output=[]
    def add(rows,kind,k):
        key=tuple(tuple(indices[o.object_id] for o in row) for row in rows)
        if not rows or key in seen:
            return
        seen.add(key)
        theta = 0 if ins.axis_alignment=="horizontal" else -math.pi/2
        def dims(o):
            return (o.asset.a_m*abs(math.cos(theta))+o.asset.b_m*abs(math.sin(theta)),
                    o.asset.a_m*abs(math.sin(theta))+o.asset.b_m*abs(math.cos(theta)))
        widths=[math.fsum(dims(o)[0] for o in row)+(len(row)-1)*ins.gap_m for row in rows]
        heights=[max(dims(o)[1] for o in row) for row in rows]
        gaps=[ins.gap_m*(2 if ins.group_by=="color" and a[0].color!=b[0].color else 1) for a,b in zip(rows,rows[1:])]
        width,height=max(widths),math.fsum(heights)+math.fsum(gaps)
        relative={}; top=height/2
        for i,row in enumerate(rows):
            left=-widths[i]/2
            for o in row:
                w,h=dims(o)
                relative[o.object_id]=Pose(left+w/2,top-heights[i]/2,theta)
                left += w+ins.gap_m
            top -= heights[i]+(gaps[i] if i<len(gaps) else 0)
        l,b,r,t=REGIONS[ins.region]
        output.append({"key":key,"kind":kind,"k":k,"width":width,"height":height,"rows":rows,
                       "gaps":gaps,"relative":relative,"J_shape":max(width/(r-l),height/(t-b))})
    for k in range(1,len(ordered)+1):
        for balanced in (False,True):
            rows=[]
            for group in groups:
                start=0
                for size in partition_sizes(len(group),k,balanced):
                    rows.append(group[start:start+size]); start+=size
            add(rows,"balanced" if balanced else "sequential",k)
    if ins.locked_ids:
        add(observed_rows(scene,ins),"observed",None)
    return output


def movement_cost(scene,ins,poses):
    selected=ordered_objects(scene,ins)
    return math.fsum((poses[o.object_id].x_m-o.source_pose.x_m)**2+
                     (poses[o.object_id].y_m-o.source_pose.y_m)**2 for o in selected)/len(selected) if selected else 0.0


def ranked_candidates(scene,ins):
    options=templates(scene,ins); candidates=[]
    for index,t in enumerate(options):
        shape_key=(round(t["J_shape"],12),round(t["width"]*t["height"],12),len(t["rows"]),t["key"])
        for x,y in _translations(t["width"],t["height"],REGIONS[ins.region],bool(ins.locked_ids)):
            poses={oid:Pose(x+p.x_m,y+p.y_m,p.theta_rad) for oid,p in t["relative"].items()}
            move=movement_cost(scene,ins,poses)
            key=(round(move,12),)+shape_key+(-y,x) if ins.locked_ids else shape_key
            candidates.append((key,index,x,y,move))
    return options,sorted(candidates,key=lambda c:c[0])


def solve(scene,ins,search_budget=20000):
    check_instruction(scene,ins)
    mode="partial" if ins.locked_ids else "full"
    meta=policy_metadata(BALANCED,search_budget)
    def result(status,reason=None,target=None,**kwargs):
        return BalancedResult(status,mode,target,reason,audit=dict(meta,**kwargs))
    if ins.expected_status in ("unsupported","ambiguous"):
        return result(ins.expected_status,ins.reason)
    if ins.expected_status=="infeasible":
        raise ValueError("request feasibility must be established by geometry, not a label")
    current=with_targets(scene,{o.object_id:o.source_pose for o in scene.objects},[o.object_id for o in scene.objects])
    if validate_layout(scene,current,ins,"no_op").valid:
        out=result("ok",target=current,J_move=0.0,J_shape=None,template_key=None,geometry_checks=0,candidate_count=0,template_count=0)
        out.mode="no_op"
        return out
    if ins.keep_all:
        out=result("infeasible","keep_all_violates_final_geometry"); out.mode="no_op"
        return out
    locked=[o for o in scene.objects if o.object_id in ins.locked_ids]
    if not geometry(locked,"source_pose",.010,.010).valid:
        return result("infeasible","immutable_objects_violate_final_geometry")
    if not validate_source(scene,"full").valid:
        return result("generation_failure","invalid_source_scene")
    options,candidates=ranked_candidates(scene,ins)
    for checked,(_,index,x,y,move) in enumerate(candidates,1):
        if checked>search_budget:
            out=result("generation_failure","search_budget",geometry_checks=search_budget,candidate_count=len(candidates))
            out.candidates_checked=search_budget
            return out
        t=options[index]
        poses={o.object_id:o.source_pose for o in locked}
        poses.update({oid:Pose(x+p.x_m,y+p.y_m,p.theta_rad) for oid,p in t["relative"].items()})
        target=with_targets(scene,poses,ins.locked_ids)
        # Exact polygon oracle: a sufficient SAT gap filter would discard valid
        # diagonal candidates, invalidating the minimum-movement claim.
        if not validate_layout(scene,target,ins,mode).valid:
            continue
        row_ids=[[o.object_id for o in row] for row in t["rows"]]
        out=result("ok",target=target,J_move=move,J_shape=t["J_shape"],template_key=t["key"],template_kind=t["kind"],
                   k=t["k"],row_sizes=[len(r) for r in row_ids],block_width_m=t["width"],block_height_m=t["height"],
                   center_m=[x,y],geometry_checks=checked,candidate_count=len(candidates),template_count=len(options))
        out.candidates_checked=checked; out.rows=row_ids
        out.relations={"horizontal_neighbors":[{"ids":[a,b],"gap_m":ins.gap_m} for row in row_ids for a,b in zip(row,row[1:])],
                       "vertical_row_neighbors":[{"upper_ids":a,"lower_ids":b,"gap_m":g} for a,b,g in zip(row_ids,row_ids[1:],t["gaps"])]}
        return out
    out=result("generation_failure","candidate_family_exhausted",geometry_checks=len(candidates),candidate_count=len(candidates))
    out.candidates_checked=len(candidates)
    return out
