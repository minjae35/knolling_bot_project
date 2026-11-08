"""Independent goal grounding, micro learnability and full layout acceptance."""
import math
import torch

from knolling_model.coordinates import angle_delta
from .contracts import oracle_objects
from .data import recovery_supervision
from .goal_labels import goal_labels, VOCAB
from .goal_joint_layout import ARCHITECTURE
from .model import CapacityError
from .evaluation import correspondences
from .evaluation_v3 import evaluate_joint, no_regression, source_pass

SCHEMA = 'recovery_metrics_v4'
MICRO_CASES = {
    'translation': ('v3_00f5b6400c9c805edd26f2f6',),
    'keep': ('v3_1f2cb71631ace3774bd5e1b6',),
    'axis': ('v3_1efbc2cf2648c75dd2dcca9f', 'v3_0c1e24b0cf284df95555f234'),
}


def goal_summary(rows, pairs, complete):
    by_id = {r['sample_id']: r for r in rows}
    checks = []
    for pair in pairs:
        members = [by_id.get(s) for s in pair['sample_ids']]
        # Both full semantic vectors must agree with their own labels; this
        # checks changed AND unchanged fields, not just a nonzero response.
        checks.append(dict(pair_id=pair['pair_id'], factor=pair['factor'],
                           passed=all(r is not None and r['passed'] for r in members)))
    unique = len({p['pair_id'] for p in pairs}) == len(pairs)
    required = {'axis_alignment','region','selection','gap_m','sort_direction','group_by'}
    contrast_ok = unique and len(pairs)==35 and {p['factor'] for p in pairs}==required and all(r['passed'] for r in checks)
    def score(values):
        correct=sum(v['correct'] for v in values);total=sum(v['total'] for v in values)
        return dict(correct=correct,total=total,accuracy=correct/total if total else None)
    attributes={k:score([r['attributes'][k] for r in rows if k in r.get('attributes',{})]) for k in ('keep',*VOCAB)}
    selection=score([r['selection'] for r in rows if 'selection' in r])
    return dict(passed=bool(complete and len(rows)==76 and all(r['passed'] for r in rows) and contrast_ok),
                correct=sum(r['correct'] for r in rows), total=sum(r['total'] for r in rows),
                min_true_probability=min((r['min_true_probability'] for r in rows), default=0.),
                attributes=attributes, selection=selection,
                contrast_passed=sum(c['passed'] for c in checks), contrast_total=len(checks),
                contrasts=checks, rows=rows)


@torch.no_grad()
def evaluate_goals(model, batches, *, stage, expected_ids, pairs):
    was_training=model.training;model.eval();rows=[]
    try:
        for batch in batches:
            for i, meta in enumerate(batch.metadata):
                x={k:v[i:i+1] for k,v in batch.inputs.items()};r=meta['record']
                ids,t=recovery_supervision(r)
                t={k:v.to(x['images'].device) for k,v in t.items()}
                labels=goal_labels(r,ids)
                attributes={k:dict(correct=0,total=1) for k in ('keep',*VOCAB) if k=='keep' or int(labels['goal_'+k])>=0}
                selection=dict(correct=0,total=len(ids))
                oracle=stage in ('grounding','micro_pose','oracle_layout')
                if oracle:
                    objects=oracle_objects([t])
                    goal=model.layout.goal_encoder(objects,model.encode_text(x['input_ids'],x['attention_mask']),x['attention_mask'])
                    matches=[(j,j) for j in range(len(ids))];complete_objects=True
                else:
                    try:out=model(**x)
                    except CapacityError:
                        rows.append(dict(sample_id=r['sample_id'],correct=0,total=1,min_true_probability=0.,passed=False,
                                         attributes=attributes,selection=selection))
                        continue
                    goal=out['goal']
                    source_matches,fp,fn=correspondences(out['source'],t,r)
                    qids=out['layout']['query_index'][0].tolist()
                    matches=[(qids.index(q),g) for q,g in source_matches if q in qids]
                    complete_objects=not fp and not fn and len(matches)==len(ids)
                p=goal['keep_logit'][0].sigmoid().item();keep=bool(labels['goal_keep'])
                probs=[p if keep else 1-p];correct=[(p>=.5)==keep]
                attributes['keep']['correct']=int(math.isfinite(p) and correct[-1])
                for k in VOCAB:
                    label=int(labels['goal_'+k])
                    if label<0:continue
                    v=goal['logits'][k][0].softmax(-1)
                    probs.append(v[label].item());correct.append(int(v.argmax())==label)
                    attributes[k]['correct']=int(math.isfinite(probs[-1]) and correct[-1])
                for q,g in matches:
                    p=goal['selected'][0,q].item();label=bool(labels['goal_selected'][g])
                    probs.append(p if label else 1-p);correct.append((p>=.5)==label)
                    selection['correct']+=int(math.isfinite(p) and correct[-1])
                finite=all(math.isfinite(p) for p in probs)
                minimum=min(probs) if finite else 0.
                passed=complete_objects and finite and all(correct) and minimum>=.9
                rows.append(dict(sample_id=r['sample_id'],correct=sum(correct),total=len(correct),
                                 min_true_probability=minimum,passed=bool(passed),attributes=attributes,selection=selection))
        ids=[r['sample_id'] for r in rows]
        complete=expected_ids is not None and len(ids)==len(set(ids))==len(expected_ids)==len(set(expected_ids)) and set(ids)==set(expected_ids)
        return goal_summary(rows,pairs,complete),complete,sorted(ids)
    finally:model.train(was_training)


def micro_metrics(rows, records, case):
    expected=MICRO_CASES.get(case,())
    by_id={r['sample_id']:r for r in rows}
    complete=len(rows)==len(expected) and set(by_id)==set(expected) and bool(expected)
    errors=[v for r in rows for v in r['target_position_mm']]
    angles=[v for r in rows for v in r['target_angle_deg']]
    position=max(errors,default=float('inf'));angle=max(angles,default=float('inf'))
    axis_error=None
    if case=='axis' and complete and all('predicted_targets' in r for r in rows):
        a,b=[by_id[s] for s in expected];ra,rb=[records[s] for s in expected]
        def targets(r):return {o['object_id']:o['target_pose']['theta_rad'] for o in r['target']['objects']}
        ta,tb=targets(ra),targets(rb)
        delta=[]
        for oid in ta:
            pd=b['predicted_targets'][oid]['theta_rad']-a['predicted_targets'][oid]['theta_rad']
            td=tb[oid]-ta[oid]
            delta.append(float(angle_delta(torch.tensor(pd),torch.tensor(td)).abs())*180/math.pi)
        axis_error=max(delta)
    valid=all(r['task_success'] for r in rows)
    passed=complete and valid and position<=1 and angle<=2 and (case!='axis' or (axis_error is not None and axis_error<=4))
    return dict(case=case, sample_ids=list(expected), passed=bool(passed),
                max_position_mm=position if math.isfinite(position) else None,
                max_angle_deg=angle if math.isfinite(angle) else None, axis_change_error_deg=axis_error)


@torch.no_grad()
def evaluate_v4(model,batches,*,stage,expected_ids=None,baseline=None,pairs=(),micro_case=None):
    from .stages import ALL_STAGES
    if stage not in ALL_STAGES or getattr(model,'architecture',None)!=ARCHITECTURE or model.threshold!=.5:
        raise ValueError('invalid v4 evaluation architecture/stage/selection')
    # The tiny verification suite is bounded; retain CPU/GPU batch references only.
    batches=list(batches)
    if stage=='perception':
        result,rows=evaluate_joint(model,batches,stage=stage,architecture=ARCHITECTURE,schema=SCHEMA,expected_ids=expected_ids)
        result['grounding']=None
        return result,rows
    goals,complete,ids=evaluate_goals(model,batches,stage=stage,expected_ids=expected_ids,pairs=pairs)
    if stage=='grounding':
        return dict(schema=SCHEMA,architecture=ARCHITECTURE,stage=stage,complete_set=complete,
                    sample_ids=ids,source_mode='oracle',evaluation_mode='free_rollout',grounding=goals,
                    fidelity=None,validity=None,contrasts=None),goals['rows']
    result,rows=evaluate_joint(model,batches,stage='oracle_layout' if stage=='micro_pose' else stage,
        architecture=ARCHITECTURE,schema=SCHEMA,expected_ids=expected_ids,pairs=pairs,baseline=baseline)
    result.update(stage=stage,grounding=goals)
    if stage=='micro_pose':
        records={m['sample_id']:m['record'] for b in batches for m in b.metadata}
        result['micro']=micro_metrics(rows,records,micro_case)
    if stage=='joint':
        result['full_set_no_regression']=bool(no_regression(result,baseline) and baseline and
            goals['correct']>=baseline.get('grounding',{}).get('correct',float('inf')) and
            goals['passed'] is True and baseline.get('grounding',{}).get('passed') is True)
    return result,rows


def gate(stage,metrics):
    from .stages import ALL_STAGES
    if stage not in ALL_STAGES:raise ValueError('unknown recovery stage')
    if (metrics.get('schema')!=SCHEMA or metrics.get('architecture')!=ARCHITECTURE or metrics.get('stage')!=stage
            or metrics.get('complete_set') is not True or metrics.get('evaluation_mode')!='free_rollout'):
        return False
    if metrics.get('source_mode')!=('oracle' if stage in ('grounding','micro_pose','oracle_layout') else 'predicted'):
        return False
    if stage=='grounding':return metrics.get('grounding',{}).get('passed') is True
    if stage=='micro_pose':return metrics.get('micro',{}).get('passed') is True
    if stage=='perception':return source_pass(metrics)
    return (metrics.get('grounding',{}).get('passed') is True and metrics.get('validity',{}).get('passed') is True
            and metrics.get('contrasts',{}).get('passed') is True
            and (stage=='oracle_layout' or source_pass(metrics))
            and (stage!='joint' or metrics.get('full_set_no_regression') is True))
