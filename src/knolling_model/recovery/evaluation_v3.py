"""Independent strict task validity, canonical fidelity, and pose-derived move."""
import math

import torch

from knolling_model.coordinates import angle_delta
from knolling_model.data import Batch
from knolling_model.diagnostics import positioned_objects
from knolling_model.evaluation import evaluate_batch, summarize, semantic_errors
from .contracts import oracle_objects
from .data import recovery_supervision
from .evaluation import _oracle_view, correspondences, evaluate_recovery
from .model import CapacityError, legacy_evaluation_view

SCHEMA = 'recovery_metrics_v3'


def semantic_errors_v3(objects, record):
    return semantic_errors(objects,record,keep_m=.001,keep_deg=1,axis_deg=2,gap_m=.002,row_m=.002)


def source_pass(metrics):
    from .stages import stage_gate
    legacy=dict(metrics,schema='recovery_metrics_v2',stage='perception')
    return stage_gate('perception',legacy)


def gate(stage, metrics):
    from .stages import STAGES
    if stage not in STAGES:
        raise ValueError('unknown recovery stage')
    if (metrics.get('schema') != SCHEMA or metrics.get('stage') != stage or not metrics.get('complete_set')
            or metrics.get('evaluation_mode') != 'free_rollout' or metrics.get('architecture') != 'recovery_v3_joint'):
        return False
    if stage == 'perception':return source_pass(metrics)
    if metrics.get('source_mode') != ('oracle' if stage=='oracle_layout' else 'predicted'):
        return False
    return (metrics.get('validity',{}).get('passed') is True and metrics.get('contrasts',{}).get('passed') is True
            and (stage=='oracle_layout' or source_pass(metrics))
            and (stage!='joint' or metrics.get('full_set_no_regression') is True))


def no_regression(current, baseline):
    if not (baseline and current.get('complete_set') and baseline.get('complete_set')
            and baseline.get('schema')==SCHEMA and current['sample_ids']==baseline['sample_ids']):return False
    for key in ('task_success','count_accuracy','coverage','color_accuracy','shape_accuracy','move_accuracy'):
        if current[key] < baseline[key]-1e-9:return False
    for key in ('false_positives','false_negatives','source_position_p95_mm','source_size_p95_mm','source_angle_p95_deg'):
        if current[key] is None or baseline[key] is None or current[key] > baseline[key]+1e-6:return False
    if current['contrasts']['success'] < baseline['contrasts']['success']:return False
    if current['validity']['keep_violations'] > baseline['validity']['keep_violations']:return False
    return all(k in current['by_mode'] and current['by_mode'][k]['task_success']>=v['task_success']
               for k,v in baseline['by_mode'].items())


@torch.no_grad()
def evaluate_v3(model, batches, *, stage, expected_ids=None, baseline=None, pairs=()):
    from .stages import STAGES
    if stage not in STAGES:
        raise ValueError('unknown recovery stage')
    if getattr(model,'architecture',None) != 'recovery_v3_joint' or model.threshold != .5:
        raise ValueError('v3 evaluator requires the v3 architecture and standard selection')
    if stage=='perception':
        metrics,rows=evaluate_recovery(model,batches,stage=stage,expected_ids=expected_ids)
        metrics.update(schema=SCHEMA,architecture=model.architecture,fidelity=None,validity=None,contrasts=None)
        return metrics,rows
    was_training=model.training;model.eval()
    rows,records=[],[]
    fp=fn=colors=shapes=move=total=0
    try:
        for batch in batches:
            for index,meta in enumerate(batch.metadata):
                inputs={k:v[index:index+1] for k,v in batch.inputs.items()}
                truth={k:v.to(inputs['images'].device) for k,v in recovery_supervision(meta['record'])[1].items()}
                one=Batch(inputs,[truth],[meta]);overflow=False
                if stage=='oracle_layout':
                    objects=oracle_objects([truth])
                    output=model.layout(objects,model.encode_text(inputs['input_ids'],inputs['attention_mask']),inputs['attention_mask'])
                    view=_oracle_view(objects,output)
                else:
                    visual=model.perception(inputs['images'])
                    try:view=legacy_evaluation_view(model.layout_from_perception(visual,inputs['input_ids'],inputs['attention_mask']))
                    except CapacityError:
                        s=visual['source'];view=dict(source=s,targets=[dict(uv=s['uv'],angle=s['angle'])]);overflow=True
                row=evaluate_batch(view,one,semantic_check=semantic_errors_v3)[0]
                if overflow:row['errors'].append('layout_capacity');row['task_success']=False
                matches,extra,missing=correspondences(view['source'],truth,meta['record'])
                fp+=extra;fn+=missing;total+=len(truth['source_xy'])
                for q,g in matches:
                    source=view['source'];target=view['targets'][0]
                    colors+=int(source['color_logits'][0,q].argmax()==truth['color'][g])
                    shapes+=int(source['shape_logits'][0,q].argmax()==truth['shape'][g])
                    pred_move=((target['uv'][0,q]-source['uv'][0,q])*source['uv'].new_tensor([.6,.4])).norm()>.003
                    pred_move=pred_move | (angle_delta(target['angle'][0,q],source['angle'][0,q]).abs()>math.pi/60)
                    gt_move=(truth['target_xy'][g]-truth['source_xy'][g]).norm()>.003
                    gt_move=gt_move | (angle_delta(truth['target_angle'][g],truth['source_angle'][g]).abs()>math.pi/60)
                    move+=int(pred_move==gt_move) if not overflow else 0
                rows.append(row);records.append(meta['record'])
        if not rows:raise ValueError('empty v3 evaluation')
        summary=summarize(rows,records,stage_count=1)
        # Legacy diagnostics use old semantic tolerances; do not expose them as v3 scores.
        allowed=('samples','count_accuracy','correspondence_coverage','raw_geometry_success','task_success',
                 'failure_counts','by_mode','by_condition','source_position_mm','source_size_mm',
                 'source_angle_deg','target_position_mm','target_angle_deg')
        result={k:summary[k] for k in allowed}
        ids=[r['sample_id'] for r in rows]
        complete=(expected_ids is not None and len(ids)==len(set(ids)) and len(expected_ids)==len(set(expected_ids))
                  and set(ids)==set(expected_ids))
        result.update(schema=SCHEMA,architecture=model.architecture,stage=stage,evaluation_mode='free_rollout',
                      source_mode='oracle' if stage=='oracle_layout' else 'predicted',sample_ids=sorted(ids),complete_set=complete,
                      coverage=summary['correspondence_coverage'],false_positives=fp,false_negatives=fn,
                      color_accuracy=colors/total,shape_accuracy=shapes/total,move_accuracy=move/total,
                      move_definition='pose_delta_gt_3mm_or_3deg')
        for name in ('source_position','source_size','source_angle','target_position','target_angle'):
            unit='deg' if 'angle' in name else 'mm';result[f'{name}_p95_{unit}']=result[f'{name}_{unit}']['p95']
        fidelity=complete and result['coverage']==1 and result['count_accuracy']==1
        fidelity=fidelity and result['target_position_p95_mm']<=1 and result['target_angle_p95_deg']<=2
        result['fidelity']=dict(passed=bool(fidelity),position_p95_mm=result['target_position_p95_mm'],
                                angle_p95_deg=result['target_angle_p95_deg'],canonical_only=True)
        result['validity']=dict(passed=bool(complete and all(r['task_success'] for r in rows)),
                                success=result['task_success'],keep_violations=sum('keep' in r['errors'] for r in rows))
        by_id={r['sample_id']:r for r in rows};by_record={r['sample_id']:r for r in records};checks=[]
        for pair in pairs:
            a,b=pair['sample_ids'];passed=False
            if a in by_id and b in by_id:
                ra,rb=by_id[a],by_id[b]
                passed=(ra['task_success'] and rb['task_success'] and
                        bool(semantic_errors_v3(positioned_objects(by_record[a],ra['predicted_targets']),by_record[b])) and
                        bool(semantic_errors_v3(positioned_objects(by_record[b],rb['predicted_targets']),by_record[a])))
            checks.append(dict(pair_id=pair['pair_id'],factor=pair['factor'],passed=bool(passed)))
        factors={p['factor'] for p in pairs};required={'axis_alignment','region','selection','gap_m','sort_direction','group_by'}
        unique=len({p['pair_id'] for p in pairs})==len(pairs)
        result['contrasts']=dict(passed=bool(complete and unique and factors==required and all(x['passed'] for x in checks)),
                                 pairs=len(checks),success=sum(x['passed'] for x in checks)/len(checks) if checks else 0.,checks=checks)
        axis=[x for x in checks if x['factor']=='axis_alignment']
        result['pure_axis_success']=sum(x['passed'] for x in axis)/len(axis) if axis else 0.
        result['full_set_no_regression']=no_regression(result,baseline)
        return result,rows
    finally:model.train(was_training)
