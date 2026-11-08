import json,copy,math
from pathlib import Path
from collections import Counter
from knolling_dataset.builder import file_hash
from knolling_dataset.validator import geometry
from knolling_model.diagnostics import positioned_objects
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
root=Path('artifacts/recovery_runs/recovery_v4_vertical_gap_once_20261108T093400Z')
expected='983a6a5fd8579edc08a10be6054c3b94b870af66db3b9215524988e0e72ed54d'
assert file_hash(root/'checkpoint-001200.pt')==expected
saved=json.loads((root/'best-evaluation.json').read_text());bundle=json.loads(Path('artifacts/recovery_v3_verification_r1/annotations.json').read_text());records={r['sample_id']:r for r in bundle['records']};before={};after={};copied=0;disagreements=[]
for row in saved['rows']:
 r=records[row['sample_id']];poses=row['predicted_targets'];src={o['object_id']:o['source_pose'] for o in r['source']['objects']}
 def evaluate(p):
  o=positioned_objects(r,p);errors=sorted(set(geometry(o,'target_pose',.010,.010).errors+semantic_errors_v3(o,r)));return dict(sample_id=r['sample_id'],mode=r['mode'],predicted_targets=p,errors=errors,task_success=not errors)
 b=evaluate(poses);assert b['errors']==sorted(row['errors']);before[r['sample_id']]=b
 intervention=copy.deepcopy(poses)
 for oid,keep in r['supervision_v3']['copy_required'].items():
  if keep:intervention[oid]=copy.deepcopy(src[oid]);copied+=1
  expected_keep=r['instruction']['keep_all'] or oid not in r['instruction']['selected_ids']
  if keep!=expected_keep:disagreements.append((r['sample_id'],oid,keep,expected_keep))
 after[r['sample_id']]=evaluate(intervention)
def summary(rows):
 checks=[]
 for pair in bundle['pairs']:
  a,b=pair['sample_ids'];passed=rows[a]['task_success'] and rows[b]['task_success'] and bool(semantic_errors_v3(positioned_objects(records[a],rows[a]['predicted_targets']),records[b])) and bool(semantic_errors_v3(positioned_objects(records[b],rows[b]['predicted_targets']),records[a]))
  checks.append(dict(pair,passed=bool(passed)))
 return dict(validity=sum(r['task_success'] for r in rows.values()),counterfactual=sum(p['passed'] for p in checks),failure_counts=dict(Counter(e for r in rows.values() for e in r['errors'])),checks=checks,by_mode={m:dict(success=sum(r['task_success'] for r in rows.values() if r['mode']==m),total=sum(r['mode']==m for r in rows.values())) for m in ('full','partial','no_op')})
b=summary(before);a=summary(after);assert(b['validity'],b['counterfactual'])==(28,3)
result=dict(schema='oracle_keep_intervention_v1',source_sha256=expected,source_predictions_sha256=file_hash(root/'best-evaluation.json'),before=b,after=a,copied_objects=copied,objects=215,selection_label_disagreements=disagreements,lost_successes=[k for k in before if before[k]['task_success'] and not after[k]['task_success']],gained_successes=[k for k in before if not before[k]['task_success'] and after[k]['task_success']],rows=list(after.values()),scope='diagnostic_only_no_neural_forward_no_training')
path=Path('artifacts/recovery_diagnostics/oracle_keep_once/result.json')
with path.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','before','after')},indent=2))
for label,item in [('before',b),('after',a)]:print(label,json.dumps({k:v for k,v in item.items() if k!='checks'}))
