from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from knolling_model.recovery.goal_labels import goal_labels, VOCAB
from knolling_model.recovery.evaluation_v4 import evaluate_v4,gate,MICRO_CASES
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.contracts import oracle_objects
from test_recovery_metrics import ExactFixture,fixture_batches
from test_recovery_dataset_v3 import suite_v3


class ExactGoal(nn.Module):
    """Annotation-backed evaluator fixture, NEVER used in production forward."""
    def __init__(self,records,confidence=12.):
        super().__init__();self.records=records;self.confidence=confidence

    def forward(self,obj,text,mask):
        r=self.records[int(text[0,0])];ids,_=recovery_supervision(r);labels=goal_labels(r,ids)
        c=self.confidence
        logits={k:torch.full((1,len(v)),-c) for k,v in VOCAB.items()}
        for k in VOCAB:logits[k][0,max(0,int(labels['goal_'+k]))]=c
        selection=torch.where(labels['goal_selected'].bool(),c,-c)[None]
        return dict(keep_logit=torch.tensor([c if labels['goal_keep'] else -c]),logits=logits,
                    selected_logit=selection,selected=selection.sigmoid(),valid=obj.valid)


class ExactLayout(nn.Module):
    def __init__(self,records):
        super().__init__();self.fixture=ExactFixture(records);self.goal_encoder=ExactGoal(records)

    def forward(self,obj,text,mask):
        return self.fixture.layout(obj,text,mask)


class ExactV4Fixture(ExactFixture):
    architecture='recovery_v4_goal_joint'
    def __init__(self,records):
        super().__init__(records)
        # A module can't shadow an inherited class method via _modules lookup.
        self.network=ExactLayout(records)
    @property
    def layout(self):return self.network
    def forward(self,images,input_ids,attention_mask):
        v=self.perception(images)
        return self.layout_from_perception(v,input_ids,attention_mask)
    def layout_from_perception(self,visual,input_ids,attention_mask):
        t=recovery_supervision(self.records[int(input_ids[0,0])])[1]
        return dict(source=visual['source'],layout=self.layout(None,input_ids,attention_mask),threshold=.5,
                    goal=self.layout.goal_encoder(oracle_objects([t]),input_ids,attention_mask))


def test_grounding_and_full_gates_require_real_complete_metrics(suite_v3):
    ds=suite_v3;ids=[r['sample_id'] for r in ds.records];b=fixture_batches(ds.records);m=ExactV4Fixture(ds.records)
    for stage in ('grounding','oracle_layout','perception','predicted_layout','joint'):
        initial,_=evaluate_v4(m,b,stage=stage,expected_ids=ids,pairs=ds.pairs)
        out,_=evaluate_v4(m,b,stage=stage,expected_ids=ids,pairs=ds.pairs,baseline=initial)
        assert gate(stage,out),(stage,out)
        if stage=='grounding':assert out['validity'] is None and out['fidelity'] is None
        if stage=='perception':assert out['grounding'] is None
    for bad in (b[:-1],b+[b[0]]):
        out,_=evaluate_v4(m,bad,stage='grounding',expected_ids=ids,pairs=ds.pairs)
        assert not gate('grounding',out)
    out,_=evaluate_v4(m,b,stage='grounding',expected_ids=ids,pairs=ds.pairs[:-1])
    assert not gate('grounding',out)
    m.layout.goal_encoder.confidence=1.
    out,_=evaluate_v4(m,b,stage='grounding',expected_ids=ids,pairs=ds.pairs)
    assert out['grounding']['correct']==out['grounding']['total']
    assert not gate('grounding',out)  # Correct argmax alone isn't confidence >= .9.


def test_goal_counterfactual_language_and_joint_regression(suite_v3):
    ds=suite_v3;ids=[r['sample_id'] for r in ds.records];b=fixture_batches(ds.records);m=ExactV4Fixture(ds.records)
    before,_=evaluate_v4(m,b,stage='joint',expected_ids=ids,pairs=ds.pairs)
    m.layout.goal_encoder.confidence=3.
    after,_=evaluate_v4(m,b,stage='joint',expected_ids=ids,pairs=ds.pairs,baseline=before)
    assert after['grounding']['passed'] and after['full_set_no_regression']
    m.layout.goal_encoder.confidence=1.
    after,_=evaluate_v4(m,b,stage='joint',expected_ids=ids,pairs=ds.pairs,baseline=before)
    assert not after['grounding']['passed'] and not after['full_set_no_regression']
    m.layout.goal_encoder.confidence=12.
    a,z=next(p for p in ds.pairs if p['factor']=='axis_alignment')['sample_ids']
    b[ids.index(z)].inputs['input_ids'][:]=ids.index(a)
    out,_=evaluate_v4(m,b,stage='grounding',expected_ids=ids,pairs=ds.pairs)
    assert not gate('grounding',out)


@pytest.mark.parametrize('case',MICRO_CASES)
def test_micro_maximum_errors_and_fixed_cases(suite_v3,case):
    records=[next(r for r in suite_v3.records if r['sample_id']==sid) for sid in MICRO_CASES[case]]
    m=ExactV4Fixture(records);b=fixture_batches(records)
    out,_=evaluate_v4(m,b,stage='micro_pose',expected_ids=list(MICRO_CASES[case]),micro_case=case)
    assert gate('micro_pose',out)
    # One object's error must fail even when a percentile might conceal it.
    original=m.layout.forward
    def shifted(*args):
        value=original(*args);value['xy']=value['xy'].clone();value['xy'][0,0,0]+=.0011;return value
    m.layout.forward=shifted
    out,_=evaluate_v4(m,b,stage='micro_pose',expected_ids=list(MICRO_CASES[case]),micro_case=case)
    assert not gate('micro_pose',out) and out['micro']['max_position_mm']>1


def test_nonfinite_goal_cannot_pass_or_emit_nan_metrics(suite_v3):
    import json
    record=next(r for r in suite_v3.records if not r['instruction']['keep_all'] and r['instruction']['axis_alignment']=='horizontal')
    model=ExactV4Fixture([record]);original=model.layout.goal_encoder.forward
    def corrupt(*args):
        goal=original(*args);goal['logits']['axis'].fill_(float('nan'));return goal
    model.layout.goal_encoder.forward=corrupt
    out,_=evaluate_v4(model,fixture_batches([record]),stage='grounding',expected_ids=[record['sample_id']])
    row=out['grounding']['rows'][0]
    assert not row['passed'] and row['min_true_probability']==0
    json.dumps(out,allow_nan=False)


def test_grounding_breakdown_distinguishes_accuracy_and_confidence(suite_v3,tmp_path):
    from knolling_model.recovery.runtime import ProgressLog
    records=suite_v3.records;model=ExactV4Fixture(records)
    model.layout.goal_encoder.confidence=1.
    out,_=evaluate_v4(model,fixture_batches(records),stage='grounding',
        expected_ids=[r['sample_id'] for r in records],pairs=suite_v3.pairs)
    goal=out['grounding']
    active=sum(not r['instruction']['keep_all'] for r in records)
    assert goal['attributes']['keep']==dict(correct=76,total=76,accuracy=1.)
    for name in VOCAB:assert goal['attributes'][name]==dict(correct=active,total=active,accuracy=1.)
    count=sum(len(r['source']['objects']) for r in records)
    assert goal['selection']==dict(correct=count,total=count,accuracy=1.)
    assert goal['contrast_total']==35 and goal['contrast_passed']==0 and not goal['passed']
    progress=ProgressLog(tmp_path)
    progress('evaluation',stage='grounding',update=0,metrics=out)
    import json
    logged=json.loads((tmp_path/'history.jsonl').read_text())
    assert logged['metrics']['grounding']['selection']==goal['selection']
