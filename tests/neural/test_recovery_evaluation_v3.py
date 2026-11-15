from copy import deepcopy
import pytest
import torch

from knolling_model.arrangement.evaluation_v3 import evaluate_v3,gate,semantic_errors_v3
from knolling_model.arrangement.stages import stage_gate
from test_recovery_metrics import ExactFixture,fixture_batches
from test_recovery_dataset_v3 import suite_v3


class ExactV3Fixture(ExactFixture):
    architecture='recovery_v3_joint'
    def layout(self,objects,text,mask):
        out=super().layout(objects,text,mask)
        del out['move_logits']
        return out


def test_exact_fixture_evaluator_gates_and_missing_pairs(suite_v3):
    model=ExactV3Fixture(suite_v3.records);batches=fixture_batches(suite_v3.records)
    ids=[r['sample_id'] for r in suite_v3.records]
    for stage in ('oracle_layout','perception','predicted_layout','joint'):
        first,_=evaluate_v3(model,batches,stage=stage,expected_ids=ids,pairs=suite_v3.pairs)
        result,_=evaluate_v3(model,batches,stage=stage,expected_ids=ids,pairs=suite_v3.pairs,baseline=first)
        assert gate(stage,result) and stage_gate(stage,result),(stage,result)
        if stage=='perception':
            assert result['validity'] is None and result['target_position_p95_mm'] is None
            assert not gate(stage,dict(result,source_mode='oracle'))
        else:
            assert result['fidelity']['passed'] and result['validity']['passed']
            assert result['move_accuracy']==1 and result['contrasts']['success']==1
    partial,_=evaluate_v3(model,batches[:-1],stage='oracle_layout',expected_ids=ids,pairs=suite_v3.pairs)
    assert not gate('oracle_layout',partial)
    no_pairs,_=evaluate_v3(model,batches,stage='oracle_layout',expected_ids=ids)
    assert not gate('oracle_layout',no_pairs)
    with pytest.raises(ValueError,match='unknown'):gate('invalid_stage',result)
    with pytest.raises(ValueError,match='unknown'):
        evaluate_v3(model,batches,stage='invalid_stage',expected_ids=ids,pairs=suite_v3.pairs)


def test_alternative_valid_layout_fails_fidelity_but_not_validity(suite_v3):
    record=next(r for r in suite_v3.records if r['mode']=='full')
    class Shift(ExactV3Fixture):
        def layout(self,*args):
            out=super().layout(*args);out['xy']=out['xy']+torch.tensor([.01,0]);return out
    model=Shift([record]);b=fixture_batches([record])
    result,_=evaluate_v3(model,b,stage='oracle_layout',expected_ids=[record['sample_id']])
    assert result['validity']['passed'] and not result['fidelity']['passed']
    class Collision(ExactV3Fixture):
        def layout(self,*args):
            out=super().layout(*args);out['xy'][:]=out['xy'][:,0:1];return out
    result,_=evaluate_v3(Collision([record]),b,stage='oracle_layout',expected_ids=[record['sample_id']])
    assert not result['validity']['passed'] and 'collision' in result['failure_counts']


def test_ignoring_counterfactual_language_fails(suite_v3):
    ids=[r['sample_id'] for r in suite_v3.records];b=fixture_batches(suite_v3.records)
    pair=next(p for p in suite_v3.pairs if p['factor']=='axis_alignment')
    a,z=[ids.index(s) for s in pair['sample_ids']]
    b[z].inputs['input_ids'][:]=a
    result,_=evaluate_v3(ExactV3Fixture(suite_v3.records),b,stage='oracle_layout',expected_ids=ids,pairs=suite_v3.pairs)
    assert not gate('oracle_layout',result) and result['pure_axis_success']<1
