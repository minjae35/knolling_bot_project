import torch
from torch import nn

from knolling_model.data import Batch
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.arrangement.evaluation import evaluate_recovery, correspondences
from knolling_model.arrangement.stages import stage_gate
from test_recovery_counterfactuals import recovery_suite


class ExactFixture(nn.Module):
    """Known outputs ONLY to test evaluator semantics; never a model benchmark."""
    threshold, max_objects = .5, 8

    def __init__(self, records):
        super().__init__()
        self.records = records

    def encode_text(self, input_ids, attention_mask):
        return input_ids

    def perception(self, images):
        t = recovery_supervision(self.records[int(images[0, 0, 0, 0])])[1]
        n = len(t['size'])
        return dict(source=dict(uv=(t['source_xy']/torch.tensor([.6,.4]))[None], size=t['size'][None],
            angle=t['source_angle'][None], objectness_logits=torch.full((1,n),10.),
            color_logits=torch.nn.functional.one_hot(t['color'],3)[None].float()*20,
            shape_logits=torch.nn.functional.one_hot(t['shape'],3)[None].float()*20))

    def layout(self, objects, text, mask):
        t = recovery_supervision(self.records[int(text[0,0])])[1]
        n = len(t['size'])
        return dict(xy=t['target_xy'][None], angle=t['target_angle'][None],
            valid=torch.ones(1,n,dtype=torch.bool), query_index=torch.arange(n)[None],
            move_logits=torch.where(t['move'][None],10.,-10.), rollout='free_rollout')

    def layout_from_perception(self, visual, input_ids, attention_mask):
        return dict(source=visual['source'], layout=self.layout(None,input_ids,attention_mask), threshold=.5)


def fixture_batches(records):
    batches = []
    for i, r in enumerate(records):
        ids, truth = recovery_supervision(r)
        batches.append(Batch(dict(images=torch.full((1,3,1,1),float(i)), input_ids=torch.tensor([[i]]),
            attention_mask=torch.tensor([[True]])), [truth], [dict(record=r, sample_id=r['sample_id'], object_ids=ids)]))
    return batches


def test_actual_evaluator_drives_every_stage_gate(recovery_suite):
    suite, _ = recovery_suite
    records = [suite.records[0]]+suite.records[32:34]
    model, batches = ExactFixture(records), fixture_batches(records)
    ids = [r['sample_id'] for r in records]
    for stage in ('oracle_layout', 'perception', 'predicted_layout', 'joint'):
        first, _ = evaluate_recovery(model, batches, stage=stage, expected_ids=ids)
        measured, _ = evaluate_recovery(model, batches, stage=stage, expected_ids=ids, baseline=first)
        assert stage_gate(stage, measured), (stage, measured)
        assert len(measured['refinement']) == (0 if stage == 'perception' else 1)
        if stage == 'perception':
            assert measured['target_position_mm']['n'] == 0
            assert measured['language_sensitivity']['evaluated'] is False
            assert all(mode['task_success'] is None for mode in measured['by_mode'].values())
        assert measured['false_positives'] == measured['false_negatives'] == 0
        incomplete, _ = evaluate_recovery(model, batches[:1], stage=stage, expected_ids=ids, baseline=first)
        assert not stage_gate(stage, incomplete)
    wrong = fixture_batches(records)
    # Both commands emit the first command's target; pure-axis test must fail.
    wrong[2].inputs['input_ids'][:] = 1
    measured, _ = evaluate_recovery(model, wrong, expected_ids=ids)
    assert measured['pure_axis_success'] == 0


def test_independent_false_positive_and_false_negative_counts(recovery_suite):
    suite, _ = recovery_suite
    r = suite.records[0]
    model = ExactFixture([r])
    source = model.perception(torch.zeros(1,3,1,1))['source']
    _, t = recovery_supervision(r)
    # Duplicate the first object and remove the final object's prediction.
    for key in source:
        source[key][:, -1] = source[key][:, 0]
    pairs, fp, fn = correspondences(source,t,r)
    assert fp == fn == 1 and len(pairs) == len(t['size'])-1
