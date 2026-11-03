from dataclasses import replace
import torch
from knolling_dataset.schema import scene_from_dict, Pose
from knolling_model.coordinates import to_normalized
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch, summarize, semantic_errors


def oracle(sample):
    t = sample['supervision']; n = len(t['source_xy']); q = 12
    source = {'uv': torch.zeros(1, q, 2), 'size': torch.full((1, q, 2), .035),
              'angle': torch.zeros(1, q), 'objectness_logits': torch.full((1, q), -10.),
              'color_logits': torch.zeros(1, q, 3), 'shape_logits': torch.zeros(1, q, 3)}
    source['uv'][0,:n] = to_normalized(t['source_xy'])
    source['size'][0,:n] = t['size']; source['angle'][0,:n] = t['source_angle']
    source['objectness_logits'][0,:n] = 10
    target = {'uv': torch.zeros(1, q, 2), 'angle': torch.zeros(1, q)}
    target['uv'][0,:n] = to_normalized(t['target_xy']); target['angle'][0,:n] = t['target_angle']
    return {'source':source,'targets':[target]*3}, Batch({}, [t], [sample['metadata']])


def test_oracle_all_32_has_full_identity_geometry_semantics_and_gate(s0_dataset):
    rows = []
    for sample in s0_dataset:
        output, batch = oracle(sample)
        rows.extend(evaluate_batch(output, batch))
    result = summarize(rows)
    assert result['count_accuracy'] == result['correspondence_coverage'] == result['raw_geometry_success'] == result['task_success'] == 1
    assert result['s0_gate'] and result['failure_counts'] == {}


def test_count_and_source_identity_failures_cannot_disappear_from_metrics(s0_dataset):
    output, batch = oracle(s0_dataset[0])
    output['source']['objectness_logits'][0, -1] = 10
    result = summarize(evaluate_batch(output, batch))
    assert result['count_accuracy'] == result['task_success'] == 0 and not result['s0_gate']
    output, batch = oracle(s0_dataset[0])
    output['source']['uv'][0, 0] += .1
    rows = evaluate_batch(output, batch)
    assert 'source_identity' in rows[0]['errors']
    assert not summarize(rows)['s0_gate']


def test_target_collisions_and_size_oracle_do_not_use_predicted_shrinking(s0_dataset):
    output, batch = oracle(s0_dataset[0])
    output['targets'][-1]['uv'][0,:2] = .5
    row = evaluate_batch(output, batch)[0]
    assert 'collision' in row['errors'] and not row['raw_geometry_success']
    # Drastically shrinking source sizes fails source envelope identity, not safety approval.
    output['source']['size'][0,:2] = .001
    assert not evaluate_batch(output, batch)[0]['task_success']


def test_prediction_keep_tolerance_not_dataset_exact_copy(s0_dataset):
    r = next(r for r in s0_dataset.records if r['instruction']['keep_all'])
    scene = scene_from_dict(r['target'])
    objects = list(scene.objects)
    o = objects[0]
    objects[0] = replace(o, target_pose=Pose(o.source_pose.x_m+.002, o.source_pose.y_m, o.source_pose.theta_rad))
    assert not semantic_errors(objects, r)
    objects[0] = replace(o, target_pose=Pose(o.source_pose.x_m+.004, o.source_pose.y_m, o.source_pose.theta_rad))
    assert semantic_errors(objects, r) == ['keep']
