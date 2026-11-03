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
    result = summarize(rows, s0_dataset.records)
    assert result['count_accuracy'] == result['correspondence_coverage'] == result['raw_geometry_success'] == result['task_success'] == 1
    assert result['s0_gate'] and result['failure_counts'] == {}
    assert result['language_sensitivity']['eligible_pairs'] == 7
    assert result['language_sensitivity']['distinguished_pairs'] == 7
    partial = result['by_condition']['partial_selection']
    assert partial['single_object']['n'] == 6 and partial['multi_object']['n'] == 2


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


# Self-contained geometry fixtures keep diagnostic regressions runnable without
# downloaded images, pilot artifacts, pretrained assets or a Teacher call.
def diagnostic_sample(scene, sample_id='horizontal', *, axis='horizontal', mode='full', selected=None):
    from dataclasses import asdict
    from knolling_dataset.schema import Instruction, scene_dict
    from knolling_dataset.english import english_instruction
    from knolling_model.data import supervision
    selected = tuple(o.object_id for o in scene.objects) if selected is None else selected
    locked = tuple(o.object_id for o in scene.objects if o.object_id not in selected)
    ins = english_instruction(scene, Instruction('fixture', 'en', selected, locked,
                                                 axis_alignment=axis, keep_all=not selected))
    target = replace(scene, objects=tuple(replace(o, target_pose=Pose(
        .20+i*(.08 if axis == 'horizontal' else .05), .20,
        0. if axis == 'horizontal' else torch.pi/2)) for i, o in enumerate(scene.objects)))
    record = {'sample_id': sample_id, 'scene_family_id': scene.family_id,
              'mode': mode, 'instruction': asdict(ins), 'source': scene_dict(scene),
              'target': scene_dict(target), 'source_render_dir': 'same-rgb',
              'file_hashes': {'same-rgb/rgb.png': 'same-image-hash'}}
    ids, labels = supervision(record)
    return {'supervision': labels, 'metadata': {'sample_id': sample_id, 'object_ids': ids, 'record': record}}


def report_samples(samples, outputs=None):
    rows = []
    for i, sample in enumerate(samples):
        out, batch = oracle(sample)
        rows.extend(evaluate_batch(outputs[i] if outputs is not None else out, batch))
    return summarize(rows, [s['metadata']['record'] for s in samples]), rows


def test_same_rgb_counterfactual_detects_language_and_counts_exclusions(simple_scene):
    a = diagnostic_sample(simple_scene)
    b = diagnostic_sample(simple_scene, 'vertical', axis='vertical')
    result, _ = report_samples([a, b])
    sensitivity = result['language_sensitivity']
    assert sensitivity['candidate_pairs'] == sensitivity['eligible_pairs'] == 1
    assert sensitivity['own_success_pairs'] == sensitivity['distinguished_pairs'] == 1
    assert sensitivity['sensitivity'] == 1 and sensitivity['excluded_pairs'] == 0
    # A model that ignores language and repeats A for both commands must fail.
    output, _ = oracle(a)
    repeated, _ = report_samples([a, b], [output, output])
    assert repeated['language_sensitivity']['sensitivity'] == 0
    # Different phrasing with identical conditions/targets supplies no contrast.
    from copy import deepcopy
    paraphrase = deepcopy(a)
    paraphrase['metadata']['sample_id'] = 'paraphrase'
    paraphrase['metadata']['record']['sample_id'] = 'paraphrase'
    paraphrase['metadata']['record']['instruction']['text'] += ' Please.'
    excluded, _ = report_samples([a, paraphrase])
    assert excluded['language_sensitivity']['excluded_pairs'] == 1
    assert excluded['language_sensitivity']['eligible_pairs'] == 0
    assert excluded['language_sensitivity']['sensitivity'] is None
    assert not excluded['language_sensitivity']['evaluated']
    # Pairing uses the RGB content hash, not merely shared family metadata.
    b['metadata']['record']['file_hashes']['same-rgb/rgb.png'] = 'different-image'
    unpaired, _ = report_samples([a, b])
    assert unpaired['language_sensitivity']['candidate_pairs'] == 0


def test_partial_single_multi_source_coverage_keep_and_noop(simple_scene):
    from knolling_dataset.schema import ObjectState
    third = ObjectState('o3', simple_scene.objects[0].asset, 'green', Pose(.45, .30, 0))
    scene = replace(simple_scene, objects=(simple_scene.objects[0],
                                           replace(simple_scene.objects[1], color='red'), third))
    single = diagnostic_sample(scene, 'single', mode='partial', selected=('o1',))
    multi = diagnostic_sample(scene, 'multi', mode='partial', selected=('o1', 'o2'))
    noop = diagnostic_sample(scene, 'noop', mode='no_op', selected=())
    # Fixed GT targets use source poses; these are hand-built test annotations.
    for sample in (single, multi, noop):
        record = sample['metadata']['record']
        locked = record['instruction']['locked_ids']
        for i, obj in enumerate(scene.objects):
            if obj.object_id in locked:
                sample['supervision']['target_xy'][i] = sample['supervision']['source_xy'][i]
                sample['supervision']['target_angle'][i] = sample['supervision']['source_angle'][i]
                record['target']['objects'][i]['target_pose'] = record['source']['objects'][i]['source_pose'].copy()
    result, rows = report_samples([single, multi, noop])
    conditions = result['by_condition']
    assert conditions['partial_selection']['single_object']['n'] == 1
    assert conditions['partial_selection']['multi_object']['n'] == 1
    assert conditions['partial_selection']['single_object']['task_success'] == 1
    assert conditions['partial_selection']['multi_object']['task_success'] == 1
    assert result['by_mode']['no_op']['task_success'] == 1
    assert result['keep_violation'] == {'n': 2, 'checked_n': 2, 'unchecked_n': 0,
                                        'violations': 0, 'scene_rate': 0., 'evaluated': True}
    output, batch = oracle(single)
    output['targets'][-1]['uv'][0, 2, 0] += .01  # 6 mm movement of locked o3.
    violated = summarize(evaluate_batch(output, batch))
    assert violated['keep_violation']['violations'] == 1
    assert violated['keep_violation']['scene_rate'] == 1
    output['source']['objectness_logits'][0, -1] = 10
    failed = summarize(evaluate_batch(output, batch))
    assert failed['keep_violation']['unchecked_n'] == 1
    assert failed['by_condition']['partial_selection']['single_object']['task_success'] == 0
    assert failed['refinement'][0]['geometry_success'] == 0
    assert failed['refinement'][0]['evaluated_n'] == 0


def test_missing_partial_coverage_is_explicit_and_conditions_include_failures(simple_scene):
    sample = diagnostic_sample(simple_scene)
    result, _ = report_samples([sample])
    conditions = result['by_condition']
    expected = {'object_count', 'shape_composition', 'size_composition_mm', 'color_composition',
                'axis_alignment', 'gap_m', 'region', 'english_instruction_type', 'mode',
                'group_by', 'sort_direction'}
    assert expected <= conditions.keys()
    for name in expected:
        assert sum(bucket['n'] for bucket in conditions[name].values()) == 1
        assert all(bucket['successes'] == bucket['n'] for bucket in conditions[name].values())
    organized = conditions['partial_source_layout']['organized']
    assert organized['n'] == 0 and organized['task_success'] is None and not organized['evaluated']
    output, batch = oracle(sample)
    output['source']['objectness_logits'][0, -1] = 10
    failed = summarize(evaluate_batch(output, batch))
    assert failed['by_condition']['object_count']['2']['task_success'] == 0
    assert failed['by_condition']['object_count']['2']['failure_counts'] == {'count': 1}


def test_refinement_summaries_report_all_stages_and_do_not_change_s0_gate(simple_scene):
    from copy import deepcopy
    sample = diagnostic_sample(simple_scene)
    output, batch = oracle(sample)
    output['targets'] = [deepcopy(stage) for stage in output['targets']]
    # Stages are allowed to regress; final-stage success is unchanged.
    output['targets'][0]['uv'] = output['targets'][0]['uv'].clone()
    output['targets'][0]['uv'][0, :2] = .5
    result = summarize(evaluate_batch(output, batch))
    assert result['s0_gate'] and result['task_success'] == 1
    assert [s['stage'] for s in result['refinement']] == [1, 2, 3]
    assert result['refinement'][0]['geometry_success'] == 0
    assert result['refinement'][0]['geometry_failure_counts']['collision'] == 1
    assert result['refinement'][0]['position_mm']['p95'] > 1
    assert result['refinement'][2]['position_mm']['p95'] < 1
    assert result['refinement'][2]['geometry_success'] == 1


def test_partial_source_structure_reports_observed_and_absent_groups(simple_scene):
    simple_scene = replace(simple_scene, objects=(simple_scene.objects[0],
                                                  replace(simple_scene.objects[1], color='red')))
    organized = diagnostic_sample(simple_scene, 'organized', mode='partial', selected=('o1',))
    first, second = simple_scene.objects
    scattered_scene = replace(simple_scene, objects=(first, replace(second, source_pose=Pose(.28, .20, .2))))
    scattered = diagnostic_sample(scattered_scene, 'scattered', mode='partial', selected=('o1',))
    result, _ = report_samples([organized, scattered])
    coverage = result['by_condition']['partial_source_layout']
    assert coverage['organized']['n'] == coverage['scattered']['n'] == 1
    assert coverage['organized']['evaluated'] and coverage['scattered']['evaluated']


def test_evaluate_includes_pair_diagnostics_and_restores_training_mode(simple_scene):
    import json
    from knolling_model.evaluation import evaluate
    samples = [diagnostic_sample(simple_scene),
               diagnostic_sample(simple_scene, 'vertical', axis='vertical')]
    outputs = [oracle(s)[0] for s in samples]
    batches = [oracle(s)[1] for s in samples]
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index = 0
        def forward(self):
            assert not self.training
            output = outputs[self.index]
            self.index += 1
            return output
    model = Predictor().train()
    metrics, rows = evaluate(model, batches, 'cpu')
    assert model.training
    assert metrics['language_sensitivity']['distinguished_pairs'] == 1
    json.dumps({'metrics': metrics, 'samples': rows}, allow_nan=False)
    # The original training mode survives a diagnostic failure as well.
    import pytest
    with pytest.raises(IndexError):
        evaluate(model, batches, 'cpu')
    assert model.training
