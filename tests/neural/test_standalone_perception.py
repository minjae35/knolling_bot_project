"""Contract, failure-path and numerical tests; tiny RT-DETR is verification only."""
import copy
import json
from pathlib import Path
import shutil

import pytest
import torch
from torchvision.ops import box_convert

from knolling_model.perception import checkpoint
from knolling_model.perception.adapter import oracle_frame, to_arrangement
from knolling_model.perception.data import SourceCatalog, source_truth
from knolling_model.perception.metrics import summarize
from knolling_model.perception.model import ObjectPerception, decode
from knolling_model.perception.pipeline import gpu_guard, train
from knolling_model.perception.representation import ObjectFrame
from knolling_model.recovery.contracts import detector_labels
from knolling_dataset.builder import file_hash
from test_recovery_perception import tiny_detector, truth


def test_representation_json_coordinate_roundtrip_and_pi_period():
    t = truth()
    t['source_angle'][0] = torch.pi
    frame = oracle_frame(t, ['left', 'right'], 'test')
    restored = ObjectFrame.from_dict(json.loads(json.dumps(frame.to_dict())))
    objects, sidecar = to_arrangement([restored], source_mode='oracle_gt')
    torch.testing.assert_close(objects.xy[0], t['source_xy'])
    torch.testing.assert_close(objects.size[0], t['size'])
    assert abs(float(objects.angle[0, 0])) < 1e-6
    assert sidecar[0]['object_ids'] == ['left', 'right']
    altered = frame.to_dict()
    altered['coordinates'] = dict(altered['coordinates'], length_unit='mm')
    with pytest.raises(ValueError, match='coordinate'):
        ObjectFrame.from_dict(altered)


def test_oracle_cannot_be_silently_used_in_predicted_adapter():
    frame = oracle_frame(truth(), ['a', 'b'], 'test')
    with pytest.raises(ValueError, match='provenance'):
        to_arrangement([frame])
    with pytest.raises(ValueError, match='capacity'):
        to_arrangement([frame], source_mode='oracle_gt', max_objects=1)


def test_empty_detections_have_only_invalid_padding():
    empty = ObjectFrame('empty', 'predicted_rgb', ())
    objects, sidecar = to_arrangement([empty])
    assert objects.valid.shape == (1, 1) and not objects.valid.any()
    assert objects.query_index.tolist() == [[-1]] and objects.features().count_nonzero() == 0
    assert sidecar[0]['object_ids'] == []


def test_duplicate_identity_and_nonfinite_state_rejected():
    frame = oracle_frame(truth(), ['a', 'b'], 'test')
    value = frame.to_dict()
    value['objects'][1]['object_id'] = 'a'
    with pytest.raises(ValueError, match='duplicate'):
        ObjectFrame.from_dict(value)
    value = frame.to_dict()
    value['objects'][0]['xy_m'] = [float('nan'), .2]
    with pytest.raises(ValueError, match='nonfinite'):
        ObjectFrame.from_dict(value)


def metric_row(scores=(.9, .8)):
    t = truth()
    return dict(family_id='family', truth=t,
        gt_boxes=box_convert(detector_labels([t])[0]['boxes'], 'cxcywh', 'xyxy'),
        boxes=box_convert(detector_labels([t])[0]['boxes'], 'cxcywh', 'xyxy'),
        scores=torch.tensor(scores), xy=t['source_xy'], size=t['size'],
        angle=t['source_angle']+torch.pi, color=t['color'], shape=t['shape'])


def test_detection_metrics_perfect_and_pi_invariant():
    result = summarize([metric_row()])
    assert result['count_accuracy'] == result['map_50_95'] == result['precision'] == result['recall'] == 1
    assert result['true_positives'] == 2
    assert result['errors']['position_mm']['mean'] == 0
    assert result['errors']['angle_deg']['maximum'] < 1e-4


def test_count_errors_false_positives_missing_detections_and_no_phony_pose():
    r = metric_row((.9, .1))
    result = summarize([r])
    assert result['count_accuracy'] == 0 and result['count_mae'] == 1
    assert result['true_positives'] == 1 and result['false_negatives'] == 1
    assert result['pose_coverage'] == .5
    r['scores'][:] = .1
    result = summarize([r])
    assert result['true_positives'] == 0 and result['false_negatives'] == 2
    assert result['errors']['position_mm']['mean'] is None
    r = metric_row()
    r['boxes'] = torch.cat((r['boxes'], r['boxes'][:1]))
    r['scores'] = torch.tensor([.9, .8, .95])
    for key in ('xy', 'size', 'angle', 'color', 'shape'):
        r[key] = torch.cat((r[key], r[key][:1]))
    result = summarize([r])
    assert result['false_positives'] == 1 and result['count_accuracy'] == 0


def test_bad_box_predictions_not_given_pose_credit_for_correct_count():
    r = metric_row()
    r['boxes'] = r['boxes']+.6
    result = summarize([r])
    assert result['count_accuracy'] == 1 and result['recall'] == result['ap50'] == 0
    assert result['false_positives'] == result['false_negatives'] == 2
    assert result['errors']['angle_deg']['mean'] is None


def test_source_supervision_never_reads_target_instruction_or_rules():
    source = dict(source=dict(objects=[dict(object_id='a', source_pose=dict(x_m=.2, y_m=.3, theta_rad=.5),
        asset=dict(a_m=.06, b_m=.03, shape='box'), color='red')]))
    ids, labels = source_truth(source)
    assert ids == ['a']
    assert not any('target' in key or 'selected' in key for key in labels)
    assert labels['source_xy'].tolist()[0] == pytest.approx([.2, .3])


def test_forward_rgb_only_and_decoded_identity_independent_of_gt():
    torch.set_num_threads(2)
    model = ObjectPerception(tiny_detector(), num_denoising=4).eval()
    images = torch.rand(1, 3, 384, 576)
    with torch.no_grad():
        output = model(images)
        all_objects = decode(output, ['rgb'], threshold=0)[0]
        assert len(all_objects.objects) == 7
        assert all(o.object_id == f'query_{o.query_index:03d}' for o in all_objects.objects)
        assert all_objects.source_mode == 'predicted_rgb'
        empty = model.predict(images, ['rgb'], threshold=1)[0]
        assert len(empty.objects) == 0
    with pytest.raises(TypeError):
        model(images, [truth()])
    with pytest.raises(ValueError, match='evaluation'):
        model.supervised(images, [truth()])


def test_checkpoint_next_optimizer_update_rng_replay_and_tamper(tmp_path, monkeypatch):
    torch.manual_seed(27)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.GELU(), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    def step():
        optimizer.zero_grad(set_to_none=True)
        value = model(torch.rand(2, 3)).square().sum()
        value.backward()
        optimizer.step()
    step()
    path = tmp_path/'state.pt'
    checkpoint.save(path, model, optimizer, config={'test': 1}, pins={}, fingerprint='dataset',
                    update=1, elapsed=1., metrics=None, purpose='unit_test')
    saved = checkpoint.read(path, fingerprint='dataset', config={'test': 1})
    with monkeypatch.context() as change:
        change.setattr(checkpoint, 'code_provenance', lambda: dict(saved['code'], **{'perception/adapter.py': 'future_adapter'}))
        assert checkpoint.read(path)['architecture'] == saved['architecture']
        with pytest.raises(ValueError, match='training code'):
            checkpoint.read(path, require_current_training_code=True)
    step()
    expected = copy.deepcopy(model.state_dict())
    checkpoint.restore(saved, model, optimizer, restore_random=True)
    step()
    for key in expected:
        torch.testing.assert_close(model.state_dict()[key], expected[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='dataset'):
        checkpoint.read(path, fingerprint='different')
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='SHA-256'):
        checkpoint.read(path)


def test_training_and_gpu_allocation_guards_never_start_work(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before invoking GPU query')
    monkeypatch.setattr('subprocess.check_output', forbidden)
    with pytest.raises(ValueError, match='disabled'):
        train({'allow_training': False}, {}, tmp_path/'run')
    with pytest.raises(ValueError, match='exclusive'):
        gpu_guard({'gpu_exclusive_window': None})
    assert not (tmp_path/'run').exists()


def test_gpu_window_rejects_busy_device_before_any_allocation(monkeypatch):
    import time
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: 'GPU-test, 123\n')
    config = dict(max_seconds=30, gpu_exclusive_window=dict(device_uuid='GPU-test',
                  start_unix=time.time()-5, end_unix=time.time()+500, authorization='unit test'))
    with pytest.raises(ValueError, match='occupied'):
        gpu_guard(config)


def test_fixed_budget_and_best_preservation_with_cpu_test_double(tmp_path, monkeypatch):
    """Exercise orchestration only; never allocate CUDA or train a real detector."""
    import knolling_model.perception.pipeline as pipeline
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def to(self, *args, **kwargs):
            return self
    model = Tiny()
    config = json.loads(Path('configs/perception_cpu.json').read_text())
    config.update(allow_training=True, device='cuda', max_updates=3, max_seconds=30,
                  final_reserve_seconds=2, evaluate_every=1, cpu_preflight_checkpoint=str(tmp_path/'preflight.pt'))
    sample = dict(image=torch.zeros(3, 2, 2), truth=truth(), family_id='one', object_ids=['a','b'])
    class Catalog:
        fingerprint = 'unit_test_only'
        report = {'test_double': True}
        def dataset(self, split):
            return [sample]
    monkeypatch.setattr(pipeline, 'catalog', lambda c: Catalog())
    monkeypatch.setattr(pipeline, 'gpu_guard', lambda c: None)
    monkeypatch.setattr(pipeline, 'build_model', lambda *a, **kw: model)
    monkeypatch.setattr(pipeline, 'loss', lambda m, b, d: (-m.weight.sum(), {}))
    def evaluate_fake(m, *args, **kwargs):
        return dict(map_50_95=1/(1+float(m.weight.detach().abs().sum())),
                    count_accuracy=1., recall=1., errors={'position_mm': {'mean': 0.}}), []
    monkeypatch.setattr(pipeline, 'evaluate', evaluate_fake)
    sha = checkpoint.save(tmp_path/'preflight.pt', model, None, config={}, pins={}, fingerprint='unit_test_only',
        update=1, elapsed=1., metrics=None, purpose='disposable_cpu_gradient_checkpoint_verification_not_trained_model')
    (tmp_path/'cpu-report.json').write_text(json.dumps(dict(passed=True, checkpoint_sha256=sha)))
    result = pipeline.train(config, {}, tmp_path/'run')
    assert result['updates'] == 3 and result['best_update'] == 0
    assert checkpoint.read(tmp_path/'run/best.pt')['update'] == 0
    assert checkpoint.read(tmp_path/'run/last.pt')['update'] == 3
    assert float(model.weight.detach()) == 0.  # Best really reloaded.


@pytest.mark.skipif(not Path('/workspace/knolling_bot_project/artifacts/pilot_v1_3_en').exists(), reason='external immutable Pilot unavailable')
@pytest.mark.parametrize('corruption', ['pose', 'missing_mask', 'mask_id', 'family_alias', 'split_leak'])
def test_real_annotation_corruption_fails_without_generating_labels(tmp_path, corruption):
    """Mutate disposable copies of actual annotations; shared data is read-only."""
    root = Path('/workspace/knolling_bot_project/artifacts/pilot_v1_3_en')
    entry = json.loads((root/'manifest.jsonl').read_text().splitlines()[0])
    original = json.loads((root/entry['annotation']).read_text())
    record = copy.deepcopy(original)
    destination = tmp_path/record['source_render_dir']
    shutil.copytree(root/record['source_render_dir'], destination)
    if corruption == 'pose':
        record['source']['objects'][0]['source_pose']['x_m'] += .02
    elif corruption == 'missing_mask':
        (destination/'instance_mask.png').unlink()
    elif corruption == 'mask_id':
        import numpy as np
        from PIL import Image
        pixels = np.array(Image.open(destination/'instance_mask.png'))
        pixels[0, 0] = 99
        Image.fromarray(pixels.astype(np.uint16)).save(destination/'instance_mask.png')
        record['file_hashes'][record['source_render_dir']+'/instance_mask.png'] = file_hash(destination/'instance_mask.png')
    records = [record]
    if corruption in ('family_alias', 'split_leak'):
        alias = copy.deepcopy(record)
        alias['sample_id'] = 'alias'
        if corruption == 'family_alias':
            alias['scene_family_id'] = alias['source']['family_id'] = 'another-family'
        else:
            alias['split'] = alias['source']['split'] = 'validation'
        records.append(alias)
    entries = []
    for r in records:
        annotation = r['sample_id']+'.json'
        (tmp_path/annotation).write_text(json.dumps(r))
        entries.append(dict(annotation=annotation, annotation_sha256=file_hash(tmp_path/annotation),
            sample_id=r['sample_id'], scene_family_id=r['scene_family_id'], split=r['split']))
    (tmp_path/'manifest.jsonl').write_text('\n'.join(json.dumps(e) for e in entries)+'\n')
    rgb = original['source_render_dir']+'/rgb.png'
    recovery_record = {k: v for k, v in original.items() if k != 'objects'}
    recovery_record['file_hashes'] = {rgb: original['file_hashes'][rgb]}
    (tmp_path/'recovery.json').write_text(json.dumps(dict(schema='recovery_dataset_v3', records=[recovery_record])))
    with pytest.raises((ValueError, FileNotFoundError)):
        SourceCatalog(tmp_path, file_hash(tmp_path/'manifest.jsonl'), tmp_path/'recovery.json',
                      file_hash(tmp_path/'recovery.json'), check_projection=False)


@pytest.mark.skipif(not Path('/workspace/knolling_bot_project/artifacts/pilot_v1_3_en').exists(), reason='external immutable Pilot unavailable')
def test_actual_pilot_recovery_annotations_family_split_and_hash_failure():
    c = json.loads(Path('configs/perception_cpu.json').read_text())
    catalog = SourceCatalog(c['dataset'], c['manifest_sha256'], c['recovery_path'], c['recovery_sha256'])
    assert catalog.report['pilot_records'] == 130
    assert catalog.report['recovery_records'] == 76
    assert catalog.report['unique_sources'] == 100 and catalog.report['recovery_new_sources'] == 0
    assert len(catalog.dataset('train')) == 70 and len(catalog.dataset('validation')) == 10
    train_ids = {r['scene_family_id'] for r in catalog.dataset('train').records}
    val_ids = {r['scene_family_id'] for r in catalog.dataset('validation').records}
    assert not train_ids & val_ids
    with pytest.raises(ValueError, match='hash'):
        SourceCatalog(c['dataset'], 'wrong', c['recovery_path'], c['recovery_sha256'])
