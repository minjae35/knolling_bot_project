"""Read-only Best evaluation and predicted-object adapter compatibility evidence."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, box_iou

from knolling_dataset.builder import file_hash
from knolling_model.perception import checkpoint
from knolling_model.perception.adapter import to_arrangement
from knolling_model.perception.metrics import distribution, evaluate
from knolling_model.perception.model import build_model
from knolling_model.perception.pipeline import catalog, setup, write_json
from knolling_model.perception.representation import ObjectFrame
from knolling_model.recovery.contracts import detector_labels


def adapter_check(values):
    frames = [ObjectFrame.from_dict(v) for v in values]
    objects, sidecar = to_arrangement(frames, source_mode='predicted_rgb')
    capacity_errors = []
    for row, frame in enumerate(frames):
        n = len(frame.objects)
        if int(objects.valid[row].sum()) != n:
            raise ValueError('adapter changed predicted object count')
        if n:
            for key, value in [('xy', [o.xy_m for o in frame.objects]),
                               ('size', [o.size_ab_m for o in frame.objects]),
                               ('query_index', [o.query_index for o in frame.objects])]:
                torch.testing.assert_close(getattr(objects, key)[row, :n],
                    torch.tensor(value, dtype=getattr(objects, key).dtype), atol=0, rtol=0)
            torch.testing.assert_close(objects.angle[row, :n], torch.tensor([o.theta_rad for o in frame.objects]))
        try:
            to_arrangement([frame], max_objects=8)
        except ValueError:
            capacity_errors.append(dict(frame_id=frame.frame_id, predicted_count=n))
    features = objects.features()
    return dict(compatible=True, source_mode='predicted_rgb', frames=len(frames),
        predicted_objects=sum(len(f.objects) for f in frames), feature_dimension=features.shape[-1],
        counts_preserved=True, xy_size_query_identity_exact=True, angles_preserved=True,
        max_objects_8_compatible=not capacity_errors, capacity_errors=capacity_errors,
        oracle_used=False, arrangement_executed=False, sidecar=sidecar)


def size_gate_errors(values, dataset):
    """Same max(|da|,|db|) per-object size definition as the existing S0 evaluator."""
    lookup = {dataset[i]['family_id']: dataset[i]['truth'] for i in range(len(dataset))}
    errors = []
    for value in values:
        frame = ObjectFrame.from_dict(value)
        if not frame.objects:
            continue
        truth = lookup[frame.frame_id]
        matrix = box_iou(torch.tensor([o.box_xyxy_image for o in frame.objects]),
                         box_convert(detector_labels([truth])[0]['boxes'], 'cxcywh', 'xyxy')).numpy()
        p, g = linear_sum_assignment(-((matrix >= .5).astype(float)*1000+matrix))
        for i, j in zip(p, g):
            if matrix[i, j] >= .5:
                errors.append(float((torch.tensor(frame.objects[i].size_ab_m)-truth['size'][j]).abs().max()*1000))
    return distribution(errors)


def run(run_path, output, device):
    root, output = Path(run_path), Path(output)
    final = json.loads((root/'result.json').read_text())
    if final['status'] != 'completed':
        raise ValueError('production run has not completed normally')
    best = root/'training/best.pt'
    payload = checkpoint.read(best)
    config, pins = payload['config'], payload['pins']
    if device == 'cuda':
        # Read-only evaluation after the training process releases its GPU.
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if busy.strip():
            raise RuntimeError('GPU is occupied; do not overlap another process')
    setup(config)
    output.mkdir(parents=True, exist_ok=False)
    data = catalog(config)
    if data.fingerprint != payload['dataset_fingerprint']:
        raise ValueError('Best dataset provenance mismatch')
    model = build_model(pins, num_denoising=config['num_denoising']).to(device).eval()
    checkpoint.restore(payload, model)
    reports, adapter = {}, {}
    for split in ('validation', 'train'):
        dataset = data.dataset(split)
        metrics, values = evaluate(model, dataset, device=device,
            batch_size=config['batch_size'], threshold=config['threshold'])
        metrics['source_size_max_component_mm'] = size_gate_errors(values, dataset)
        write_json(output/(split+'-metrics.json'), metrics)
        write_json(output/(split+'-predictions.json'), values)
        reports[split], adapter[split] = metrics, adapter_check(values)
    validation = dict(reports['validation'])
    validation.pop('source_size_max_component_mm')
    replay_exact = validation == payload['metrics']
    if device == 'cuda' and not replay_exact:
        raise ValueError('Best CUDA validation differs from saved measurements')
    v = reports['validation']
    def bounded(key, limit):
        value = v['errors'][key]['p95']
        return value is not None and value <= limit
    source_gate = (v['count_accuracy'] == 1 and v['pose_coverage'] == 1
        and v['false_positives'] == v['false_negatives'] == 0
        and bounded('position_mm', 1) and bounded('angle_deg', 2)
        and v['source_size_max_component_mm']['p95'] is not None
        and v['source_size_max_component_mm']['p95'] <= 1
        and v['color_accuracy_on_matches'] == v['shape_accuracy_on_matches'] == 1)
    summary = dict(schema='perception_demo_assessment_v1', optimizer_updates_this_evaluation=0,
        production_training_runs=1, training_updates=final['updates'], best_update=payload['update'],
        best_checkpoint=str(best.resolve()), best_sha256=file_hash(best),
        last_sha256=file_hash(root/'training/last.pt'), device=device, validation_replay_exact=replay_exact,
        validation=reports['validation'], train=reports['train'], adapter=adapter,
        existing_source_precision_gate_passed=bool(source_gate),
        source_gate_reference='recovery/stages.py: count/coverage=1, no FP/FN, position/size P95<=1mm, angle P95<=2deg, semantics=1',
        demo_ready=bool(source_gate and adapter['validation']['max_objects_8_compatible']),
        arrangement_executed=False, end_to_end_success_measured=False,
        test_evaluated=False, automatic_retraining=False)
    write_json(output/'demo-assessment.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('validation', 'train', 'adapter')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    run(args.run, args.output, args.device)
