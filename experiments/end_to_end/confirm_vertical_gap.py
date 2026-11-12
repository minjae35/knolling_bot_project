"""Before/after of opening a short vertical gap to the request. Does not open the test set."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'experiments' / 'end_to_end')]
if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    del os.environ['CUDA_VISIBLE_DEVICES']
os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import torch
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.geometry_decoder import apply_geometry, decode_batch, normalize_axis_angle
from knolling_model.recovery.data import recovery_supervision
from dev_splits import family_splits, fresh_split, heldout_validation
from score_end_to_end import DATA
from train_selector_full import blank, judge_scene, load_adopted, perceive
from confirm_axis_alignment import forward

OUT = Path('artifacts/experiments/selector_full/vertical_gap_confirm.json')
WATCH = ('vertical_gap', 'horizontal_gap', 'collision', 'clearance', 'boundary', 'requested_region')


def band_geometry(output, objects, goal):
    """Axis correction with the previous vertical band. No ground truth."""
    angle = normalize_axis_angle(output['angle'], objects.valid, goal)
    decoded, status = decode_batch(
        output['xy'], angle, output['row'], output['slot'],
        objects.size[..., 0], objects.size[..., 1], objects.color, goal, objects.valid)
    replaced = dict(output)
    replaced['xy'] = decoded
    replaced['angle'] = angle
    replaced['geometry_status'] = status
    return replaced


def compare(before, after):
    before_ok = {row['sample_id'] for row in before if row['task']}
    after_ok = {row['sample_id'] for row in after if row['task']}
    lost = [row for row in after if row['sample_id'] in before_ok and row['sample_id'] not in after_ok]

    def counts(rows):
        return Counter(name for row in rows if not row['task'] for name in row['errors'])

    before_counts, after_counts = counts(before), counts(after)
    sole = Counter(row['errors'][0] for row in after if not row['task'] and len(row['errors']) == 1)
    return dict(
        before=len(before_ok), after=len(after_ok), total=len(before),
        gained=len(after_ok - before_ok), lost=len(before_ok - after_ok),
        lost_errors=dict(Counter(name for row in lost for name in row['errors'])),
        errors={name: dict(before=before_counts[name], after=after_counts[name],
                           delta=after_counts[name] - before_counts[name]) for name in WATCH},
        sole_after=dict(sole.most_common(6)))


def score_split(system, relation, model, head, data, indices, name):
    before, after = [], []
    for done, index in enumerate(indices, start=1):
        record = data.records[index]
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = data[index]['image'][None].to(stable.DEVICE)
        try:
            visual = perceive(system, head, image)
            output, objects, goal = forward(
                system, relation, model, visual, [record['instruction']['text']])
        except ValueError:
            row = blank(record, ['count'], len(identities), 0, '')
            before.append(row)
            after.append(row)
            continue
        previous = band_geometry(output, objects, goal)
        corrected = apply_geometry(output, objects, goal)
        before.append(judge_scene(
            previous, objects, visual, record, identities, truth, 0, system.threshold, detail=False))
        after.append(judge_scene(
            corrected, objects, visual, record, identities, truth, 0, system.threshold, detail=False))
        if done % 200 == 0:
            print(name, done, sum(row['task'] for row in before), sum(row['task'] for row in after), flush=True)
    return compare(before, after)


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    validation = ExpandedDataset(DATA, 'validation')
    train = ExpandedDataset(DATA, 'train')
    val_full = [index for index, record in enumerate(validation.records) if is_full(record)]
    fit, dev, _blocked = fresh_split(train.records)
    _fit, _dev, hold = family_splits(train.records)
    splits = (
        ('validation', validation, val_full),
        ('dev512', train, dev),
        ('dev256', train, heldout_validation(fit)),
        ('holdout', train, hold))
    print('SPLITS', {name: len(indices) for name, _data, indices in splits}, flush=True)
    system, relation, model, head, _saved = load_adopted()
    report = {}
    for name, data, indices in splits:
        report[name] = score_split(system, relation, model, head, data, indices, name)
        print(name, json.dumps(report[name]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
