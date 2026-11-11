"""Show one supported scene: the input BEV beside the predicted arrangement.

This does not score the scene and does not claim a task success. Partial and
no-op records are refused instead of being drawn as a supported result.
"""
import argparse
import json
from pathlib import Path

import run_phase1_pairwise_affine as runner
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import instruction_supported, render_comparison
from diagnose_row_transition import multi_row
from pipeline import load, predict
from score_end_to_end import DATA


def candidates(records, index):
    if index is not None:
        return [index]
    return [candidate for candidate, record in enumerate(records)
            if multi_row(record) and instruction_supported(record)]


def main():
    parser = argparse.ArgumentParser(description='Run the adopted knolling system on one scene')
    parser.add_argument('--output', default='artifacts/experiments/end_to_end/demo.png')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    runner.setup()
    data = ExpandedDataset(DATA, args.split)
    chosen = candidates(data.records, args.index)
    if not chosen:
        raise SystemExit('no supported full arrangement in this split')
    if any(not instruction_supported(data.records[index]) for index in chosen):
        raise SystemExit('refusing to demo an unsupported instruction: partial and no-op are not supported')
    system, relation, row_model, head, checkpoint = load()
    skipped = 0
    prediction = item = record = index = None
    for index in chosen:
        record = data.records[index]
        item = data[index]
        try:
            prediction = predict(system, relation, row_model, head, item['image'], record['instruction']['text'])
            break
        except ValueError:
            skipped += 1
            prediction = None
    if prediction is None:
        raise SystemExit('no supported scene produced 1 to 8 detections')
    image = render_comparison(item['image'], prediction['poses'], supported=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(json.dumps(dict(
        sample_id=record['sample_id'], split=args.split, index=index,
        instruction=record['instruction']['text'], mode=record['mode'],
        supported=True, not_supported=['partial', 'no_op'],
        objects=prediction['objects'], seconds=prediction['seconds'],
        detection_failures_skipped=skipped,
        poses=prediction['poses'], checkpoint_update=checkpoint.get('update'),
        image=str(path), task_success=None), indent=2), flush=True)


if __name__ == '__main__':
    main()
