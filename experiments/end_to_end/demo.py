"""Show one supported scene: the original BEV beside a Blender render.

The render uses source assets at the predicted poses. Partial and no-op
records are refused. Dataset target poses are not read.
"""
import argparse
import json
from pathlib import Path

import run_phase1_pairwise_affine as runner
from PIL import Image

from knolling_dataset.rendering import make_job, render_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.inference import instruction_supported
from knolling_model.experiments.predicted_view import associate_source_assets, scene_at_predicted_poses
from knolling_model.experiments.size_refinement import poses_from_detection
from diagnose_row_transition import multi_row
from demo_counts import compose
from pipeline import load
from score_end_to_end import DATA
import run_relation_stable as stable


def candidates(records, index):
    if index is not None:
        return [index]
    return [candidate for candidate, record in enumerate(records)
            if multi_row(record) and is_full(record)]


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
    association = item = record = index = None
    for index in chosen:
        record = data.records[index]
        item = data[index]
        image = item['image'][None].to(stable.DEVICE)
        try:
            output, objects, visual = poses_from_detection(
                system, relation, row_model, image, [record['instruction']['text']], head)
            association = associate_source_assets(record, visual, output, objects, system.threshold)
            view = scene_at_predicted_poses(association)
            break
        except ValueError:
            skipped += 1
            association = None
    if association is None:
        raise SystemExit('no supported scene produced a corresponded arrangement')
    folder = Path('artifacts/experiments/end_to_end/renders')/record['sample_id']
    job = make_job(view, 'target_pose', folder, record['sample_id'], render_seed=view.seed)
    render_batch([job], folder/'work')
    before = Image.open(data.root/record['source_render_dir']/'rgb.png').convert('RGB')
    after = Image.open(folder/'rgb.png').convert('RGB')
    count = len(record['source']['objects'])
    panel = compose(before, after, record['instruction']['text'], count)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    print(json.dumps(dict(
        sample_id=record['sample_id'], split=args.split, index=index,
        instruction=record['instruction']['text'], mode=record['mode'],
        supported=True, not_supported=[],
        objects=count, matched=[row['object_id'] for row in association['matched']],
        omitted=association['omitted'], detection_failures_skipped=skipped,
        checkpoint_update=checkpoint.get('update'), used_gt_target_pose=False,
        image=str(path), task_success=None), indent=2), flush=True)


if __name__ == '__main__':
    main()
