"""Score one frozen system on the sealed v2 holdout.

The script does not train, and it does not choose a threshold or a checkpoint.
Point --code-root at the tree to score. v1.0.0 and v2 use the same command.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.code_root.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts'), str(root / 'experiments' / 'end_to_end')]
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
    import run_phase1_pairwise_affine as runner
    import run_relation_stable as stable
    from knolling_model.experiments.size_refinement import poses_from_detection
    from knolling_model.recovery.data import recovery_supervision
    from score_end_to_end import blank, judge_scene, load_adopted

    data = args.data.resolve()
    if data.name == 'dataset' and 'generalization_20000' in str(data):
        raise SystemExit('the official 20,000-scene dataset is not this holdout')
    bundles = [json.loads(path.read_text()) for path in sorted((data / 'families').glob('*/family.json'))]
    records = [row for bundle in bundles for row in bundle['records']]
    if len(records) != 640 or len({row['scene_family_id'] for row in records}) != 128:
        raise SystemExit('expected 128 holdout families and 640 samples')
    if any(not row['sample_id'].startswith('v2hold_') for row in records):
        raise SystemExit('holdout sample ids must stay on the v2hold_ families')
    runner.setup()
    system, relation, model, head, _saved = load_adopted()
    if abs(float(system.threshold) - 0.5) > 1e-9:
        raise SystemExit('foreground threshold is frozen at 0.5')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        for number, record in enumerate(records):
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            image_path = data / record['source_render_dir'] / 'rgb.png'
            with Image.open(image_path) as image:
                pixels = stable.torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
            started = time.perf_counter()
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, model, pixels[None].to(stable.DEVICE),
                    [record['instruction']['text']], head)
            except ValueError:
                elapsed = time.perf_counter() - started
                row = blank(record, ['count'], len(identities), elapsed, '')
            else:
                elapsed = time.perf_counter() - started
                row = judge_scene(
                    output, objects, visual, record, identities, truth, elapsed, system.threshold, detail=False)
            payload = dict(
                sample_id=row['sample_id'], scene_family_id=record['scene_family_id'],
                mode=record['mode'], task=bool(row['task']), errors=list(row['errors']),
                objects=int(row['objects']), seconds=float(row['seconds']))
            stream.write(json.dumps(payload) + '\n')
            if (number + 1) % 160 == 0:
                print('progress', number + 1, flush=True)
    print('WROTE', args.output, flush=True)


if __name__ == '__main__':
    main()
