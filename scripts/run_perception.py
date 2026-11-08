"""Standalone Perception CLI. Run with PYTHONPATH=src and the pinned neural env."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from knolling_model.perception import checkpoint
from knolling_model.perception.metrics import evaluate
from knolling_model.perception.model import build_model
from knolling_model.perception.pipeline import catalog, cpu_check, setup, train, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'cpu-check', 'evaluate', 'infer', 'train'))
    parser.add_argument('--config', default='configs/perception_cpu.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--image')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'development'), default='validation')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    pins = json.loads(Path(config['pretrained']).read_text())
    setup(config)
    if args.command == 'cpu-check':
        result = cpu_check(config, pins, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != 'baseline'}, indent=2))
    elif args.command == 'train':
        print(json.dumps(train(config, pins, args.output), indent=2))
    elif args.command == 'audit':
        result = catalog(config).report
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        if not args.checkpoint:
            parser.error('--checkpoint is required')
        # CPU-only CLI for inference/evaluation here; CUDA runs use the guarded train path.
        payload = checkpoint.read(args.checkpoint)
        if payload['pins'] != pins:
            raise ValueError('checkpoint pretrained provenance differs from config')
        model = build_model(pins, num_denoising=payload['config']['num_denoising']).eval()
        checkpoint.restore(payload, model)
        if args.command == 'infer':
            if not args.image:
                parser.error('--image is required')
            with Image.open(args.image) as image:
                if image.mode != 'RGB' or image.size != (576, 384):
                    raise ValueError('expected calibrated 576x384 BEV RGB; no automatic resize')
                pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()[None]/255
            result = model.predict(pixels, [Path(args.image).stem], threshold=config['threshold'])[0].to_dict()
            result['checkpoint_purpose'] = payload['purpose']
        else:
            data = catalog(config)
            if data.fingerprint != payload['dataset_fingerprint']:
                raise ValueError('evaluation dataset differs from checkpoint')
            metrics, frames = evaluate(model, data.dataset(args.split), threshold=config['threshold'], batch_size=config['batch_size'])
            result = dict(metrics=metrics, predictions=frames, checkpoint_purpose=payload['purpose'])
        write_json(args.output, result)
        print(json.dumps(result.get('metrics', {k: v for k, v in result.items() if k != 'predictions'}), indent=2))


if __name__ == '__main__':
    main()
