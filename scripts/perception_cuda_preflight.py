"""Actual RT-DETR CUDA preflight, zero optimizer updates, no Arrangement graph."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from knolling_dataset.builder import file_hash
from knolling_model.perception import checkpoint
from knolling_model.perception.data import collate
from knolling_model.perception.model import build_model
from knolling_model.perception.pipeline import catalog, gpu_guard, loss, setup, write_json


def run(config_path, output):
    if Path(sys.prefix).name != 'knolling':
        raise RuntimeError('use the existing knolling environment')
    config = json.loads(Path(config_path).read_text())
    pins = json.loads(Path(config['pretrained']).read_text())
    gpu_guard(config)
    setup(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = catalog(config)
    cpu = checkpoint.read(config['cpu_preflight_checkpoint'], fingerprint=data.fingerprint,
                          require_current_training_code=True)
    if cpu['pins'] != pins:
        raise ValueError('CPU preflight pretrained pins mismatch')
    dataset = data.dataset('train')
    counts = [len(r['source']['objects']) for r in dataset.records]
    batch = collate([dataset[counts.index(min(counts))], dataset[counts.index(max(counts))]])
    model = build_model(pins, num_denoising=config['num_denoising'])
    loading = model.detector.pretrained_loading_info
    if any(loading.values()):
        raise ValueError('incomplete pretrained loading')
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gpu_guard(config)  # Recheck immediately before allocation.
    model.cuda().eval()
    if '4090' not in torch.cuda.get_device_name():
        raise RuntimeError('this approved run requires the RTX 4090')
    torch.cuda.reset_peak_memory_stats()
    images = batch['images'].cuda()
    with torch.no_grad():
        output_rgb = model(images)
        native = model.detector(pixel_values=F.interpolate(images, (640, 640), mode='bilinear', align_corners=False))
        boxes = output_rgb['stages'][-1]['boxes']
        logits = output_rgb['stages'][-1]['logits']
        torch.testing.assert_close(boxes, native.pred_boxes, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(logits, native.logits, atol=3e-5, rtol=3e-5)
        native_box_error = float((boxes-native.pred_boxes).abs().max())
        frames = model.predict(images, batch['family_ids'], threshold=config['threshold'])
        if any(f.source_mode != 'predicted_rgb' for f in frames):
            raise ValueError('inference provenance mismatch')
    model.train()
    value, components = loss(model, batch, 'cuda')
    value.backward()
    gradients = {}
    for name, module in dict(backbone=model.detector.model.backbone, encoder=model.detector.model.encoder,
                             proposal=model.detector.model.enc_score_head, decoder=model.detector.model.decoder,
                             foreground=model.detector.class_embed, physical=model.physical).items():
        values = [p.grad for p in module.parameters() if p.grad is not None]
        total = sum(float(v.abs().sum()) for v in values)
        if not values or not all(torch.isfinite(v).all() for v in values) or total <= 0:
            raise ValueError(f'nonfinite/zero gradient: {name}')
        gradients[name] = total
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    # No optimizer is constructed or stepped, and frozen BN must not mutate buffers.
    for key, value_now in model.state_dict().items():
        if not torch.equal(value_now.detach().cpu(), original[key]):
            raise ValueError('preflight changed a model parameter/buffer')
    report = dict(schema='perception_cuda_preflight_v1', passed=True, optimizer_updates=0,
        pretrained_loading=loading, pretrained_weight_sha256=pins['detector_sha256'],
        config_sha256=file_hash(config_path), dataset_fingerprint=data.fingerprint,
        cpu_preflight_sha256=file_hash(config['cpu_preflight_checkpoint']),
        model_code=checkpoint.model_provenance(), code=checkpoint.code_provenance(),
        gpu_name=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
        native_box_max_error=native_box_error, model_state_unchanged=True,
        loss=float(value.detach()), losses={k: float(v.detach()) for k, v in components.items()},
        gradient_l1=gradients, gradient_norm=float(norm),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        elapsed_seconds=time.monotonic()-started)
    write_json(output/'cuda-report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.output)
