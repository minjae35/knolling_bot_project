"""Explicit CUDA preflight with disposable optimizer steps; never launches S0."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import torchvision
from huggingface_hub import HfApi, snapshot_download

from knolling_dataset.builder import file_hash
from knolling_dataset.migration import tree_hashes
from knolling_model import runner
from knolling_model.data import FamilySampler
from knolling_model.losses import S0Criterion
from verify_s0_data import verify


def state_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/s0.json')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = runner.read_config(args.config)
    runner.protect_dataset_output(args.report_dir, cfg)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    report = {'valid': False, 'training_started': False}
    try:
        assert Path(sys.prefix).resolve() == Path('/venv/knolling').resolve(), sys.prefix
        assert sys.version_info[:2] == (3, 12)
        assert torch.__version__ == '2.9.1+cu128', torch.__version__
        assert torchvision.__version__ == '0.24.1+cu128', torchvision.__version__
        runner.seed_everything(cfg['seed'], cfg['cpu_threads'])
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert '4090' in torch.cuda.get_device_name(0)
        assert torch.cuda.get_device_capability(0) == (8, 9)
        report['environment'] = {'python': sys.version, 'prefix': sys.prefix,
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'compute_capability': list(torch.cuda.get_device_capability(0)),
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory}
        (args.report_dir/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
        (args.report_dir/'requirements.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        print('Auditing all pilot files and all 32 S0 samples...', flush=True)
        before_data = tree_hashes(cfg['dataset'])
        report['dataset'] = verify(Path(cfg['dataset']), Path(cfg['selection']))
        print('Downloading/verifying pinned pretrained assets...', flush=True)
        pins = json.loads(Path(cfg['pretrained']).read_text())
        snapshot = Path(snapshot_download(pins['text_model'], revision=pins['text_revision'], token=False,
            allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                            'special_tokens_map.json', 'sentencepiece.bpe.model']))
        assert snapshot.name == pins['text_revision']
        info = HfApi(token=False).model_info(pins['text_model'], revision=pins['text_revision'], files_metadata=True)
        metadata = next(s for s in info.siblings if s.rfilename == 'model.safetensors')
        text_hash = file_hash(snapshot/'model.safetensors')
        assert text_hash == metadata.lfs.sha256, (text_hash, metadata.lfs)
        report['pretrained'] = {'pins': pins, 'snapshot': str(snapshot), 'text_sha256': text_hash,
                                'text_lfs_sha256_verified': True}
        rtdetr = pins.get('backend') == 'rtdetr_r18'
        if rtdetr:
            from knolling_model.rtdetr import load_detector
            detector = load_detector(pins, local_files_only=False)
            info = HfApi(token=False).model_info(pins['detector_model'],
                revision=pins['detector_revision'], files_metadata=True)
            metadata = next(s for s in info.siblings if s.rfilename == 'model.safetensors')
            assert metadata.lfs.sha256 == pins['detector_sha256']
            report['pretrained']['detector_loading'] = detector.pretrained_loading_info
            report['pretrained']['detector_lfs_sha256_verified'] = True
            del detector
        report['forward_backward'] = runner.preflight(cfg, args.report_dir/'forward-backward.json', 'cuda', offline=False)
        if not rtdetr:
            resnet = Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
            report['pretrained']['resnet_sha256'] = file_hash(resnet)
            assert report['pretrained']['resnet_sha256'].startswith('f37072fd')
        gc.collect(); torch.cuda.empty_cache()
        print('Testing real effective-batch AdamW steps and exact checkpoint replay...', flush=True)
        dataset, model, collator = runner.prepare(cfg, 'cuda', True)
        if rtdetr:
            from rtdetr_preflight_checks import stock_sampling_check, initial_detection, target_gradients_after_update
            report['sampling_compatibility'] = stock_sampling_check()
            report['source_detection'] = initial_detection(model, dataset, collator, args.report_dir)
        optimizer = runner.make_optimizer(model, cfg)
        criterion = S0Criterion()
        sampler = FamilySampler(dataset.records, cfg['seed'])
        groups = runner.update_groups(list(sampler), cfg['effective_batch'])
        initial = state_digest({n:p for n,p in model.named_parameters() if p.requires_grad})
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        def step(indices):
            model.train(); optimizer.zero_grad(set_to_none=True)
            terms = runner.backward_group(model, criterion, dataset, collator, indices, cfg['micro_batch'], 'cuda')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert all(p.grad is None for p in model.text_encoder.parameters())
            optimizer.step(); torch.cuda.synchronize()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert all(torch.isfinite(v).all() for state in optimizer.state.values()
                       for v in state.values() if isinstance(v, torch.Tensor))
            return terms, float(norm)
        first, norm = step(groups[0])
        assert initial != state_digest({n:p for n,p in model.named_parameters() if p.requires_grad})
        assert optimizer.state and all(state['step'].item() == 1 for state in optimizer.state.values())
        if rtdetr:
            report['target_gradients'] = target_gradients_after_update(model, dataset, collator, state_digest)
        cursor = {'epoch': 0, 'group': 1, 'updates': 1}
        evidence = runner.provenance(cfg)
        checkpoint = args.report_dir/'optimizer-step.pt'
        runner.save_checkpoint(checkpoint, model, optimizer, cursor, None, evidence)
        saved_hash = state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        second, _ = step(groups[1])
        reference = state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        loaded_cursor, _ = runner.load_checkpoint(checkpoint, model, optimizer, evidence)
        assert loaded_cursor == cursor
        assert saved_hash == state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        replay, _ = step(groups[1])
        assert second == replay, (second, replay)
        assert reference == state_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        report['optimizer_checkpoint'] = {'disposable_optimizer_steps': 3,
            'first_effective_scenes': len(groups[0]), 'replayed_tail_scenes': len(groups[1]),
            'first_loss': first, 'gradient_norm_before_clip': norm,
            'all_gradients_parameters_and_optimizer_state_finite': True,
            'checkpoint_load_exact': True, 'second_update_replay_exact': True,
            'checkpoint': str(checkpoint.resolve()), 'seconds': time.monotonic()-start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'allocated_with_optimizer_state_bytes': torch.cuda.memory_allocated(),
            'optimizer_state_bytes': sum(v.numel()*v.element_size() for state in optimizer.state.values()
                for v in state.values() if isinstance(v, torch.Tensor))}
        assert before_data == tree_hashes(cfg['dataset'])
        report['dataset_files_unchanged'] = len(before_data)
        report['valid'] = True
        print('GPU PREFLIGHT PASS', flush=True)
    except Exception as exc:
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        runner.write_json(args.report_dir/'report.json', report)


if __name__ == '__main__':
    main()
