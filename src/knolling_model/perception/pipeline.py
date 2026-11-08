"""Bounded standalone training and real CPU preflight, with no Arrangement graph."""
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm

from . import checkpoint
from .adapter import oracle_frame, to_arrangement
from .data import SourceCatalog, collate
from .metrics import evaluate
from .model import build_model
from .representation import ObjectFrame


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def setup(config):
    if config.get('schema') != 'standalone_perception_run_v1' or config['cpu_threads'] < 1:
        raise ValueError('unsupported perception config')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def catalog(config):
    return SourceCatalog(config['dataset'], config['manifest_sha256'], config['recovery_path'], config['recovery_sha256'])


def optimizer_for(model, config):
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith('detector.model.backbone.') else other).append(parameter)
    return torch.optim.AdamW([dict(params=backbone, lr=config['backbone_lr']), dict(params=other, lr=config['lr'])],
                             weight_decay=config['weight_decay'])


def loss(model, batch, device):
    truths = [{k: v.to(device) for k, v in t.items()} for t in batch['truths']]
    output = model.supervised(batch['images'].to(device), truths)
    losses, _ = model.detection_loss(output, truths)
    value = sum(losses.values())
    if not torch.isfinite(value):
        raise ValueError('nonfinite perception loss')
    return value, losses


def cpu_check(config, pins, output):
    """One disposable real optimizer update; no training or production best claim."""
    setup(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = catalog(config)
    write_json(output/'annotation-audit.json', data.report)
    for split in ('train', 'validation', 'test', 'development'):
        dataset = data.dataset(split)
        for i in range(len(dataset)):
            dataset[i]  # Real loader traversal of every source, not just one batch.
    train, validation = data.dataset('train'), data.dataset('validation')
    model = build_model(pins, num_denoising=config['num_denoising']).cpu()
    optimizer = optimizer_for(model, config)
    baseline, frames = evaluate(model, validation, threshold=config['threshold'], batch_size=config['batch_size'])
    baseline['initialization'] = 'pinned pretrained detector geometry; new random foreground and physical heads'
    write_json(output/'validation-initial.json', baseline)
    write_json(output/'validation-predictions.json', frames)
    for value in frames:
        frame = ObjectFrame.from_dict(json.loads(json.dumps(value)))
        to_arrangement([frame])
    checkpoint.save(output/'initial.pt', model, None, config=config, pins=pins, fingerprint=data.fingerprint,
                    update=0, elapsed=time.monotonic()-started, metrics=baseline, purpose='untrained_baseline_not_best')
    # Real minimum/maximum source counts probe variable-length labels and DN.
    counts = [len(r['source']['objects']) for r in train.records]
    batch = collate([train[counts.index(min(counts))], train[counts.index(max(counts))]])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    value, components = loss(model, batch, 'cpu')
    value.backward()
    gradients = {}
    for name, module in dict(backbone=model.detector.model.backbone, encoder=model.detector.model.encoder,
                             proposal_score=model.detector.model.enc_score_head,
                             proposal_box=model.detector.model.enc_bbox_head,
                             decoder=model.detector.model.decoder,
                             foreground=model.detector.class_embed, physical=model.physical,
                             denoising=model.detector.model.denoising_class_embed).items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        norm = sum(float(g.abs().sum()) for g in grads)
        if not grads or not all(torch.isfinite(g).all() for g in grads) or norm <= 0:
            raise ValueError(f'missing/nonfinite/zero gradient: {name}')
        gradients[name] = norm
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        expected = model(batch['images'])
    sha = checkpoint.save(output/'cpu-preflight.pt', model, optimizer, config=config, pins=pins,
        fingerprint=data.fingerprint, update=1, elapsed=time.monotonic()-started, metrics=None,
        purpose='disposable_cpu_gradient_checkpoint_verification_not_trained_model')
    saved = checkpoint.read(output/'cpu-preflight.pt', fingerprint=data.fingerprint, config=config)
    # Strict restore into the real graph; perturb parameters first to catch no-op loads.
    with torch.no_grad():
        next(model.parameters()).add_(.1)
    checkpoint.restore(saved, model, optimizer, restore_random=True)
    with torch.no_grad():
        actual = model(batch['images'])
    for key in expected['source']:
        torch.testing.assert_close(expected['source'][key], actual['source'][key], rtol=0, atol=0)
    for i, truth in enumerate(batch['truths']):
        frame = oracle_frame(truth, batch['object_ids'][i], batch['family_ids'][i])
        objects, _ = to_arrangement([frame], source_mode='oracle_gt')
        torch.testing.assert_close(objects.xy[0, :len(truth['source_xy'])], truth['source_xy'])
        try:
            to_arrangement([frame])
        except ValueError:
            pass
        else:
            raise AssertionError('oracle accepted as predicted input')
    report = dict(schema='perception_cpu_verification_v1', passed=True, device='cpu',
        optimizer_updates=1, purpose='disposable verification only; GPU training not run',
        sources_loaded=len(data.records), dataset_fingerprint=data.fingerprint,
        gradient_l1=gradients, gradient_norm=float(norm),
        loss=float(value.detach()), loss_components={k: float(v.detach()) for k, v in components.items()},
        checkpoint_sha256=sha, checkpoint_reload_max_error=0., adapter_roundtrip=True,
        baseline=baseline, elapsed_seconds=time.monotonic()-started)
    write_json(output/'cpu-report.json', report)
    return report


def gpu_guard(config):
    """Only an explicitly allocated window permits GPU work, even if currently idle."""
    window = config.get('gpu_exclusive_window')
    if not isinstance(window, dict) or set(window) != {'device_uuid', 'start_unix', 'end_unix', 'authorization'}:
        raise ValueError('GPU training requires a separately confirmed exclusive time window')
    now = time.time()
    if (not window['authorization'] or not window['start_unix'] <= now < window['end_unix']
            or window['end_unix']-now < config['max_seconds']+120):
        raise ValueError('GPU allocation does not cover the fixed training/evaluation budget')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == window['device_uuid'] for line in busy.splitlines()):
        raise ValueError('allocated GPU is already occupied; training not started')
    from os import environ
    if environ.get('CUDA_VISIBLE_DEVICES') != window['device_uuid']:
        raise ValueError('CUDA_VISIBLE_DEVICES must select exactly the allocated GPU UUID')
    return window


def train(config, pins, output):
    """One finite run; no retries, search, auto-resume or implicit budget extension."""
    if config.get('allow_training') is not True:
        raise ValueError('training disabled in this config')
    if (config['device'] != 'cuda' or type(config['max_updates']) is not int or config['max_updates'] <= 0
            or not math.isfinite(config['max_seconds']) or config['max_seconds'] <= 0
            or config['evaluate_every'] <= 0 or config['batch_size'] < 1
            or not 0 < config['final_reserve_seconds'] < config['max_seconds']):
        raise ValueError('expected explicit CUDA run with finite update/time budget')
    if not config.get('cpu_preflight_checkpoint'):
        raise ValueError('a passing real CPU preflight checkpoint is required before training')
    preflight_path = Path(config['cpu_preflight_checkpoint'])
    preflight = checkpoint.read(preflight_path, require_current_training_code=True)
    preflight_report = json.loads((preflight_path.parent/'cpu-report.json').read_text())
    if (preflight['purpose'] != 'disposable_cpu_gradient_checkpoint_verification_not_trained_model'
            or preflight['pins'] != pins or preflight_report.get('passed') is not True
            or preflight_report['checkpoint_sha256'] != preflight_path.with_suffix('.pt.sha256').read_text().strip()):
        raise ValueError('CPU preflight evidence/provenance mismatch')
    gpu_guard(config)  # Before model construction/any CUDA allocation.
    setup(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'fixed-run-config.json', dict(config=config, pins=pins))
    started = time.monotonic()
    data = catalog(config)
    if preflight['dataset_fingerprint'] != data.fingerprint:
        raise ValueError('CPU preflight dataset differs from training data')
    write_json(output/'annotation-audit.json', data.report)
    training, validation = data.dataset('train'), data.dataset('validation')
    model = build_model(pins, num_denoising=config['num_denoising']).to('cuda')
    optimizer = optimizer_for(model, config)
    best_score, update, epoch = None, 0, 0
    def assess():
        nonlocal best_score
        metrics, frames = evaluate(model, validation, device='cuda', batch_size=config['batch_size'], threshold=config['threshold'])
        write_json(output/f'validation-{update:06d}.json', metrics)
        write_json(output/f'predictions-{update:06d}.json', frames)
        # Validation only; diagnostic GT assignment never participates in selection.
        position = metrics['errors']['position_mm']['mean']
        score = (metrics['map_50_95'], metrics['count_accuracy'], metrics['recall'],
                 -position if position is not None else -float('inf'))
        if best_score is None or score > best_score:
            best_score = score
            checkpoint.save(output/'best.pt', model, optimizer, config=config, pins=pins,
                fingerprint=data.fingerprint, update=update, elapsed=time.monotonic()-started,
                metrics=metrics, purpose='validation_selected_best')
    assess()  # Preserve initial state too; later updates cannot erase the best.
    stop = 'update_budget'
    bar = tqdm(total=config['max_updates'], desc='Perception RT-DETR + Pose', unit='update',
               mininterval=1, dynamic_ncols=True)
    try:
        while update < config['max_updates']:
            generator = torch.Generator().manual_seed(config['seed']+epoch)
            order = torch.randperm(len(training), generator=generator).tolist()
            for start in range(0, len(order), config['batch_size']):
                if update >= config['max_updates']:
                    break
                if time.monotonic()-started >= config['max_seconds']-config['final_reserve_seconds']:
                    stop = 'time_budget'
                    break
                model.train()
                batch = collate([training[i] for i in order[start:start+config['batch_size']]])
                optimizer.zero_grad(set_to_none=True)
                value, _ = loss(model, batch, 'cuda')
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                update += 1
                bar.update(1)
                bar.set_postfix(loss=f'{float(value.detach()):.4f}',
                                elapsed_s=f'{time.monotonic()-started:.0f}', refresh=False)
                with (output/'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=update, epoch=epoch, loss=float(value.detach()),
                        gradient_norm=float(norm), elapsed_seconds=time.monotonic()-started), allow_nan=False)+'\n')
                if update % config['evaluate_every'] == 0:
                    assess()
                    tqdm.write(f'Validation update={update} best_mAP={best_score[0]:.5f} '
                               f'best_count={best_score[1]:.3f}')
            epoch += 1
            if stop == 'time_budget':
                break
        assess()
        checkpoint.save(output/'last.pt', model, optimizer, config=config, pins=pins,
            fingerprint=data.fingerprint, update=update, elapsed=time.monotonic()-started,
            metrics=None, purpose='last_training_state')
        saved = checkpoint.read(output/'best.pt', fingerprint=data.fingerprint, config=config)
        checkpoint.restore(saved, model)
        metrics, _ = evaluate(model, validation, device='cuda', batch_size=config['batch_size'], threshold=config['threshold'])
        if metrics != saved['metrics']:
            raise ValueError('best checkpoint validation replay differs')
        result = dict(stop=stop, updates=update, best_update=saved['update'], metrics=metrics,
                      elapsed_seconds=time.monotonic()-started, test_evaluated=False)
        write_json(output/'training-report.json', result)
        return result
    except BaseException as error:
        write_json(output/'failure.json', dict(error=type(error).__name__, message=str(error), updates=update))
        raise
    finally:
        bar.close()
