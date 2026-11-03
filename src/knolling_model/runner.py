"""S0 only. Explicit training entry point; preflight never updates weights."""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

from knolling_dataset.builder import file_hash
from .data import S0Dataset, Collator, FamilySampler
from .evaluation import evaluate
from .losses import S0Criterion, POSITION_SCALE, SIZE_SCALE, ANGLE_SCALE, STAGE_WEIGHTS
from .model import load_pretrained


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config['schema'] != 's0_run_v1':
        raise ValueError('unsupported run schema')
    for key in ('micro_batch', 'effective_batch', 'max_updates', 'checkpoint_every', 'cpu_threads'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['effective_batch'] != 16 or config['micro_batch'] > 16 or config['max_updates'] > 2000:
        raise ValueError('S0 requires effective batch 16, micro batch <=16 and <=2000 updates')
    for key in ('backbone_lr', 'new_modules_lr', 'weight_decay', 'gradient_clip'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'invalid {key}')
    return config


def seed_everything(seed, cpu_threads=2):
    # Must precede any CUDA context creation. Fail rather than silently use a
    # nondeterministic kernel; identical results across devices are not promised.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_optimizer(model, config):
    no_decay = set()
    for name, module in model.named_modules():
        for leaf, _ in module.named_parameters(recurse=False):
            if leaf.endswith('bias') or isinstance(module, nn.LayerNorm):
                no_decay.add(f'{name}.{leaf}' if name else leaf)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = config['backbone_lr'] if name.startswith('backbone.body.') else config['new_modules_lr']
        decay = 0. if name in no_decay else config['weight_decay']
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{'params': params, 'lr': lr, 'weight_decay': decay}
                             for (lr, decay), params in sorted(groups.items())], foreach=False)


def update_groups(indices, effective_batch):
    return [indices[i:i+effective_batch] for i in range(0, len(indices), effective_batch)]


def batches(dataset, collator, indices, micro_batch):
    for start in range(0, len(indices), micro_batch):
        yield collator([dataset[i] for i in indices[start:start+micro_batch]])


def backward_group(model, criterion, dataset, collator, indices, micro_batch, device):
    """Accumulate the exact scene mean, including uneven tail micro batches."""
    if not indices:
        raise ValueError('empty update group')
    totals = {}
    for batch in batches(dataset, collator, indices, micro_batch):
        batch = batch.to(device)
        terms, _ = criterion(model(**batch.inputs), batch.supervision)
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError('nonfinite loss; update aborted')
        weight = len(batch.supervision)/len(indices)
        (terms['loss'] * weight).backward()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * weight
    return totals


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def provenance(config):
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {'config': config, 'manifest_sha256': file_hash(Path(config['dataset'])/'manifest.jsonl'),
            'selection_sha256': file_hash(config['selection']),
            'selection': json.loads(Path(config['selection']).read_text()),
            'pretrained': json.loads(Path(config['pretrained']).read_text()),
            'teacher_policy': 'balanced_rows_v1.3', 'language_policy': 'english_only_v1',
            'source_sha256': source.hexdigest(), 'git_head': git('rev-parse', 'HEAD'),
            'git_dirty': bool(git('status', '--porcelain')),
            'loss': {'position_scale_m': POSITION_SCALE, 'size_scale_m': SIZE_SCALE,
                     'angle_scale_rad': ANGLE_SCALE, 'stage_weights': list(STAGE_WEIGHTS),
                     'negative_objectness_weight': .1, 'class_weight': .1,
                     'detection_weight': 1., 'target_weight': 1.},
            'tokens': {'max_length': 512, 'truncation': False, 'padding': 'batch_longest'},
            'precision': 'float32', 'dropout': 0., 'scheduler': None,
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'platform': platform.platform(), 'cuda_runtime': torch.version.cuda}}


def save_checkpoint(path, model, optimizer, cursor, best_score, evidence):
    # Frozen text weights are identified by the pinned revision and reloaded.
    # All trainable parameters AND visual FrozenBN buffers are retained.
    payload = {'format': 's0_checkpoint_v1', 'model': {k: v for k,v in model.state_dict().items()
                                                      if not k.startswith('text_encoder.')},
               'optimizer': optimizer.state_dict(), 'scheduler': None,
               'cursor': cursor, 'best_score': best_score, 'rng': rng_state(), 'provenance': evidence}
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, evidence):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format'] != 's0_checkpoint_v1':
        raise ValueError('unsupported checkpoint')
    for key in ('config', 'manifest_sha256', 'selection_sha256', 'pretrained', 'source_sha256',
                'teacher_policy', 'language_policy', 'loss', 'tokens', 'precision', 'dropout'):
        if saved['provenance'][key] != evidence[key]:
            raise ValueError(f'resume mismatch: {key}')
    mismatch = model.load_state_dict(saved['model'], strict=False)
    if mismatch.unexpected_keys or any(not key.startswith('text_encoder.') for key in mismatch.missing_keys):
        raise ValueError(f'checkpoint model keys mismatch: {mismatch}')
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return saved['cursor'], saved['best_score']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def protect_dataset_output(path, config):
    if Path(path).resolve().is_relative_to(Path(config['dataset']).resolve()):
        raise ValueError('output must not be inside the immutable input dataset')


def append_event(path, event):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False)+'\n')


def prepare(config, device, offline):
    seed_everything(config['seed'], config['cpu_threads'])
    dataset = S0Dataset(config['dataset'], config['selection'])
    model, tokenizer = load_pretrained(config['pretrained'], local_files_only=offline)
    model.to(device)
    return dataset, model, Collator(tokenizer)


def preflight(config, report, device='cpu', offline=True):
    protect_dataset_output(report, config)
    dataset, model, collator = prepare(config, device, offline)
    lengths = [int(collator([dataset[i]]).inputs['attention_mask'].sum()) for i in range(len(dataset))]
    before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    start = time.monotonic()
    batch = collator([dataset[i] for i in range(config['micro_batch'])]).to(device)
    terms, _ = S0Criterion()(model(**batch.inputs), batch.supervision)
    terms['loss'].backward()
    finite = bool(torch.isfinite(terms['loss'])) and all(torch.isfinite(p.grad).all().item()
                       for p in model.parameters() if p.grad is not None)
    unchanged = all(torch.equal(p, before[n]) for n,p in model.named_parameters() if n in before)
    frozen = all(p.grad is None and not p.requires_grad for p in model.text_encoder.parameters())
    if not finite or not unchanged or not frozen:
        raise RuntimeError('preflight gradient/immutability check failed')
    result = {'valid': True, 'device': str(device), 'samples': len(dataset),
              'scene_families': len(FamilySampler(dataset.records, config['seed'])),
              'tokens': {'min': min(lengths), 'max': max(lengths)},
              'micro_batch': config['micro_batch'], 'optimizer_steps': 0,
              'weights_unchanged': unchanged, 'finite_gradients': finite,
              'frozen_text_encoder': frozen, 'forward_backward_seconds': time.monotonic()-start,
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'initial_loss': float(terms['loss'].detach()), 'provenance': provenance(config)}
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    write_json(report, result)
    return result


def train(config, run_dir, device='cuda', offline=True, resume=None):
    run_dir = Path(run_dir)
    protect_dataset_output(run_dir, config)
    # Resumption always writes into a NEW output directory, preserving old logs.
    # Its parent checkpoint + exact cursor establish the continuation lineage.
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset, model, collator = prepare(config, device, offline)
    optimizer, criterion = make_optimizer(model, config), S0Criterion()
    evidence = provenance(config)
    evidence['device'] = str(device)
    evidence['device_name'] = torch.cuda.get_device_name(device) if str(device).startswith('cuda') else platform.machine()
    cursor, best_score = {'epoch': 0, 'group': 0, 'updates': 0}, None
    if resume:
        cursor, best_score = load_checkpoint(resume, model, optimizer, evidence)
        evidence['resume'] = {'checkpoint': str(Path(resume).resolve()), 'sha256': file_hash(resume)}
    write_json(run_dir/'provenance.json', evidence)
    (run_dir/'resolved-requirements.txt').write_text(subprocess.check_output(
        [os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    sampler = FamilySampler(dataset.records, config['seed'])
    last_evaluated = None
    passed = False
    while cursor['updates'] < config['max_updates'] and not passed:
        epoch = cursor['epoch']
        sampler.set_epoch(epoch)
        groups = update_groups(list(sampler), config['effective_batch'])
        for group_index in range(cursor['group'], len(groups)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            terms = backward_group(model, criterion, dataset, collator, groups[group_index],
                                   config['micro_batch'], device)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            cursor = {'epoch': epoch, 'group': group_index+1, 'updates': cursor['updates']+1}
            if cursor['group'] == len(groups):
                cursor['epoch'], cursor['group'] = epoch+1, 0
            append_event(run_dir/'metrics.jsonl', {'event': 'update', **cursor, 'losses': terms,
                         'gradient_norm_before_clip': float(grad_norm), 'scenes': len(groups[group_index])})
            if cursor['updates'] % config['checkpoint_every'] == 0:
                save_checkpoint(run_dir/'last.pt', model, optimizer, cursor, best_score, evidence)
            if cursor['updates'] >= config['max_updates']:
                break
        # Evaluate every family epoch and at a partial-epoch budget stop. All 32
        # instructions are tested, regardless of that epoch's family choices.
        metrics, rows = evaluate(model, batches(dataset, collator, list(range(len(dataset))),
                                                config['micro_batch']), device)
        append_event(run_dir/'metrics.jsonl', {'event': 'evaluation', **cursor, 'metrics': metrics})
        write_json(run_dir/'evaluation-latest.json', {'cursor': cursor, 'metrics': metrics, 'samples': rows})
        # Best checkpoint: lexicographic count, identity coverage, then normalized
        # errors. This is overfit selection, never held-out model selection.
        error_score = 0.
        for key, scale in [('source_position_mm',1), ('target_position_mm',1), ('source_size_mm',1),
                           ('source_angle_deg',2), ('target_angle_deg',2)]:
            error_score += (metrics[key]['p95'] if metrics[key]['p95'] is not None else 1000)/scale
        score = (1-metrics['count_accuracy'], 1-metrics['correspondence_coverage'], error_score)
        if best_score is None or score < best_score:
            best_score = score
            save_checkpoint(run_dir/'best.pt', model, optimizer, cursor, best_score, evidence)
        passed, last_evaluated = metrics['s0_gate'], metrics
    if last_evaluated is None:
        last_evaluated, rows = evaluate(model, batches(dataset, collator, list(range(len(dataset))),
                                                       config['micro_batch']), device)
        passed = last_evaluated['s0_gate']
        write_json(run_dir/'evaluation-latest.json', {'cursor': cursor, 'metrics': last_evaluated, 'samples': rows})
    save_checkpoint(run_dir/'last.pt', model, optimizer, cursor, best_score, evidence)
    result = {'cursor': cursor, 's0_gate': passed, 'last_evaluation': last_evaluated,
              'stop_reason': 'memorization_gate' if passed else 'update_budget',
              'generalization_claim': False}
    write_json(run_dir/'result.json', result)
    return result
