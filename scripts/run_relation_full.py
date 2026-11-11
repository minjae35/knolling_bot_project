"""One full relation-head training run. No 24-scene retune and no joint update.

Learning rate stays at 0.0003 after warmup. The saved checkpoint is the best
validation free rollout, and training stops when that score stops rising.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.relation_pose import NeuralRelationPose, relation_loss
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable

SPEC = json.loads(Path('configs/phase1_relation_full.json').read_text())
ROOT = Path(SPEC['output'])


def learning_rate(step):
    warm = SPEC['warmup_steps']
    if step < warm:
        return SPEC['lr'] * step / warm
    return SPEC['lr']


def rank_of(metrics):
    rgb, oracle = metrics['rgb'], metrics['oracle']
    return (rgb['task'] + oracle['task'], rgb['task'],
            rgb['count']['4']['success'] + oracle['count']['4']['success'],
            rgb['count']['3']['success'] + oracle['count']['3']['success'],
            -(rgb['gap']['median'] if rgb['gap']['median'] is not None else 1e9))


def brief(name, metrics):
    return dict(split=name, task=metrics['task'], total=metrics['total'],
                count=metrics['count'], order=metrics['order'], selection_order=metrics['selection_order'],
                gap=metrics['gap'], failures=metrics['failures'])


def train_once(caches, datasets):
    train_data = datasets['train']
    epochs, batch = SPEC['max_epochs'], SPEC['batch_size']
    generator = torch.Generator().manual_seed(SPEC['seed'])
    head = NeuralRelationPose(12).to(stable.DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=SPEC['lr'], weight_decay=SPEC['weight_decay'], foreach=False)
    best, waited, history = None, 0, []
    step = 0
    for epoch in range(1, epochs+1):
        permutation = torch.randperm(len(train_data), generator=generator).tolist()
        losses = []
        head.train()
        for start in range(0, len(permutation), batch):
            step += 1
            rate = learning_rate(step)
            for group in optimizer.param_groups:
                group['lr'] = rate
            choice = permutation[start:start+batch]
            data = stable.take(caches['train_oracle'], choice, stable.DEVICE)
            output = head(data['features'], data['valid'], data['goal'], data['order'])
            terms = relation_loss(output, data['target_xy'], data['target_angle'], data['order'])
            optimizer.zero_grad(set_to_none=True)
            terms['loss'].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), SPEC['grad_clip'])
            optimizer.step()
            losses.append(float(terms['loss'].detach()))
        validation = dict(oracle=stable.official_metrics(head, caches['val_oracle'], datasets['val']),
                          rgb=stable.official_metrics(head, caches['val_rgb'], datasets['val']))
        point = dict(epoch=epoch, step=step, lr=rate, train_loss=float(np.mean(losses)),
                     validation={mode: brief(mode, validation[mode]) for mode in validation})
        score = rank_of(validation)
        improved = best is None or score > best['score']
        if epoch % SPEC['train_eval_every'] == 0:
            training = dict(oracle=stable.official_metrics(head, caches['train_oracle'], train_data),
                            rgb=stable.official_metrics(head, caches['train_rgb'], train_data))
            point['train'] = {mode: brief(mode, training[mode]) for mode in training}
        history.append(point)
        with (ROOT/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(point)+'\n')
        val = point['validation']
        print('EPOCH', dict(epoch=epoch, step=step, lr=rate, loss=point['train_loss'],
                            oracle=val['oracle']['task'], rgb=val['rgb']['task'],
                            oracle4=val['oracle']['count']['4']['success'],
                            rgb4=val['rgb']['count']['4']['success']), flush=True)
        if improved:
            best = dict(score=score, epoch=epoch, step=step, validation=validation,
                        state={key: value.detach().cpu() for key, value in head.state_dict().items()})
            waited = 0
            torch.save(dict(model=best['state'], epoch=epoch, step=step, lr=SPEC['lr'],
                            scheduler=SPEC['scheduler'], selection='validation_free_rollout',
                            validation={mode: brief(mode, validation[mode]) for mode in validation}),
                       ROOT/'checkpoint-best.pt')
        else:
            waited += 1
            if waited >= SPEC['patience_epochs']:
                print('EARLY STOP', dict(epoch=epoch, waited=waited), flush=True)
                break
    head.load_state_dict(best['state'])
    best['train'] = {mode: brief(mode, stable.official_metrics(head, caches[f'train_{mode}'], train_data))
                     for mode in ('oracle', 'rgb')}
    return best, history


def joint_case(validation):
    oracle, rgb = validation['oracle'], validation['rgb']
    return dict(
        beats_rgb_23=rgb['task'] > SPEC['baseline_rgb_task'],
        rgb_4_above_baseline=rgb['count']['4']['success'] > SPEC['baseline_rgb_4'],
        oracle_minus_rgb=oracle['task']-rgb['task'],
        four_oracle=oracle['count']['4']['success'],
        four_rgb=rgb['count']['4']['success'])


def main():
    stable.smoke()
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    if any(ROOT.iterdir()):
        raise RuntimeError('full-training output already exists')
    train_data, val_data = runner.datasets()
    model = load_integrated(device=stable.DEVICE)
    print('CACHE start', flush=True)
    caches = dict(train_oracle=stable.cache_oracle(model, train_data),
                  val_oracle=stable.cache_oracle(model, val_data),
                  val_rgb=stable.cache_rgb(model, val_data),
                  train_rgb=stable.cache_rgb(model, train_data))
    del model
    torch.cuda.empty_cache()
    best, history = train_once(caches, dict(train=train_data, val=val_data))
    validation = {mode: brief(mode, best['validation'][mode]) for mode in best['validation']}
    report = dict(status='compared', scheduler=SPEC['scheduler'], lr=SPEC['lr'],
                  selected=dict(epoch=best['epoch'], step=best['step']),
                  epochs_ran=history[-1]['epoch'], updates=history[-1]['step'],
                  validation=validation, train=best['train'],
                  baseline_rgb='23/94', joint_case=joint_case(best['validation']),
                  joint_training=False, test_used=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(epoch=best['epoch'], step=best['step'],
                                  rgb=validation['rgb']['task'], oracle=validation['oracle']['task'],
                                  rgb4=validation['rgb']['count']['4']['success'])), flush=True)


if __name__ == '__main__':
    main()
