"""Fine-tune only the visual encoder from the modular best.

The relation head, goal encoder, and language encoder stay at the modular
checkpoint. Inference still reads RGB and text only. The modular and joint
checkpoints are not overwritten.
"""
import json
import time
from pathlib import Path

import torch
from knolling_model.experiments.integration import load_integrated
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from run_relation_joint import changed, install_live_text, module_state, supervise

ROOT = Path('artifacts/phase1_relation_visual')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
VISUAL_LR = 1e-6
MAX_UPDATES = 500
EVAL_EVERY = 100
BATCH = 8
RGB_BASE = 56
ORACLE_BASE = 60
RGB4_BASE = 4


def stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def summary(oracle, rgb):
    return dict(oracle=oracle['task'], rgb=rgb['task'],
                oracle4=oracle['count']['4']['success'], rgb4=rgb['count']['4']['success'],
                rgb_count_errors=rgb['failures'].get('count', 0),
                oracle_order=oracle['order'], rgb_order=rgb['order'],
                oracle_gap=oracle['gap'], rgb_gap=rgb['gap'],
                oracle_failures=oracle['failures'], rgb_failures=rgb['failures'])


def evaluate(model, head, val):
    head.eval()
    oracle = stable.official_metrics(head, stable.cache_oracle(model, val), val)
    rgb = stable.official_metrics(head, stable.cache_rgb(model, val), val)
    return summary(oracle, rgb)


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('visual-only output already exists')
    held = stamp(MODULAR)
    runner.setup()
    train, val = runner.datasets()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    model = load_integrated(device=stable.DEVICE)
    install_live_text(model)
    model.eval()
    head = cap.load_head(128, saved['model'])
    head.eval()
    for module in (model.arrangement.text_encoder, model.layout.goal_encoder, head):
        module.requires_grad_(False)
        module.eval()
    model.perception.requires_grad_(True)
    model.perception.eval()
    frozen = dict(language=module_state(model.arrangement.text_encoder),
                  goal=module_state(model.layout.goal_encoder), relation=module_state(head))
    visual_before = module_state(model.perception)
    baseline = evaluate(model, head, val)
    print('REPRODUCE', json.dumps(dict(oracle=baseline['oracle'], rgb=baseline['rgb'], rgb4=baseline['rgb4'],
                                      count_errors=baseline['rgb_count_errors'])), flush=True)
    if (baseline['oracle'], baseline['rgb'], baseline['rgb4']) != (ORACLE_BASE, RGB_BASE, RGB4_BASE):
        ROOT.mkdir(parents=True)
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reproduction', got=baseline))
        print('STOP reproduction mismatch', flush=True)
        return
    visual = [parameter for parameter in model.perception.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(visual, lr=VISUAL_LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    order, cursor, step, skips, losses = [], 0, 0, 0, []
    best = None
    reason = 'max_updates'
    started = time.perf_counter()
    ROOT.mkdir(parents=True)
    handle = (ROOT/'history.jsonl').open('w')
    while step < MAX_UPDATES:
        if cursor >= len(order):
            order = torch.randperm(len(train), generator=generator).tolist()
            cursor = 0
        choice = order[cursor:cursor+BATCH]
        cursor += len(choice)
        records = [train.records[index] for index in choice]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        prepared = supervise(model, head, images, records)
        if prepared is None:
            skips += 1
            if skips >= 20:
                reason = 'no_matched_scenes'
                break
            continue
        skips = 0
        optimizer.zero_grad(set_to_none=True)
        prepared['terms']['loss'].backward()
        if step == 0:
            visual_grad = sum(float(parameter.grad.detach().norm()) for parameter in visual if parameter.grad is not None)
            frozen_grad = any(parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0
                              for module in (model.arrangement.text_encoder, model.layout.goal_encoder, head)
                              for parameter in module.parameters())
            features_connected = bool(prepared['objects'].features().requires_grad) and visual_grad > 0 and not frozen_grad
            print('PATH', json.dumps(dict(visual_grad=visual_grad, frozen_grad=frozen_grad,
                                          features=features_connected)), flush=True)
            if not features_connected:
                reason = 'visual_path_blocked'
                break
        raw = torch.nn.utils.clip_grad_norm_(visual, 1.)
        if not torch.isfinite(raw):
            reason = 'nonfinite_grad'
            break
        optimizer.step()
        step += 1
        losses.append(float(prepared['terms']['loss'].detach()))
        if step % EVAL_EVERY != 0 and step != MAX_UPDATES:
            continue
        point = evaluate(model, head, val)
        point.update(update=step, loss=sum(losses)/len(losses), grad_clip=float(raw),
                     seconds=time.perf_counter()-started)
        losses = []
        handle.write(json.dumps(point)+'\n')
        handle.flush()
        print('EVAL', json.dumps(dict(update=step, oracle=point['oracle'], rgb=point['rgb'],
                                     rgb4=point['rgb4'], count_errors=point['rgb_count_errors'])), flush=True)
        if point['oracle'] < ORACLE_BASE or point['rgb'] < RGB_BASE:
            reason = 'regression'
            break
        if point['rgb'] > RGB_BASE and (best is None or (point['rgb'], point['rgb4'], -point['rgb_count_errors']) > best['rank']):
            best = dict(rank=(point['rgb'], point['rgb4'], -point['rgb_count_errors']), update=step, metrics=point)
            torch.save(dict(update=step, perception={key: value.detach().cpu() for key, value in model.perception.state_dict().items()},
                            source='modular_best', visual_lr=VISUAL_LR,
                            frozen=('relation_head', 'goal_encoder', 'language_encoder')),
                       ROOT/'checkpoint-best.pt')
    handle.close()
    movement = dict(perception=changed(visual_before, model.perception),
                    language=changed(frozen['language'], model.arrangement.text_encoder),
                    goal=changed(frozen['goal'], model.layout.goal_encoder),
                    relation=changed(frozen['relation'], head))
    status = 'STOP' if reason == 'visual_path_blocked' else 'PASS' if best is not None else 'FAIL'
    report = dict(status=status, stop=reason, updates=step, best_update=None if best is None else best['update'],
                  baseline=baseline, best=None if best is None else best['metrics'],
                  frozen_movement=movement, visual_lr=VISUAL_LR,
                  modular_checkpoint_unchanged=stamp(MODULAR) == held, test_used=False,
                  seconds=time.perf_counter()-started)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=status, stop=reason, update=report['best_update'],
                                 movement=movement)), flush=True)


if __name__ == '__main__':
    main()
