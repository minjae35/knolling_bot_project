"""One joint fine-tune of the saved relation head with the pretrained encoders.

Inference reads RGB and the instruction only. Target poses supervise the loss
and never enter the forward. The existing 128-d checkpoint is not overwritten.
"""
import json
import time
from pathlib import Path

import torch
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.relation_pose import relation_loss
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import SourceSelectionError, match_selected, select_queries
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable

ROOT = Path('artifacts/phase1_relation_joint')
BASELINE = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
ENCODER_LR = 1e-6
ARRANGEMENT_LR = 1e-4/3
MAX_UPDATES = 2000
EVAL_EVERY = 250
BATCH = 8
EXPECTED = dict(train_oracle=566, train_rgb=479, val_oracle=60, val_rgb=56, train_rgb_4=76, val_rgb_4=4)


def install_live_text(model):
    encoder = model.arrangement.text_encoder

    def encode(input_ids, attention_mask):
        with torch.no_grad():
            return encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    model.encode_text = encode
    model.arrangement.encode_text = encode


def module_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()
            if torch.is_tensor(value) and value.is_floating_point()}


def changed(before, module):
    moved, total = 0, 0
    for name, value in module.state_dict().items():
        if name not in before:
            continue
        total += 1
        if not torch.equal(before[name], value.detach().cpu() if before[name].device.type == 'cpu' else value.detach()):
            moved += 1
    return dict(tensors=total, changed=moved)


def grad_report(module):
    total, live = 0., 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        live += 1
        total += float(parameter.grad.detach().norm())
    return dict(parameters_with_grad=live, norm_sum=total)


def isolate(visual, index):
    def cut(value):
        return value[index:index+1] if torch.is_tensor(value) else value
    return dict(source={key: cut(value) for key, value in visual['source'].items()},
                stages=[{key: cut(value) for key, value in stage.items()} for stage in visual['stages']])


def predict_pose(model, head, images, texts):
    """RGB and natural language in, target pose out. No record or target tensor."""
    visual = model.perception(images)
    objects = pack_sources(visual['source'], select_queries(visual['source'], .5))
    tokens = model.tokenizer(texts, return_tensors='pt', padding=True, truncation=False)
    mask = tokens['attention_mask'].to(images.device).bool()
    text = model.arrangement.text_encoder(input_ids=tokens['input_ids'].to(images.device), attention_mask=mask).last_hidden_state
    goal = model.layout.goal_encoder(objects, text, mask)['vector']
    output = head(objects.features(), objects.valid, goal)
    return output, objects


def supervise(model, head, images, records):
    visual = model.perception(images)
    kept, queries, truths, ids = [], [], [], []
    for index, record in enumerate(records):
        obj_ids, truth = recovery_supervision(record)
        truth = {key: value.to(images.device) for key, value in truth.items()}
        try:
            matches = match_selected(model, isolate(visual, index), [truth])
        except (SourceSelectionError, ValueError):
            continue
        selected, gt = matches[0]
        if len(selected) != len(obj_ids) or set(gt.tolist()) != set(range(len(obj_ids))):
            continue
        order = torch.empty(len(obj_ids), dtype=torch.long, device=images.device)
        order[gt] = selected
        kept.append(index)
        queries.append(order)
        truths.append(truth)
        ids.append(obj_ids)
    if not kept:
        return None
    source = {key: value[kept] for key, value in visual['source'].items()}
    objects = pack_sources(source, queries)
    chosen = [records[index] for index in kept]
    tokens = model.tokenizer([record['instruction']['text'] for record in chosen],
                             return_tensors='pt', padding=True, truncation=False)
    mask = tokens['attention_mask'].to(images.device).bool()
    text = model.arrangement.text_encoder(input_ids=tokens['input_ids'].to(images.device),
                                          attention_mask=mask).last_hidden_state
    goal = model.layout.goal_encoder(objects, text, mask)['vector']
    width = objects.valid.shape[1]
    teacher = runner.teacher_order(chosen, ids, width, images.device)
    output = head(objects.features(), objects.valid, goal, teacher)
    target_xy = objects.xy.new_zeros(len(chosen), width, 2)
    target_angle = objects.angle.new_zeros(len(chosen), width)
    for index, truth in enumerate(truths):
        count = len(truth['target_xy'])
        target_xy[index, :count] = truth['target_xy']
        target_angle[index, :count] = truth['target_angle']
    terms = relation_loss(output, target_xy, target_angle, teacher)
    return dict(terms=terms, kept=len(chosen), text=text, objects=objects)


def headlines(measured):
    return dict(train_oracle=measured['train']['oracle']['task'], train_rgb=measured['train']['rgb']['task'],
                val_oracle=measured['val']['oracle']['task'], val_rgb=measured['val']['rgb']['task'],
                train_rgb_4=measured['train']['rgb']['count']['4']['success'],
                val_rgb_4=measured['val']['rgb']['count']['4']['success'])


def anchor_mm(head, stored):
    head.eval()
    item = stable.take(stored, list(range(len(stored['valid']))), stable.DEVICE)
    with torch.no_grad():
        output = head(item['features'], item['valid'], item['goal'])
    rows = torch.arange(len(item['order']), device=stable.DEVICE)
    first = item['order'][:, 0].clamp_min(0)
    error = (output['xy'][rows, first]-item['target_xy'][rows, first]).norm(dim=-1)*1000
    return float(error.median())


def caches_of(model, train, val):
    print('CACHE start', flush=True)
    return dict(train_oracle=stable.cache_oracle(model, train), val_oracle=stable.cache_oracle(model, val),
                val_rgb=stable.cache_rgb(model, val), train_rgb=stable.cache_rgb(model, train))


def rank_of(measured):
    rgb = measured['val']['rgb']
    median = rgb['gap']['median'] if rgb['gap']['median'] is not None else 1e9
    return (rgb['task'], rgb['count']['4']['success'], measured['train']['rgb']['count']['4']['success'], -median)


def step_check(model, head, train):
    before = dict(perception=module_state(model.perception), language=module_state(model.arrangement.text_encoder),
                  goal=module_state(model.layout.goal_encoder), relation=module_state(head))
    images, records = [], []
    for index, record in enumerate(train.records[:32]):
        images.append(train[index]['image'])
        records.append(record)
        if len(images) == 2:
            batch = torch.stack(images).to(stable.DEVICE)
            with torch.no_grad():
                pose, objects = predict_pose(model, head, batch, [record['instruction']['text'] for record in records])
            inference_ok = (torch.isfinite(pose['xy']).all() and torch.isfinite(pose['angle']).all()
                            and pose['rollout'] == 'free_rollout' and objects.valid.shape[0] == 2)
            prepared = supervise(model, head, batch, records)
            if inference_ok and prepared is not None:
                break
            images, records = [], []
    else:
        return dict(passed=False, reason='no phase-1 scene produced a pose and a matched training loss')
    encoder = [p for p in list(model.perception.parameters())+list(model.arrangement.text_encoder.parameters()) if p.requires_grad]
    arrangement = [p for p in list(model.layout.goal_encoder.parameters())+list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=encoder, lr=ENCODER_LR), dict(params=arrangement, lr=ARRANGEMENT_LR)],
                                  weight_decay=0., foreach=False)
    optimizer.zero_grad(set_to_none=True)
    prepared['terms']['loss'].backward()
    grads = dict(visual=grad_report(model.perception), language=grad_report(model.arrangement.text_encoder),
                 goal=grad_report(model.layout.goal_encoder), relation=grad_report(head))
    torch.nn.utils.clip_grad_norm_(encoder+arrangement, 1.)
    optimizer.step()
    movement = dict(perception=changed(before['perception'], model.perception),
                    language=changed(before['language'], model.arrangement.text_encoder),
                    goal=changed(before['goal'], model.layout.goal_encoder),
                    relation=changed(before['relation'], head))
    for module, saved in ((model.perception, before['perception']), (model.arrangement.text_encoder, before['language']),
                          (model.layout.goal_encoder, before['goal']), (head, before['relation'])):
        current = module.state_dict()
        for name, value in saved.items():
            current[name].copy_(value.to(current[name].device))
    passed = inference_ok and all(grads[name]['norm_sum'] > 0 for name in grads) and all(item['changed'] > 0 for item in movement.values())
    return dict(passed=passed, kept=prepared['kept'], grads=grads, movement=movement,
                text_in_graph=bool(prepared['text'].requires_grad), loss=float(prepared['terms']['loss'].detach()))


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('joint output already exists')
    stamp = (BASELINE.stat().st_mtime_ns, BASELINE.stat().st_size)
    runner.setup()
    train, val = runner.datasets()
    saved = torch.load(BASELINE, map_location='cpu', weights_only=False)
    model = load_integrated(device=stable.DEVICE)
    install_live_text(model)
    model.eval()
    for module in (model.perception, model.arrangement.text_encoder, model.layout.goal_encoder):
        module.requires_grad_(True)
        module.eval()
    head = cap.load_head(128, saved['model'])
    head.eval()
    check = step_check(model, head, train)
    print('STEP', json.dumps(check), flush=True)
    ROOT.mkdir(parents=True)
    stable.write(ROOT/'step.json', check)
    if not check['passed']:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='weight_update', step=check))
        print('STOP weight update did not reach every module', flush=True)
        return
    if (BASELINE.stat().st_mtime_ns, BASELINE.stat().st_size) != stamp:
        raise RuntimeError('baseline checkpoint changed during the step check')
    built = caches_of(model, train, val)
    reference = cap.evaluate(head, built, dict(train=train, val=val))
    got = headlines(reference)
    print('REPRODUCE', got, flush=True)
    if got != EXPECTED:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reproduction', got=got, expected=EXPECTED))
        print('STOP reproduction mismatch', flush=True)
        return
    baseline_ids = {split: {mode: cap.ids_of(reference[split][mode]) for mode in ('oracle', 'rgb')}
                    for split in ('train', 'val')}
    pairwise = cap.pairwise_ids()
    base_anchor = anchor_mm(head, built['val_oracle'])
    encoder = [p for p in list(model.perception.parameters())+list(model.arrangement.text_encoder.parameters()) if p.requires_grad]
    arrangement = [p for p in list(model.layout.goal_encoder.parameters())+list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=encoder, lr=ENCODER_LR), dict(params=arrangement, lr=ARRANGEMENT_LR)],
                                  weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    order, cursor, step, losses, skips = [], 0, 0, [], 0
    best = None
    last = None
    handle = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()
    reason = 'max_updates'
    while step < MAX_UPDATES:
        if cursor >= len(order):
            order = torch.randperm(len(train), generator=generator).tolist()
            cursor = 0
        choice = order[cursor:cursor+BATCH]
        cursor += len(choice)
        records = [train.records[index] for index in choice]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        head.train()
        prepared = supervise(model, head, images, records)
        if prepared is None:
            skips += 1
            if skips >= 20:
                reason = 'no_matched_scenes'
                break
            continue
        skips = 0
        step += 1
        optimizer.zero_grad(set_to_none=True)
        prepared['terms']['loss'].backward()
        raw = torch.nn.utils.clip_grad_norm_(encoder+arrangement, 1.)
        if not torch.isfinite(raw):
            reason = 'nonfinite_grad'
            break
        optimizer.step()
        losses.append(float(prepared['terms']['loss'].detach()))
        if step % EVAL_EVERY != 0 and step != MAX_UPDATES:
            continue
        head.eval()
        built = caches_of(model, train, val)
        measured = cap.evaluate(head, built, dict(train=train, val=val))
        point = dict(update=step, loss=sum(losses)/len(losses), grad_clip=float(raw),
                     anchor_mm=anchor_mm(head, built['val_oracle']),
                     seconds=time.perf_counter()-started, headlines=headlines(measured))
        del built
        losses = []
        handle.write(json.dumps(dict(point, metrics=cap.snapshot(measured, baseline_ids, pairwise)))+'\n')
        handle.flush()
        print('EVAL', dict(update=step, loss=point['loss'], grad=round(point['grad_clip'], 4),
                           anchor_mm=round(point['anchor_mm'], 3), **point['headlines']), flush=True)
        last = dict(measured=measured, anchor_mm=point['anchor_mm'])
        score = rank_of(measured)
        if best is None or score > best['rank']:
            best = dict(measured=measured, rank=score, update=step, anchor_mm=point['anchor_mm'])
            torch.save(dict(update=step, relation={key: value.detach().cpu() for key, value in head.state_dict().items()},
                            perception={key: value.detach().cpu() for key, value in model.perception.state_dict().items()},
                            language={key: value.detach().cpu() for key, value in model.arrangement.text_encoder.state_dict().items()},
                            goal={key: value.detach().cpu() for key, value in model.layout.goal_encoder.state_dict().items()},
                            selection='validation_rgb_then_4_then_train4'), ROOT/'checkpoint-best.pt')
    handle.close()
    if best is None:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='training', stop=reason, updates=step))
        print('STOP no evaluation', reason, flush=True)
        return
    final = best['measured']
    numbers = headlines(final)
    failures = final['val']['rgb']['failures']
    regression = dict(validation_rgb=numbers['val_rgb'] < EXPECTED['val_rgb'],
                      validation_rgb_4=numbers['val_rgb_4'] < EXPECTED['val_rgb_4'],
                      train_rgb_4=numbers['train_rgb_4'] < EXPECTED['train_rgb_4'],
                      clearance_or_collision=failures.get('clearance', 0)+failures.get('collision', 0))
    improved = any(numbers[key] > EXPECTED[key] for key in ('val_rgb', 'val_rgb_4', 'train_rgb_4'))
    status = 'PASS' if improved and not any(regression[key] for key in ('validation_rgb', 'validation_rgb_4', 'train_rgb_4')) else 'FAIL'
    report = dict(status=status, stop=reason, updates=step, best_update=best['update'],
                  baseline=EXPECTED, best=cap.snapshot(final, baseline_ids, pairwise),
                  final=None if last is None else cap.snapshot(last['measured'], baseline_ids, pairwise),
                  baseline_anchor_mm=base_anchor, best_anchor_mm=best.get('anchor_mm'), regression=regression,
                  baseline_checkpoint_unchanged=(BASELINE.stat().st_mtime_ns, BASELINE.stat().st_size) == stamp,
                  seconds=time.perf_counter()-started, encoder_lr=ENCODER_LR, arrangement_lr=ARRANGEMENT_LR,
                  joint_executed=True, test_used=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', dict(status=status, update=best['update'], **numbers), flush=True)


if __name__ == '__main__':
    main()
