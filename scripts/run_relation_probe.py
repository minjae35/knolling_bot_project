"""Connect pose loss to vision and language, then overfit the relation head.

The overfit uses 24 existing four-object training scenes. It stops without a
full training run when those scenes do not reach the task and 2 mm gap.
"""
import json
import os
from pathlib import Path

os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import numpy as np
import torch
from torch.nn import functional as F
from knolling_dataset.validator import geometry, inferred_rows, polygon
from knolling_model.diagnostics import positioned_objects
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.phase1_evaluation import semantic_errors_phase1
from knolling_model.experiments.relation_pose import NeuralRelationPose, relation_loss
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
import run_phase1_pairwise_affine as runner
from run_phase1_pairwise_affine import teacher_order

STEPS = 500
SCENE_COUNT = 24
SUCCESS_REQUIRED = SCENE_COUNT


def grad_total(module):
    total, live = 0., 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        live += 1
        total += float(parameter.grad.detach().abs().sum())
    return dict(parameters_with_grad=live, absolute_sum=total)


def enable(module, flag):
    module.requires_grad_(flag)
    return module


def top_objects(visual, width):
    scores = visual['source']['objectness_logits']
    index = scores.topk(width, dim=-1).indices
    tokens = visual['stages'][-1]['tokens']
    boxes = visual['stages'][-1]['boxes']
    batch = torch.arange(len(index), device=index.device)[:, None]
    token, box = tokens[batch, index], boxes[batch, index]
    xy = torch.stack((box[..., 0]*.6, (1-box[..., 1])*.4), -1)
    size = F.softplus(token[..., :2])+0.02
    angle = .1*token[..., 2]
    color = token[..., 3:6].softmax(-1)
    shape = token[..., 6:9].softmax(-1)
    valid = torch.ones(len(index), width, dtype=torch.bool, device=index.device)
    objects = Objects(xy, size, angle, color, shape, valid, index)
    objects.validate()
    return objects, token.requires_grad, box.requires_grad


def probe(model, record, device):
    sample = None
    for index, candidate in enumerate(model_data.records):
        if candidate['sample_id'] == record['sample_id']:
            sample = model_data[index]
            break
    image = sample['image'][None].to(device)
    tokens = model.tokenizer([record['instruction']['text']], return_tensors='pt', padding=True, truncation=False)
    ids = tokens['input_ids'].to(device)
    mask = tokens['attention_mask'].to(device).bool()
    perception, language = model.perception, model.arrangement.text_encoder
    goal_encoder = model.layout.goal_encoder
    enable(perception, True)
    enable(language, True)
    enable(goal_encoder, True)
    visual = perception(image)
    objects, token_grad, box_grad = top_objects(visual, 4)
    text = language(input_ids=ids, attention_mask=mask).last_hidden_state
    goal = goal_encoder(objects, text, mask)['vector']
    head = NeuralRelationPose(objects.features().shape[-1]).to(device)
    teacher = torch.arange(4, device=device)[None]
    output = head(objects.features(), objects.valid, goal, teacher)
    target = torch.zeros(1, 4, 2, device=device)
    target[0, :, 0] = torch.linspace(.2, .5, 4, device=device)
    loss = relation_loss(output, target, torch.zeros(1, 4, device=device), teacher)['loss']
    loss.backward()
    report = dict(loss=float(loss.detach()), token_requires_grad=bool(token_grad),
                  box_requires_grad=bool(box_grad), text_requires_grad=bool(text.requires_grad),
                  visual=grad_total(perception), language=grad_total(language),
                  goal=grad_total(goal_encoder), relation=grad_total(head))
    enable(perception, False)
    enable(language, False)
    enable(goal_encoder, False)
    model.zero_grad(set_to_none=True)
    connected = all(report[name]['absolute_sum'] > 0 for name in ('visual', 'language', 'goal', 'relation'))
    return connected, report


def choose(records):
    groups = {15: [], 20: [], 30: []}
    for record in records:
        if len(record['source']['objects']) != 4:
            continue
        _, truth = recovery_supervision(record)
        major = truth['size'][:, 0]
        groups[round(record['instruction']['gap_m']*1000)].append((float(major.max()-major.min()), record))
    chosen = []
    for gap in (15, 20, 30):
        ordered = sorted(groups[gap], key=lambda item: item[0])
        picks = ordered[:4]+ordered[-4:]
        if len(picks) != 8:
            raise RuntimeError(f'not enough {gap} mm scenes')
        chosen.extend(picks)
    return chosen


def prepare(model, records, device):
    stored = []
    for record in records:
        objects, text, mask, _ = runner.arguments(model, [record], device)
        with torch.no_grad():
            goal = model.layout.goal_encoder(objects, text, mask)['vector']
            features = objects.features()
        ids, truth = recovery_supervision(record)
        order = teacher_order([record], [ids], objects.valid.shape[1], device)
        stored.append(dict(record=record, ids=ids, features=features[0, :len(ids)].cpu(),
                           goal=goal[0].cpu(), target_xy=truth['target_xy'][:len(ids)].cpu(),
                           target_angle=truth['target_angle'][:len(ids)].cpu(),
                           order=order[0, :len(ids)].cpu()))
    return stored


def stack(rows, key):
    return torch.stack([row[key] for row in rows])


def scene_success(record, ids, xy, angle):
    targets = {oid: dict(x_m=float(xy[i, 0]), y_m=float(xy[i, 1]), theta_rad=float(angle[i]))
               for i, oid in enumerate(ids)}
    positioned = positioned_objects(record, targets)
    semantic = semantic_errors_phase1(positioned, record)
    safety = geometry(positioned, 'target_pose', .010, .010)
    errors = sorted(set(semantic+list(safety.errors)))
    gaps = []
    for line in inferred_rows(positioned):
        for left, right in zip(line, line[1:]):
            opening = polygon(right, right.target_pose).bounds[0]-polygon(left, left.target_pose).bounds[2]
            gaps.append((opening-record['instruction']['gap_m'])*1000)
    moved = [oid for row in record['supervision_v3']['rows'] for oid in row
             if not record['supervision_v3']['copy_required'][oid]]
    predicted = sorted(targets, key=lambda oid: (targets[oid]['x_m'], oid))
    return dict(task=not errors, gaps=gaps, order=predicted == moved, errors=errors)


def evaluate(head, rows, device):
    head.eval()
    results = []
    with torch.no_grad():
        features = stack(rows, 'features').to(device)
        goal = stack(rows, 'goal').to(device)
        valid = torch.ones(len(rows), features.shape[1], dtype=torch.bool, device=device)
        output = head(features, valid, goal)
        xy, angle, order = output['xy'].cpu(), output['angle'].cpu(), output['predicted_order'].cpu()
    for index, row in enumerate(rows):
        item = scene_success(row['record'], row['ids'], xy[index], angle[index])
        item['predicted_order'] = order[index].tolist()
        item['teacher_order'] = row['order'].tolist()
        results.append(item)
    head.train()
    gaps = [gap for item in results for gap in item['gaps']]
    return dict(success=sum(item['task'] for item in results), scenes=len(results),
                within_2mm=sum(abs(gap) <= 2 for gap in gaps), pairs=len(gaps),
                gap_median=float(np.median(np.abs(gaps))) if gaps else None,
                order_exact=sum(item['order'] for item in results),
                results=results)


def overfit(rows, device):
    head = NeuralRelationPose(12).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.)
    features = stack(rows, 'features').to(device)
    goal = stack(rows, 'goal').to(device)
    target_xy = stack(rows, 'target_xy').to(device)
    target_angle = stack(rows, 'target_angle').to(device)
    order = stack(rows, 'order').to(device)
    valid = torch.ones(len(rows), features.shape[1], dtype=torch.bool, device=device)
    history = []
    passed_at = None
    for step in range(1, STEPS+1):
        choice = torch.randint(len(rows), (8,))
        output = head(features[choice], valid[choice], goal[choice], order[choice])
        terms = relation_loss(output, target_xy[choice], target_angle[choice], order[choice])
        optimizer.zero_grad(set_to_none=True)
        terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if step % 50 == 0 or step == STEPS:
            metrics = evaluate(head, rows, device)
            history.append(dict(step=step, loss=float(terms['loss'].detach()),
                                 delta=float(terms['delta'].detach()), order_ce=float(terms['order_ce'].detach()),
                                 success=metrics['success'], within_2mm=metrics['within_2mm'],
                                 gap_median=metrics['gap_median'], order_exact=metrics['order_exact']))
            print('OVERFIT', history[-1], flush=True)
            if metrics['success'] == SUCCESS_REQUIRED and metrics['within_2mm'] == metrics['pairs']:
                passed_at = step
                break
    final = evaluate(head, rows, device)
    return dict(passed=passed_at is not None, passed_at=passed_at, steps=STEPS, history=history,
                final={key: final[key] for key in ('success', 'scenes', 'within_2mm', 'pairs', 'gap_median', 'order_exact')})


def main():
    runner.setup()
    global model_data
    train, _ = runner.datasets()
    model_data = train
    device = 'cuda'
    model = load_integrated(device=device)
    chosen = choose(train.records)
    connected, gradients = probe(model, chosen[0][1], device)
    print('GRADIENT', json.dumps(gradients), 'connected', connected, flush=True)
    if not connected:
        report = dict(gradient=gradients, gradient_passed=False, overfit=None, status='FAIL',
                      continued=False)
    else:
        rows = prepare(model, [record for _, record in chosen], device)
        del model
        torch.cuda.empty_cache()
        result = overfit(rows, device)
        report = dict(gradient=gradients, gradient_passed=True, scenes=len(rows),
                      gaps=sorted({round(row['record']['instruction']['gap_m']*1000) for row in rows}),
                      overfit=result, status='PASS' if result['passed'] else 'FAIL', continued=False)
    Path('/tmp/relation_probe.json').write_text(json.dumps(report)+'\n')
    print('STATUS', report['status'], flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(0)


if __name__ == '__main__':
    main()
