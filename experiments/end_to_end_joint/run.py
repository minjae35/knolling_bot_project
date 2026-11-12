"""One joint pilot from RGB and language through the adopted row layout.

The relation anchor and neighbor chain stay frozen. Their checkpoint files are
only read. GT pose, row, and slot supervise the row losses and never enter
predict_poses. The old 260-scene family holdout is not the adoption set.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.end_to_end import (
    consistency_loss, encode_instruction, predict_poses, supervise_arrangement)
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement
from knolling_model.recovery.contracts import Objects, oracle_objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_row_layout import attach_rows

ROOT = Path('artifacts/experiments/end_to_end_joint')
ROW_BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
RELATION_BEST = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DX_BEST = Path('artifacts/row_layout_dx_fixed_origin/checkpoint-best.pt')
HOLDOUT = Path('artifacts/dev_holdout/multi_row_families.json')
DATA = Path('artifacts/generalization_20000/dataset')
ENCODER_LR = 1e-6
GOAL_LR = 1e-5
HIDDEN_LR = 1e-5
ROW_LR = 1e-4
BATCH = 2
EVAL_EVERY = 200
MAX_UPDATES = 1000
MIN_UPDATES = 400
PATIENCE = 3
CONFIG = dict(pilot='end_to_end_joint', encoder_lr=ENCODER_LR, goal_lr=GOAL_LR, hidden_lr=HIDDEN_LR,
              row_lr=ROW_LR, batch=BATCH, eval_every=EVAL_EVERY, max_updates=MAX_UPDATES,
              min_updates=MIN_UPDATES, patience_evals=PATIENCE, seed=1203,
              consistency=0.1, selection='pilot_family_dev', validation_used_for_selection=False)


def clone_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def restore_state(module, saved):
    current = module.state_dict()
    for key, value in saved.items():
        current[key].copy_(value.to(current[key].device))


def moved(before, module):
    changed = 0
    for key, value in module.state_dict().items():
        if torch.is_tensor(value) and not torch.equal(value, before[key].to(value.device)):
            changed += 1
    return changed


def grad_norm(parameters):
    total, live = 0., 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        live += 1
        total += float(parameter.grad.detach().norm())
    return dict(parameters_with_grad=live, norm_sum=total)


def frozen_names(relation):
    names = []
    for name, _ in relation.named_parameters():
        if name == 'begin' or name.startswith(('anchor.', 'relation.', 'query.', 'angle.')):
            names.append(name)
    return names


def prepare(relation, row_model, system):
    for parameter in relation.parameters():
        parameter.requires_grad_(False)
    for name, parameter in relation.named_parameters():
        if name.startswith('hidden.'):
            parameter.requires_grad_(True)
    row_model.requires_grad_(True)
    system.perception.requires_grad_(True)
    system.arrangement.text_encoder.requires_grad_(True)
    system.layout.goal_encoder.requires_grad_(True)
    system.perception.eval()
    system.arrangement.text_encoder.eval()
    system.layout.goal_encoder.eval()


def oracle_pack(system, records):
    truths = [{key: value.to(stable.DEVICE) for key, value in recovery_supervision(record)[1].items()}
              for record in records]
    objects = oracle_objects(truths)
    text, mask = encode_instruction(system, [record['instruction']['text'] for record in records], stable.DEVICE)
    return objects, text, mask


def capture(system, relation, row_model, record, kind):
    objects, text, _ = oracle_pack(system, [record])
    with torch.no_grad():
        goal = system.layout.goal_encoder(objects, text, _)['vector']
        if kind == 'relation':
            output = relation(objects.features(), objects.valid, goal)
        else:
            hidden = relation._hidden(objects.features())
            output = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1], None, None, True)
    kept = int(objects.valid[0].sum())
    scene = {key: getattr(objects, key)[0, :kept].detach().cpu()
             for key in ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
    scene.update(text=record['instruction']['text'], ref_xy=output['xy'][0, :kept].detach().cpu(),
                 ref_angle=output['angle'][0, :kept].detach().cpu())
    return scene


def scene_objects(scene):
    values = [scene[key].to(stable.DEVICE).unsqueeze(0) for key in
              ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')]
    return Objects(*values).validate()


def anchor_loss(system, relation, row_model, scenes, kind):
    chosen = scenes[:4]
    total = scene_objects(chosen[0]).xy.new_zeros(())
    for scene in chosen:
        objects = scene_objects(scene)
        text, mask = encode_instruction(system, [scene['text']], stable.DEVICE)
        goal = system.layout.goal_encoder(objects, text, mask)['vector']
        if kind == 'relation':
            live = relation(objects.features(), objects.valid, goal)
        else:
            hidden = relation._hidden(objects.features())
            live = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1], None, None, True)
        reference = dict(xy=scene['ref_xy'].to(stable.DEVICE).unsqueeze(0),
                         angle=scene['ref_angle'].to(stable.DEVICE).unsqueeze(0))
        total = total+consistency_loss(live, reference, objects.valid)
    return total/len(chosen)


def bucket_subset(data):
    buckets = {}
    for index, record in enumerate(data.records):
        if not is_full(record):
            continue
        count = len(record['source']['objects'])
        if count not in (2, 4, 6, 8):
            continue
        kind = 'single' if len(record['supervision_v3']['rows']) == 1 else 'multi'
        buckets.setdefault((count, kind), []).append(index)
    chosen = []
    census = {}
    for key, indices in sorted(buckets.items()):
        indices.sort(key=lambda index: data.records[index]['sample_id'])
        census[f'{key[0]}_{key[1]}'] = len(indices)
        if len(indices) <= 32:
            picked = indices
        else:
            picked = [indices[int(i*(len(indices)-1)/31)] for i in range(32)]
        chosen.extend(picked)
    return chosen, census


def split_families(data, indices):
    families = sorted({data.records[index]['scene_family_id'] for index in indices})
    generator = torch.Generator().manual_seed(cap.SEED)
    order = torch.randperm(len(families), generator=generator).tolist()
    dev_n = max(1, len(families)//5)
    dev_families = {families[index] for index in order[:dev_n]}
    dev = [index for index in indices if data.records[index]['scene_family_id'] in dev_families]
    fit = [index for index in indices if data.records[index]['scene_family_id'] not in dev_families]
    return fit, dev


def median(values):
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float).median())


def score_e2e(system, relation, row_model, data, indices):
    system.perception.eval()
    relation.eval()
    row_model.eval()
    scenes = []
    horizontal, vertical = [], []
    with torch.no_grad():
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            try:
                output, objects, visual = predict_poses(system, relation, row_model, image, [record['instruction']['text']])
            except ValueError:
                scenes.append(dict(objects=len(record['source']['objects']), task=False, errors=['count'],
                                   collision=False, boundary=False))
                continue
            source = {key: value.detach().cpu() for key, value in visual['source'].items()}
            layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                          valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                          rollout='free_rollout')
            view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
            obj_ids, truth = recovery_supervision(record)
            truth = {key: value.cpu() if torch.is_tensor(value) else value for key, value in truth.items()}
            meta = dict(sample_id=record['sample_id'], object_ids=obj_ids, record=record)
            judged = evaluate_batch(view, Batch({}, [truth], [meta]), semantic_check=semantic_errors_v3)[0]
            detail = arrangement_errors(record, judged['predicted_targets']) if judged.get('predicted_targets') else None
            if detail is not None:
                horizontal.extend(detail['horizontal_mm'])
                vertical.extend(detail['vertical_mm'])
            scenes.append(dict(objects=len(record['source']['objects']), task=bool(judged['task_success']),
                               errors=list(judged['errors']),
                               collision=bool(detail and detail['collision']),
                               boundary=bool(detail and detail['boundary'])))
    def bucket(count):
        chosen = [scene for scene in scenes if scene['objects'] == count]
        return dict(success=sum(scene['task'] for scene in chosen), total=len(chosen),
                    collision=sum(scene['collision'] for scene in chosen),
                    boundary=sum(scene['boundary'] for scene in chosen))
    return dict(task=sum(scene['task'] for scene in scenes), total=len(scenes),
                by_count={str(count): bucket(count) for count in (2, 4, 6, 8)},
                horizontal=median(horizontal), vertical=median(vertical),
                collision=sum(scene['collision'] for scene in scenes),
                boundary=sum(scene['boundary'] for scene in scenes),
                failures=dict(Counter(error for scene in scenes for error in scene['errors'])))


def refresh_goals(system, records, stored):
    goals = []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk = records[start:start+16]
            objects, text, mask = oracle_pack(system, chunk)
            goal = system.layout.goal_encoder(objects, text, mask)['vector']
            goals.extend(goal[index].detach().cpu() for index in range(len(chunk)))
    return [dict(item, goal=goal) for item, goal in zip(stored, goals)]


def brief_modular(point):
    counts = point['by_count']
    return dict(task=point['task'], total=point['total'],
                by_count={count: dict(success=counts[count]['success'], total=counts[count]['total'])
                          for count in counts},
                row_accuracy=point['row_accuracy'], slot_accuracy=point['slot_accuracy'],
                horizontal=point['horizontal_gap_mm']['median'], vertical=point['vertical_gap_mm']['median'],
                collision=point['collision'], boundary=point['boundary'],
                origin_y_8=counts['8']['origin_y_mm'])


def trainable(relation, row_model, system):
    hidden = [parameter for parameter in relation.hidden.parameters() if parameter.requires_grad]
    encoders = [parameter for parameter in list(system.perception.parameters())
                +list(system.arrangement.text_encoder.parameters()) if parameter.requires_grad]
    goal = [parameter for parameter in system.layout.goal_encoder.parameters() if parameter.requires_grad]
    rows = [parameter for parameter in row_model.parameters() if parameter.requires_grad]
    return hidden, encoders, goal, rows


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('end-to-end joint result already exists')
    stamps = dict(row=stamp(ROW_BEST), relation=stamp(RELATION_BEST), dx=stamp(DX_BEST))
    protocol = json.loads(HOLDOUT.read_text())
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    phase_train, _ = runner.datasets()
    train = ExpandedDataset(DATA, 'train')
    val = ExpandedDataset(DATA, 'validation')
    indices, census = bucket_subset(train)
    fit, dev = split_families(train, indices)
    old_dev = set(protocol['dev_sample_ids'])
    exposure = dict(old_holdout_scenes=len(old_dev), old_holdout_in_historical_training=True,
                    old_holdout_used_for_selection=False,
                    pilot_fit_overlap=sum(train.records[index]['sample_id'] in old_dev for index in fit),
                    pilot_dev_overlap=sum(train.records[index]['sample_id'] in old_dev for index in dev),
                    note='The 260 family-holdout scenes were inside the earlier 1273-scene fit. This pilot does not adopt on them.')
    print('SUBSET', json.dumps(dict(census=census, fit=len(fit), dev=len(dev), exposure=exposure)), flush=True)
    if len(fit) < 16 or not dev:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='subset', exposure=exposure))
        print('STOP subset is too small', flush=True)
        return
    multi_val = [record for record in val.records if multi_row(record)]
    single_val = [record for record in val.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    phase_val = subset(val, 'validation', 94)
    system = load_integrated(device=stable.DEVICE)
    relation = cap.load_head(128, torch.load(RELATION_BEST, map_location='cpu', weights_only=False)['model']).to(stable.DEVICE)
    row_model = NeuralRowPlacement().to(stable.DEVICE)
    row_state = torch.load(ROW_BEST, map_location=stable.DEVICE, weights_only=False)
    row_model.load_state_dict(row_state['model'])
    save_checkpoint(ROOT/'row-reload-probe.pt', row_model, torch.optim.AdamW(row_model.parameters(), lr=ROW_LR),
                    int(row_state['update']), CONFIG)
    fresh = NeuralRowPlacement().to(stable.DEVICE)
    load_checkpoint(ROOT/'row-reload-probe.pt', fresh, map_location=stable.DEVICE)
    if any(not torch.equal(fresh.state_dict()[name], value) for name, value in row_model.state_dict().items()):
        raise RuntimeError('adopted row checkpoint did not reload')
    del fresh
    body_eval = system
    phase_cache = stable.cache_oracle(body_eval, phase_val)
    multi_stored = attach_rows(multi_val, cache_records(body_eval, multi_val))
    single_stored = cache_records(body_eval, single_val)
    oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
    single = score(relation, single_stored, single_val)['task']
    modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')
    print('RELOAD', json.dumps(dict(oracle=oracle, single=single, free=modular['task'])), flush=True)
    if (oracle, single, modular['task']) != (60, 62, 66):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload_baseline',
                                              oracle=oracle, single=single, free=modular['task'],
                                              checkpoints_unchanged=all(stamp(path) == stamps[name] for name, path in
                                                                        (('row', ROW_BEST), ('relation', RELATION_BEST), ('dx', DX_BEST)))))
        print('STOP best checkpoints did not reproduce', flush=True)
        return
    prepare(relation, row_model, system)
    frozen = {name: parameter.detach().clone() for name, parameter in relation.named_parameters()
              if name in frozen_names(relation)}
    phase_records = list(phase_train.records)
    phase_records.sort(key=lambda record: record['sample_id'])
    relation_scenes = [capture(system, relation, row_model, record, 'relation') for record in phase_records[:16]]
    row_records = [train.records[index] for index in fit if multi_row(train.records[index])][:16]
    row_scenes = [capture(system, relation, row_model, record, 'row') for record in row_records]
    dev_e2e = score_e2e(system, relation, row_model, train, dev)
    val_indices = [index for index, record in enumerate(val.records) if multi_row(record)]
    val_e2e = score_e2e(system, relation, row_model, val, val_indices)
    print('E2E0', json.dumps(dict(dev=dev_e2e['task'], dev_total=dev_e2e['total'], val=val_e2e['task'],
                                 val_by=dev_summary(val_e2e))), flush=True)
    hidden, encoders, goal_params, rows = trainable(relation, row_model, system)
    optimizer = torch.optim.AdamW([dict(params=encoders, lr=ENCODER_LR), dict(params=goal_params, lr=GOAL_LR),
                                  dict(params=hidden, lr=HIDDEN_LR), dict(params=rows, lr=ROW_LR)],
                                  weight_decay=0., foreach=False)
    before = dict(perception=clone_state(system.perception), language=clone_state(system.arrangement.text_encoder),
                  goal=clone_state(system.layout.goal_encoder), relation=clone_state(relation), row=clone_state(row_model))
    check = step_check(system, relation, row_model, optimizer, train, fit, relation_scenes, row_scenes, frozen)
    print('STEP', json.dumps(check), flush=True)
    for module, saved in ((system.perception, before['perception']), (system.arrangement.text_encoder, before['language']),
                          (system.layout.goal_encoder, before['goal']), (relation, before['relation']),
                          (row_model, before['row'])):
        restore_state(module, saved)
    restored = all(moved(saved, module) == 0 for module, saved in
                   ((system.perception, before['perception']), (relation, before['relation']), (row_model, before['row'])))
    if not check['passed'] or not restored:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='weight_update', step=check, restored=restored,
                                              exposure=exposure))
        print('STOP weight update did not stay on the trainable path', flush=True)
        return
    optimizer = torch.optim.AdamW([dict(params=encoders, lr=ENCODER_LR), dict(params=goal_params, lr=GOAL_LR),
                                  dict(params=hidden, lr=HIDDEN_LR), dict(params=rows, lr=ROW_LR)],
                                  weight_decay=0., foreach=False)
    write_checkpoint(ROOT/'checkpoint-0000.pt', system, relation, row_model, optimizer, 0)
    baseline = dict(update=0, dev_e2e=dev_e2e, val_e2e=val_e2e, modular=brief_modular(modular),
                    oracle=oracle, single=single)
    (ROOT/'history.jsonl').write_text(json.dumps(slim(baseline))+'\n')
    best = baseline
    write_checkpoint(ROOT/'checkpoint-best.pt', system, relation, row_model, optimizer, 0)
    reason, stale, skips = 'max_updates', 0, 0
    generator = torch.Generator().manual_seed(cap.SEED)
    order, cursor = [], 0
    running, seen = 0., 0
    started = time.perf_counter()
    history = (ROOT/'history.jsonl').open('a')
    update = 0
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit), generator=generator).tolist()
            cursor = 0
        choice = [fit[order[index]] for index in range(cursor, cursor+BATCH)]
        cursor += BATCH
        records = [train.records[index] for index in choice]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        row_model.train()
        relation.train()
        prepared = supervise_arrangement(system, relation, row_model, images, records)
        if prepared is None:
            skips += 1
            if skips >= 30:
                reason = 'no_matched_scenes'
                break
            continue
        skips = 0
        update += 1
        optimizer.zero_grad(set_to_none=True)
        loss = prepared['terms']['loss']+CONFIG['consistency']*(
            anchor_loss(system, relation, row_model, relation_scenes, 'relation')
            +anchor_loss(system, relation, row_model, row_scenes, 'row'))
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoders+goal_params+hidden+rows, 1.)
        optimizer.step()
        if any(not torch.equal(parameter, frozen[name]) for name, parameter in relation.named_parameters()
               if name in frozen):
            reason = 'relation_pose_moved'
            break
        running += float(loss.detach())
        seen += 1
        if update % 25 == 0:
            print('STEP_LOSS', json.dumps(dict(update=update, loss=running/seen)), flush=True)
        if update % EVAL_EVERY and update != MAX_UPDATES:
            continue
        point = evaluate_point(system, relation, row_model, train, dev, val, val_indices, multi_val, multi_stored,
                              phase_cache, phase_val, single_stored, single_val, update, running/max(seen, 1), started)
        running, seen = 0., 0
        history.write(json.dumps(slim(point))+'\n')
        history.flush()
        print('EVAL', json.dumps(slim(point)), flush=True)
        if point['oracle'] < 56 or point['modular']['task'] < 54:
            reason = 'collapse'
            break
        acceptable = point['oracle'] >= 58 and point['modular']['task'] >= 60
        if acceptable and point['dev_e2e']['task'] > best['dev_e2e']['task']:
            best = point
            stale = 0
            write_checkpoint(ROOT/'checkpoint-best.pt', system, relation, row_model, optimizer, update)
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    final = finish(system, relation, row_model, train, dev, val, val_indices, multi_val, multi_stored,
                   phase_cache, phase_val, single_stored, single_val, baseline, best, reason, stamps, exposure, census)
    print('DONE', json.dumps(dict(status=final['status'], stop=final['stop'], update=final['best_update'])), flush=True)


def dev_summary(point):
    return {count: point['by_count'][count]['success'] for count in point['by_count']}


def step_check(system, relation, row_model, optimizer, train, fit, relation_scenes, row_scenes, frozen):
    images = records = None
    matched = False
    for start in range(0, min(len(fit), 40), 2):
        choice = fit[start:start+2]
        records = [train.records[index] for index in choice]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        prepared = supervise_arrangement(system, relation, row_model, images, records)
        if prepared is not None:
            matched = True
            del prepared
            break
    if not matched:
        return dict(passed=False, reason='no matched RGB batch')
    texts = [record['instruction']['text'] for record in records]
    optimizer.zero_grad(set_to_none=True)
    predicted, _, _ = predict_poses(system, relation, row_model, images, texts)
    decoy = torch.zeros((), device=stable.DEVICE, requires_grad=True)
    predicted['xy'].sum().backward()
    inference = dict(perception=grad_norm(system.perception.parameters()),
                     language=grad_norm(system.arrangement.text_encoder.parameters()),
                     goal=grad_norm(system.layout.goal_encoder.parameters()),
                     hidden=grad_norm(relation.hidden.parameters()),
                     row=grad_norm(row_model.parameters()),
                     anchor=grad_norm(relation.anchor.parameters()),
                     decoy_grad=decoy.grad is not None)
    del predicted
    optimizer.zero_grad(set_to_none=True)
    prepared = supervise_arrangement(system, relation, row_model, images, records)
    loss = prepared['terms']['loss']+0.1*(anchor_loss(system, relation, row_model, relation_scenes, 'relation')
                                         +anchor_loss(system, relation, row_model, row_scenes, 'row'))
    loss.backward()
    training = dict(perception=grad_norm(system.perception.parameters()),
                    language=grad_norm(system.arrangement.text_encoder.parameters()),
                    goal=grad_norm(system.layout.goal_encoder.parameters()),
                    hidden=grad_norm(relation.hidden.parameters()),
                    row=grad_norm(row_model.parameters()),
                    anchor=grad_norm(relation.anchor.parameters()))
    before = {name: parameter.detach().clone() for name, parameter in relation.named_parameters()}
    row_before = clone_state(row_model)
    perception_before = clone_state(system.perception)
    language_before = clone_state(system.arrangement.text_encoder)
    goal_before = clone_state(system.layout.goal_encoder)
    torch.nn.utils.clip_grad_norm_(clip_flat(relation, row_model, system), 1.)
    optimizer.step()
    pose_same = all(torch.equal(parameter, frozen[name]) for name, parameter in relation.named_parameters() if name in frozen)
    hidden_moved = any(not torch.equal(parameter, before[name]) for name, parameter in relation.named_parameters()
                       if name.startswith('hidden.'))
    updates = dict(row=moved(row_before, row_model), perception=moved(perception_before, system.perception),
                   language=moved(language_before, system.arrangement.text_encoder),
                   goal=moved(goal_before, system.layout.goal_encoder), hidden=int(hidden_moved))
    connected = all(inference[name]['norm_sum'] > 0 for name in ('perception', 'language', 'goal', 'hidden', 'row'))
    supervised = all(training[name]['norm_sum'] > 0 for name in ('perception', 'language', 'goal', 'hidden', 'row'))
    passed = connected and supervised and inference['anchor']['norm_sum'] == 0 and training['anchor']['norm_sum'] == 0
    passed = passed and not inference['decoy_grad'] and pose_same and all(value > 0 for value in updates.values())
    return dict(passed=passed, kept=prepared['kept'], inference=inference, training=training,
                pose_same=pose_same, updates=updates)


def clip_flat(relation, row_model, system):
    hidden, encoders, goal_params, rows = trainable(relation, row_model, system)
    return encoders+goal_params+hidden+rows


def evaluate_point(system, relation, row_model, train, dev, val, val_indices, multi_val, multi_stored,
                   phase_cache, phase_val, single_stored, single_val, update, loss, started):
    row_model.eval()
    relation.eval()
    dev_e2e = score_e2e(system, relation, row_model, train, dev)
    val_e2e = score_e2e(system, relation, row_model, val, val_indices)
    modular_stored = refresh_goals(system, multi_val, multi_stored)
    modular = evaluate(row_model, relation, modular_stored, multi_val, 'free')
    goals = []
    with torch.no_grad():
        for index, record in enumerate(phase_val.records):
            objects, text, mask = oracle_pack(system, [record])
            goals.append(system.layout.goal_encoder(objects, text, mask)['vector'][0].detach().cpu())
    phase = dict(phase_cache)
    phase['goal'] = torch.stack(goals)
    oracle = stable.official_metrics(relation, phase, phase_val)['task']
    single_goals = refresh_goals(system, single_val, single_stored)
    single = score(relation, single_goals, single_val)['task']
    return dict(update=update, loss=loss, seconds=time.perf_counter()-started, dev_e2e=dev_e2e, val_e2e=val_e2e,
                modular=brief_modular(modular), oracle=oracle, single=single)


def slim(point):
    return dict(update=point['update'], loss=point.get('loss'), oracle=point['oracle'], single=point['single'],
                dev_e2e=point['dev_e2e']['task'], dev_by=dev_summary(point['dev_e2e']),
                val_e2e=point['val_e2e']['task'], val_by=dev_summary(point['val_e2e']),
                modular=point['modular']['task'], modular_by=dev_summary(point['modular']),
                modular_horizontal=point['modular']['horizontal'], modular_row=point['modular']['row_accuracy'])


def write_checkpoint(path, system, relation, row_model, optimizer, update):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(update=update, config=CONFIG,
                    perception=clone_state(system.perception),
                    language=clone_state(system.arrangement.text_encoder),
                    goal=clone_state(system.layout.goal_encoder),
                    relation=clone_state(relation), row=clone_state(row_model),
                    optimizer=optimizer.state_dict()), path)


def finish(system, relation, row_model, train, dev, val, val_indices, multi_val, multi_stored, phase_cache,
           phase_val, single_stored, single_val, baseline, best, reason, stamps, exposure, census):
    saved = torch.load(ROOT/'checkpoint-best.pt', map_location='cpu', weights_only=False)
    restore_state(system.perception, saved['perception'])
    restore_state(system.arrangement.text_encoder, saved['language'])
    restore_state(system.layout.goal_encoder, saved['goal'])
    restore_state(relation, saved['relation'])
    restore_state(row_model, saved['row'])
    final = evaluate_point(system, relation, row_model, train, dev, val, val_indices, multi_val, multi_stored,
                           phase_cache, phase_val, single_stored, single_val, saved['update'], None, time.perf_counter())
    reload_ok = final['dev_e2e']['task'] == best['dev_e2e']['task'] and final['modular']['task'] == best['modular']['task']
    improved = final['dev_e2e']['task'] > baseline['dev_e2e']['task'] and final['val_e2e']['task'] > baseline['val_e2e']['task']
    preserved = (final['oracle'] >= 58 and final['modular']['task'] >= 60 and final['single'] >= 58
                 and final['modular']['horizontal'] <= baseline['modular']['horizontal']+1.)
    status = 'PASS' if improved and preserved and reload_ok and saved['update'] > 0 else 'FAIL'
    report = dict(status=status, stop=reason, best_update=saved['update'], config=CONFIG, census=census,
                  exposure=exposure, baseline=slim(baseline), selected=slim(final),
                  baseline_detail=dict(dev=baseline['dev_e2e'], val=baseline['val_e2e'], modular=baseline['modular']),
                  selected_detail=dict(dev=final['dev_e2e'], val=final['val_e2e'], modular=final['modular']),
                  reload_matches_logged_success=reload_ok, validation_used_for_selection=False, test_used=False,
                  gt_at_inference=False, partial_noop_included=False,
                  checkpoints_unchanged=all(stamp(path) == stamps[name] for name, path in
                                            (('row', ROW_BEST), ('relation', RELATION_BEST), ('dx', DX_BEST))),
                  full_9600_ran=False,
                  full_9600_plan=None if status != 'PASS' else dict(
                      data='train full arrangements of 2 to 8 objects, single-row and multi-row',
                      exclude='test split and partial or no-op scenes',
                      updates='continue from this checkpoint, eval every 250, stop on plateau or if oracle falls below 58 or modular free falls below 60',
                      selection='a family slice held out of that run, not the historical 260 and not validation'))
    stable.write(ROOT/'result.json', report)
    return report


if __name__ == '__main__':
    main()
