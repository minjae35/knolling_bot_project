"""One probe: can absolute row coordinates place 4, 6, and 8 objects?

Eight multi-row training scenes are fit with teacher rows and teacher slots.
The relation head stays frozen and is not a neighbor chain here. Predicted
rows and slots are trained only after those scenes are actually placed.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, collate, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import NeuralRowPlacement, assignment_loss, order_loss, placement_loss
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_transition import choose

ROOT = Path('artifacts/row_layout')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
COORDINATE_STEPS = 2000
SELECTOR_STEPS = 1000
EVAL_EVERY = 50


def attach_rows(records, stored):
    for record, item in zip(records, stored):
        ids, truth = recovery_supervision(record)
        if not torch.allclose(item['target_xy'], truth['target_xy'][:len(ids)]):
            raise RuntimeError('cached targets do not match the supervision order')
        location = {oid: index for index, oid in enumerate(ids)}
        row = torch.full((len(ids),), -1)
        slot = torch.full((len(ids),), -1)
        seen = []
        for row_id, members in enumerate(record['supervision_v3']['rows']):
            for slot_id, oid in enumerate(members):
                row[location[oid]] = row_id
                slot[location[oid]] = slot_id
                seen.append(oid)
        if row.min() < 0 or len(seen) != len(set(seen)) or set(seen) != set(ids):
            raise RuntimeError('supervision rows do not cover every object once')
        item['row'] = row
        item['slot'] = slot
    return stored


def collate_rows(items, device):
    packed = collate(items, device)
    width = packed['valid'].shape[1]
    for key in ('row', 'slot'):
        rows = []
        for item in items:
            value = item[key]
            missing = width-value.shape[0]
            rows.append(torch.cat((value, value.new_full((missing,), -1))) if missing else value)
        packed[key] = torch.stack(rows).to(device)
    return packed


def hidden_of(head, features):
    with torch.no_grad():
        return head._hidden(features)


def heavy_pass(scenes):
    chosen = [scene for scene in scenes if scene['objects'] in (4, 6, 8)]
    return len(chosen) == 6 and all(scene['task'] for scene in chosen)


def measure(model, head, stored, records, mode):
    model.eval()
    scenes = []
    errors = []
    with torch.no_grad():
        packed = collate_rows(stored, stable.DEVICE)
        hidden = hidden_of(head, packed['features'])
        if mode == 'teacher':
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], False)
        elif mode == 'free':
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], None, None, True)
        else:
            raise ValueError(mode)
        delta = torch.linalg.vector_norm(output['xy']-packed['target_xy'], dim=-1)
        valid = packed['valid']
        row_hits = int(((output['row'] == packed['row']) & valid).sum())
        slot_hits = int(((output['slot'] == packed['slot']) & valid).sum())
        labeled = int(valid.sum())
        for local, record in enumerate(records):
            kept = stored[local]['valid'].shape[0]
            values = [float(value)*1000 for value in delta[local, :kept]]
            errors.extend(values)
            item = {key: packed[key][local:local+1, :kept].cpu() for key in
                    ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
            objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                              item['valid'], item['query_index'])
            layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                          angle=output['angle'][local:local+1, :kept].cpu(),
                          valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
            ids, truth = recovery_supervision(record)
            meta = dict(sample_id=record['sample_id'], object_ids=ids, record=record)
            judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [meta]),
                                    semantic_check=semantic_errors_v3)[0]
            detail = arrangement_errors(record, judged['predicted_targets']) if judged.get('predicted_targets') else None
            scenes.append(dict(sample_id=record['sample_id'], task=bool(judged['task_success']),
                               objects=len(record['source']['objects']),
                               rows=len(record['supervision_v3']['rows']),
                               median_mm=float(torch.tensor(values).median()),
                               max_mm=max(values), errors=judged['errors'],
                               row_match=bool(detail and detail['row_match'])))
    by_count = {str(count): sum(scene['task'] for scene in scenes if scene['objects'] == count)
                for count in (2, 4, 6, 8)}
    return dict(task=sum(scene['task'] for scene in scenes), total=len(scenes), by_count=by_count,
                median_mm=float(torch.tensor(errors).median()), max_mm=max(errors),
                row_accuracy=row_hits/labeled, slot_accuracy=slot_hits/labeled,
                failures=dict(Counter(error for scene in scenes for error in scene['errors'])),
                scenes=scenes, passed=heavy_pass(scenes))


def brief(point):
    return dict(task=point['task'], by_count=point['by_count'], median_mm=point['median_mm'],
                max_mm=point['max_mm'], row_accuracy=point['row_accuracy'],
                slot_accuracy=point['slot_accuracy'], failures=point['failures'], passed=point['passed'])


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row layout result already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    phase_val = subset(val_data, 'validation', 94)
    multi_train = [record for record in train_data.records if multi_row(record)]
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    probe_records = choose(multi_train)
    if len(single_val) != 104 or [len(record['source']['objects']) for record in probe_records] != [2, 2, 4, 4, 6, 6, 8, 8]:
        raise RuntimeError('probe or single-row pool changed')
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    head.requires_grad_(False)
    head.eval()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(body, phase_val)
    single_stored = cache_records(body, single_val)
    probe_stored = attach_rows(probe_records, cache_records(body, probe_records))
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single)), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    model = NeuralRowPlacement().to(stable.DEVICE)
    model.angle.load_state_dict(head.angle.state_dict())
    model.order_query.load_state_dict(head.query.state_dict())
    with torch.no_grad():
        model.order_begin.copy_(head.begin)
    for parameter in model.selector_parameters():
        parameter.requires_grad_(False)
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()
    packed = collate_rows(probe_stored, stable.DEVICE)

    def log(stage, step, point, extra):
        payload = dict(stage=stage, update=step, seconds=time.perf_counter()-started, **extra, **brief(point))
        history.write(json.dumps(payload)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(stage=stage, update=step, task=point['task'], by_count=point['by_count'],
                                     median_mm=point['median_mm'], passed=point['passed'],
                                     failures=point['failures'])), flush=True)
        return payload

    point = measure(model, head, probe_stored, probe_records, 'teacher')
    baseline = log('coordinate', 0, point, dict(loss=None))
    optimizer = torch.optim.AdamW(model.coordinate_parameters(), lr=LR, weight_decay=0., foreach=False)
    reason = 'coordinate_failed'
    passed = False
    for step in range(1, COORDINATE_STEPS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        hidden = hidden_of(head, packed['features'])
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], False)
        losses = placement_loss(output, packed['target_xy'], packed['target_angle'], packed['valid'])
        losses['loss'].backward()
        if step == 1:
            coordinate_grad = sum(parameter.grad.abs().sum().item() for parameter in model.coordinate_parameters()
                                  if parameter.grad is not None)
            selector_grad = any(parameter.grad is not None for parameter in model.selector_parameters())
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            print('PATH', json.dumps(dict(coordinate=coordinate_grad, selector=selector_grad, head=head_grad)), flush=True)
            if coordinate_grad == 0 or selector_grad or head_grad:
                reason = 'coordinate_path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.coordinate_parameters(), 10.)
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == COORDINATE_STEPS:
            point = measure(model, head, probe_stored, probe_records, 'teacher')
            log('coordinate', step, point, dict(loss=float(losses['loss']), xy=float(losses['xy']),
                                                angle=float(losses['angle'])))
            if point['passed']:
                passed = True
                reason = 'coordinate_passed'
                break
    free = None
    if passed:
        for parameter in model.coordinate_parameters():
            parameter.requires_grad_(False)
        for parameter in model.selector_parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.selector_parameters(), lr=LR, weight_decay=0., foreach=False)
        held_coordinates = measure(model, head, probe_stored, probe_records, 'teacher')
        if not held_coordinates['passed']:
            reason = 'coordinates_moved'
            passed = False
        else:
            reason = 'selector_failed'
            for step in range(1, SELECTOR_STEPS+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                hidden = hidden_of(head, packed['features'])
                output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
                loss = assignment_loss(output, packed['row'], packed['valid'])+order_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.selector_parameters(), 1.)
                optimizer.step()
                if step % EVAL_EVERY == 0 or step == SELECTOR_STEPS:
                    free = measure(model, head, probe_stored, probe_records, 'free')
                    teacher = measure(model, head, probe_stored, probe_records, 'teacher')
                    log('free', step, free, dict(loss=float(loss), teacher_passed=teacher['passed'],
                                                teacher_task=teacher['task']))
                    if not teacher['passed']:
                        reason = 'coordinates_moved'
                        break
                    if free['passed']:
                        reason = 'free_passed'
                        break
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    status = 'PASS' if reason == 'free_passed' and oracle_end == 60 and single_end == 62 else 'FAIL'
    history.close()
    result = dict(status=status, stop=reason, pilot=False,
                  baseline=dict(task=baseline['task'], by_count=baseline['by_count'],
                                median_mm=baseline['median_mm'], passed=baseline['passed']),
                  coordinate=brief(point), free=None if free is None else brief(free),
                  oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
                  relation_parameters_unchanged=unchanged,
                  modular_checkpoint_unchanged=stamp(MODULAR) == held,
                  full_training_recommended=status == 'PASS', full_9600=False, joint=False, test_used=False,
                  adopted=False)
    stable.write(ROOT/'result.json', result)
    if reason == 'free_passed':
        torch.save(dict(model=model.state_dict(), stage=reason), ROOT/'checkpoint-probe.pt')
    print('DONE', json.dumps(dict(status=status, stop=reason, oracle=oracle_end, single=single_end,
                                 unchanged=unchanged, coordinate=point['by_count'],
                                 free=None if free is None else free['by_count'])), flush=True)


if __name__ == '__main__':
    main()
