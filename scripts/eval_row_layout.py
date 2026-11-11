"""Evaluate the saved row-layout probe. No training and no weight updates."""
import json
from collections import Counter
from pathlib import Path

import torch
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import NeuralRowPlacement
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
from run_row_layout import attach_rows, collate_rows, hidden_of
from run_row_transition import choose

ROOT = Path('artifacts/row_layout')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
PROBE = ROOT/'checkpoint-probe.pt'
DATA = Path('artifacts/generalization_20000/dataset')


def stats(values):
    if not values:
        return dict(median=None, p95=None, n=0)
    tensor = torch.tensor(values, dtype=torch.float)
    return dict(median=float(tensor.median()), p95=float(torch.quantile(tensor, .95)), n=len(values))


def evaluate(model, head, stored, records, mode):
    model.eval()
    scenes = []
    positions, horizontal, vertical = [], [], []
    row_hits = slot_hits = labeled = 0
    with torch.no_grad():
        for start in range(0, len(records), 16):
            indices = range(start, min(start+16, len(records)))
            chunk_records = [records[index] for index in indices]
            packed = collate_rows([stored[index] for index in indices], stable.DEVICE)
            hidden = hidden_of(head, packed['features'])
            widths, heights = packed['size'][..., 0], packed['size'][..., 1]
            if mode == 'free':
                output = model(hidden, packed['valid'], packed['goal'], widths, heights, None, None, True)
            elif mode == 'teacher':
                output = model(hidden, packed['valid'], packed['goal'], widths, heights, packed['row'], packed['slot'], False)
            else:
                raise ValueError(mode)
            delta = torch.linalg.vector_norm(output['xy']-packed['target_xy'], dim=-1)
            valid = packed['valid']
            row_hits += int(((output['row'] == packed['row']) & valid).sum())
            slot_hits += int(((output['slot'] == packed['slot']) & valid).sum())
            labeled += int(valid.sum())
            for local, record in enumerate(chunk_records):
                kept = stored[start+local]['valid'].shape[0]
                values = [float(item)*1000 for item in delta[local, :kept]]
                positions.extend(values)
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
                if detail is not None:
                    horizontal.extend(detail['horizontal_mm'])
                    vertical.extend(detail['vertical_mm'])
                count = len(record['source']['objects'])
                valid_row = packed['valid'][local, :kept]
                used_row = output['row'][local, :kept].clamp_min(0)
                used_slot = output['slot'][local, :kept]
                slot0 = valid_row & (used_slot == 0)
                origin_error = (output['origins'][local].index_select(0, used_row)-packed['target_xy'][local, :kept]).abs()*1000
                origin_x = [float(item) for item in origin_error[slot0, 0]]
                origin_y = [float(item) for item in origin_error[slot0, 1]]
                origin_e = [float(item) for item in torch.linalg.vector_norm(origin_error[slot0], dim=-1)]
                row_hit = (output['row'][local, :kept] == packed['row'][local, :kept]) & valid_row
                slot_hit = (output['slot'][local, :kept] == packed['slot'][local, :kept]) & valid_row
                scenes.append(dict(objects=count, rows=len(record['supervision_v3']['rows']),
                                   task=bool(judged['task_success']), errors=judged['errors'],
                                   boundary='boundary' in judged['errors'], collision='collision' in judged['errors'],
                                   row_match=bool(detail and detail['row_match']),
                                   row_exact=bool(row_hit[valid_row].all()) if bool(valid_row.any()) else False,
                                   slot_exact=bool(slot_hit[valid_row].all()) if bool(valid_row.any()) else False,
                                   row_hits=int(row_hit.sum()), slot_hits=int(slot_hit.sum()), labeled=int(valid_row.sum()),
                                   horizontal_mm=list(detail['horizontal_mm']) if detail is not None else [],
                                   vertical_mm=list(detail['vertical_mm']) if detail is not None else [],
                                   position_mm=values, origin_mm=origin_e, origin_x_mm=origin_x, origin_y_mm=origin_y))
    def bucket(count):
        chosen = [scene for scene in scenes if scene['objects'] == count]
        xy = [value for scene in chosen for value in scene['position_mm']]
        labeled_count = sum(scene['labeled'] for scene in chosen)
        return dict(success=sum(scene['task'] for scene in chosen), total=len(chosen),
                    row_exact=sum(scene['row_exact'] for scene in chosen),
                    slot_exact=sum(scene['slot_exact'] for scene in chosen),
                    row_match=sum(scene['row_match'] for scene in chosen),
                    row_accuracy=None if not labeled_count else sum(scene['row_hits'] for scene in chosen)/labeled_count,
                    slot_accuracy=None if not labeled_count else sum(scene['slot_hits'] for scene in chosen)/labeled_count,
                    boundary=sum(scene['boundary'] for scene in chosen),
                    collision=sum(scene['collision'] for scene in chosen),
                    position_mm=stats(xy),
                    origin_mm=stats([value for scene in chosen for value in scene['origin_mm']]),
                    origin_x_mm=stats([value for scene in chosen for value in scene['origin_x_mm']]),
                    origin_y_mm=stats([value for scene in chosen for value in scene['origin_y_mm']]),
                    horizontal_gap_mm=stats([value for scene in chosen for value in scene['horizontal_mm']]),
                    vertical_gap_mm=stats([value for scene in chosen for value in scene['vertical_mm']]))
    return dict(task=sum(scene['task'] for scene in scenes), total=len(scenes),
                by_count={str(count): bucket(count) for count in (2, 4, 6, 8)},
                row_accuracy=None if not labeled else row_hits/labeled,
                slot_accuracy=None if not labeled else slot_hits/labeled,
                row_exact_scenes=sum(scene['row_exact'] for scene in scenes),
                slot_exact_scenes=sum(scene['slot_exact'] for scene in scenes),
                row_match=sum(scene['row_match'] for scene in scenes),
                boundary=sum(scene['boundary'] for scene in scenes),
                collision=sum(scene['collision'] for scene in scenes),
                position_mm=stats(positions),
                origin_mm=stats([value for scene in scenes for value in scene['origin_mm']]),
                origin_x_mm=stats([value for scene in scenes for value in scene['origin_x_mm']]),
                origin_y_mm=stats([value for scene in scenes for value in scene['origin_y_mm']]),
                horizontal_gap_mm=stats(horizontal), vertical_gap_mm=stats(vertical),
                vertical_within_2mm=None if not vertical else sum(value <= 2. for value in vertical)/len(vertical),
                failures=dict(Counter(error for scene in scenes for error in scene['errors'])))


def main():
    if not PROBE.is_file():
        raise RuntimeError('row layout probe checkpoint is missing')
    held = stamp(MODULAR)
    probe_stamp = stamp(PROBE)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    phase_val = subset(val_data, 'validation', 94)
    multi_train = [record for record in train_data.records if multi_row(record)]
    multi_val = [record for record in val_data.records if multi_row(record)]
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    probe_records = choose(multi_train)
    counts = Counter(len(record['source']['objects']) for record in multi_val)
    if len(single_val) != 104 or len(multi_val) != 152 or dict(counts) != {2: 30, 4: 40, 6: 40, 8: 42}:
        raise RuntimeError(f'validation pools changed: {len(single_val)} {len(multi_val)} {dict(counts)}')
    overlap = {record['sample_id'] for record in probe_records} & {record['sample_id'] for record in multi_val}
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    head.requires_grad_(False)
    head.eval()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    model = NeuralRowPlacement().to(stable.DEVICE)
    model.load_state_dict(torch.load(PROBE, map_location=stable.DEVICE, weights_only=False)['model'])
    model.requires_grad_(False)
    model.eval()
    phase_cache = stable.cache_oracle(body, phase_val)
    single_stored = cache_records(body, single_val)
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    probe_stored = attach_rows(probe_records, cache_records(body, probe_records))
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('BASE', json.dumps(dict(oracle=oracle, single=single)), flush=True)
    free = evaluate(model, head, val_stored, multi_val, 'free')
    teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    train_free = evaluate(model, head, probe_stored, probe_records, 'free')
    print('VAL', json.dumps(dict(task=free['task'], by_count={key: value['success'] for key, value in free['by_count'].items()},
                                row=free['row_accuracy'], slot=free['slot_accuracy'],
                                position=free['position_mm'], boundary=free['boundary'], collision=free['collision'])),
          flush=True)
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    result = dict(evaluation_only=True, trained=False, architecture_changed=False, test_used=False,
                  oracle=oracle, single=single, oracle_expected=60, single_expected=62,
                  preserved=oracle == 60 and single == 62, overlap_with_probe=len(overlap),
                  validation_free=free, validation_teacher=teacher, probe_free=train_free,
                  relation_parameters_unchanged=unchanged,
                  modular_checkpoint_unchanged=stamp(MODULAR) == held,
                  probe_checkpoint_unchanged=stamp(PROBE) == probe_stamp)
    stable.write(ROOT/'validation.json', result)
    print('DONE', json.dumps(dict(oracle=oracle, single=single, val=free['task'], probe=train_free['task'],
                                 teacher=teacher['task'], unchanged=unchanged)), flush=True)


if __name__ == '__main__':
    main()
