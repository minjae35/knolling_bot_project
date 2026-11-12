"""Judge borderline full and partial scenes on the full solver and the router. No test split."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'experiments' / 'end_to_end')]
if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    del os.environ['CUDA_VISIBLE_DEVICES']
os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import torch
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.partial_arrangement import command_is_full
from knolling_model.experiments.size_refinement import (
    encode_instruction, install_selection_head, pack_sources, poses_from_detection)
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import select_queries
from score_end_to_end import DATA
from train_selector_full import judge_scene, load_adopted


def _forced_full(system, relation, model, objects, encoded):
    goal = encoded['vector']
    hidden = relation._hidden(objects.features())
    output = model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                   None, None, True, colors=objects.color)
    from knolling_model.experiments.constraint_decode import select_layout
    from knolling_model.experiments.geometry_decoder import apply_geometry
    output = select_layout(model, hidden, objects, goal, output)
    placed = apply_geometry(output, objects, goal)
    placed['row'] = output['row']
    return placed


def main():
    runner.setup()
    data = ExpandedDataset(DATA, 'validation')
    system, relation, model, head, _saved = load_adopted()
    install_selection_head(system)
    for index, record in enumerate(data.records):
        mode = record['mode']
        if mode == 'full' and not is_full(record):
            continue
        if mode not in ('full', 'partial'):
            continue
        image = data[index]['image'][None].to(stable.DEVICE)
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        with torch.no_grad():
            visual = system.perception(image)
        queries = select_queries(visual['source'], system.threshold)
        if any(len(item) < 1 or len(item) > 8 for item in queries):
            continue
        objects = pack_sources(visual['source'], queries)
        text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
        encoded = system.layout.goal_encoder(objects, text, mask)
        count = int(objects.valid[0].sum())
        low = float(encoded['selected'][0, :count].min())
        full_route = command_is_full(encoded['keep_logit'], encoded['selected'], objects.valid)
        interesting = (mode == 'full' and not full_route) or (mode == 'partial' and low >= 0.20)
        if not interesting:
            continue
        current, _objects, visual = poses_from_detection(
            system, relation, model, image, [record['instruction']['text']], head)
        forced = _forced_full(system, relation, model, objects, encoded)
        current_task = judge_scene(
            current, objects, visual, record, identities, truth, 0, system.threshold, detail=False)['task']
        forced_task = judge_scene(
            forced, objects, visual, record, identities, truth, 0, system.threshold, detail=False)['task']
        print('BORDER', mode, record['sample_id'], 'min', round(low, 4),
              'current', current_task, 'forced_full', forced_task, 'route', current.get('route'), flush=True)


if __name__ == '__main__':
    main()
