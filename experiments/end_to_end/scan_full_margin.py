"""Print full scenes the router does not call full, and the partial score margin. No test split."""
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
from knolling_model.experiments.size_refinement import encode_instruction, install_selection_head, pack_sources
from knolling_model.recovery.selection import select_queries
from score_end_to_end import DATA
from train_selector_full import load_adopted


def main():
    runner.setup()
    data = ExpandedDataset(DATA, 'validation')
    system, _relation, _model, _head, _saved = load_adopted()
    install_selection_head(system)
    partial_mins = []
    for index, record in enumerate(data.records):
        if record['mode'] not in ('full', 'partial'):
            continue
        if record['mode'] == 'full' and not is_full(record):
            continue
        image = data[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
        queries = select_queries(visual['source'], system.threshold)
        if any(len(item) < 1 or len(item) > 8 for item in queries):
            continue
        objects = pack_sources(visual['source'], queries)
        text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
        encoded = system.layout.goal_encoder(objects, text, mask)
        count = int(objects.valid[0].sum())
        scores = encoded['selected'][0, :count].detach().cpu()
        low = float(scores.min())
        if record['mode'] == 'partial':
            partial_mins.append(low)
            continue
        if not command_is_full(encoded['keep_logit'], encoded['selected'], objects.valid):
            print('FULL_MISS', record['sample_id'], 'min', round(low, 4),
                  'scores', [round(float(item), 4) for item in scores], flush=True)
    ordered = sorted(partial_mins)
    print('PARTIAL_MIN', dict(
        n=len(ordered), median=round(ordered[len(ordered) // 2], 4),
        p90=round(ordered[int(len(ordered) * .9)], 4),
        above_40=sum(item >= .40 for item in ordered),
        above_35=sum(item >= .35 for item in ordered)), flush=True)


if __name__ == '__main__':
    main()
