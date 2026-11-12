"""Print scores and instruction text for the borderline scenes. No test split."""
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
from knolling_model.experiments.size_refinement import encode_instruction, install_selection_head, pack_sources
from knolling_model.recovery.selection import select_queries
from score_end_to_end import DATA
from train_selector_full import load_adopted

WANTED = {
    'expanded_9934f3f23b511b3e40aa3aa3_0',
    'expanded_9934f3f23b511b3e40aa3aa3_1',
    'expanded_a99a7c6d51c25f858b8f622d_2',
    'expanded_4bc8ccf0abaf3e5be4e8753c_2',
    'expanded_ff2eeac17172fd926ca3ccb7_2',
    'expanded_0c99a5757507cfac96f5c33a_2',
}


def main():
    runner.setup()
    data = ExpandedDataset(DATA, 'validation')
    system, _relation, _model, _head, _saved = load_adopted()
    install_selection_head(system)
    found = 0
    for index, record in enumerate(data.records):
        if record['sample_id'] not in WANTED:
            continue
        image = data[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
            queries = select_queries(visual['source'], system.threshold)
            objects = pack_sources(visual['source'], queries)
            text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
            encoded = system.layout.goal_encoder(objects, text, mask)
        count = int(objects.valid[0].sum())
        scores = [round(float(item), 4) for item in encoded['selected'][0, :count]]
        print(record['mode'], record['sample_id'], 'keep', round(float(encoded['keep_logit'].sigmoid()[0]), 4),
              'scores', scores, 'text', record['instruction']['text'], flush=True)
        found += 1
        if found == len(WANTED):
            break


if __name__ == '__main__':
    main()
