"""Local progress, immutable-input caches and explicit precision policy."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm


def precision_context(config, stage):
    precision = config.get('precision', {}).get(stage, 'fp32')
    if precision == 'fp32':
        return nullcontext()
    if precision != 'bf16' or config['device'] != 'cuda':
        raise ValueError('BF16 requires an explicitly configured CUDA stage')
    return torch.autocast('cuda', dtype=torch.bfloat16)


class CachedDataset:
    """Materialize the tiny S0 suite once, sharing identical source RGB tensors."""
    def __init__(self, dataset):
        self.records = dataset.records
        self.fingerprint = dataset.fingerprint
        self.samples, images = [], {}
        for i, record in enumerate(self.records):
            sample = dataset[i]
            family = record['scene_family_id']
            if family in images:
                sample['image'] = images[family]
            else:
                images[family] = sample['image']
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FrozenTextCache:
    """Cache only frozen encoder outputs keyed by actual input token IDs.

    Padding is rebuilt for each batch. No labels, source metadata or targets are
    used. The wrapped encoder stays FP32 even when layout uses autocast.
    """
    def __init__(self, encode):
        self.encode, self.values = encode, {}

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask):
        lengths = attention_mask.sum(-1).tolist()
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < attention_mask.sum(-1)[:, None]
        if not lengths or min(lengths) < 1 or not torch.equal(attention_mask, expected):
            raise ValueError('text cache requires nonempty right-padded instructions')
        keys = [tuple(row[:n]) for row, n in zip(input_ids.tolist(), lengths)]
        missing = [i for i, key in enumerate(keys) if key not in self.values]
        if missing:
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                states = self.encode(input_ids[missing], attention_mask[missing]).detach()
            for i, state in zip(missing, states):
                self.values[keys[i]] = state[:lengths[i]].clone()
        return torch.stack([torch.cat((self.values[key], self.values[key].new_zeros(
            input_ids.shape[1]-n, self.values[key].shape[-1]))) for key, n in zip(keys, lengths)])


@torch.no_grad()
def warm_text_cache(model, dataset, collator, batch_size, device):
    """Rebuild derived text state in the same order/shapes as initial evaluation.

    The initial evaluator visits one instruction at a time within padded batches.
    Replaying that order avoids changing frozen-encoder batch shapes on resume.
    Only input tokens enter the encoder; labels and images are not transferred.
    """
    if not isinstance(model.encode_text, FrozenTextCache):
        return
    model.encode_text.values.clear()
    for start in range(0, len(dataset), batch_size):
        batch = collator([dataset[i] for i in range(start, min(start+batch_size, len(dataset)))])
        ids = batch.inputs['input_ids'].to(device)
        mask = batch.inputs['attention_mask'].to(device)
        for i in range(len(ids)):
            model.encode_text(ids[i:i+1], mask[i:i+1])


class ProgressLog:
    """tqdm on the local terminal plus append-only per-stage JSONL evidence."""
    def __init__(self, output):
        self.output, self.bar = Path(output), None

    def __call__(self, event, **record):
        with (self.output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event, time=time.time(), **record), allow_nan=False)+'\n')
        if event == 'start':
            labels = dict(grounding='Goal Grounding', micro_pose='Micro Pose Diagnostic',
                          oracle_layout='Stage 1 Arrangement', perception='Stage 2 Perception',
                          predicted_layout='Stage 3a End-to-End', joint='Stage 3b Joint')
            self.bar = tqdm(total=record['max_updates'], initial=record['update'],
                            desc=labels[record['stage']], unit='step', mininterval=1, dynamic_ncols=True)
        elif event == 'update' and self.bar is not None:
            self.bar.update(record['update']-self.bar.n)
            self.bar.set_postfix(loss=f"{record['loss']:.4f}", budget_s=f"{record['remaining_seconds']:.0f}", refresh=False)
        elif event == 'evaluation':
            m = record['metrics']
            tqdm.write(f"{record['stage']} step={record['update']} target_P95_mm={m.get('target_position_p95_mm')} "
                       f"task={m.get('task_success')} axis={m.get('pure_axis_success')} move={m.get('move_accuracy')}")
        elif event == 'extension':
            grant = record['authorization']
            tqdm.write(f"Approved extension from step {grant['origin_update']}, cumulative elapsed "
                       f"{grant['origin_elapsed_seconds']:.2f}s; progress check at step {grant['progress_update']}; "
                       f"total caps {grant['total_max_updates']} updates / {grant['total_max_seconds']}s.")

    def close(self):
        if self.bar is not None:
            self.bar.close()
