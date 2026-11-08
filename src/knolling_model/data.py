"""Manifest-only read adapter. Model inputs and supervision never share a dict."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from knolling_dataset.builder import file_hash
from knolling_dataset.english import audit_english_records
from knolling_dataset.schema import COLORS, SHAPES


@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    supervision: list[dict[str, torch.Tensor]]
    metadata: list[dict]

    def to(self, device, non_blocking=False):
        return Batch({k: v.to(device, non_blocking=non_blocking) for k, v in self.inputs.items()},
                     [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in self.supervision], self.metadata)

    def pin_memory(self):
        return Batch({k: v.pin_memory() for k, v in self.inputs.items()},
                     [{k: v.pin_memory() for k, v in t.items()} for t in self.supervision], self.metadata)


def supervision(record):
    source = {o['object_id']: o for o in record['source']['objects']}
    target = {o['object_id']: o for o in record['target']['objects']}
    if (len(source) != len(record['source']['objects']) or
            len(target) != len(record['target']['objects']) or source.keys() != target.keys()):
        raise ValueError('duplicate or inconsistent source/target object identity')
    ids = sorted(source)
    def pose(objects, key):
        return torch.tensor([[objects[i][key]['x_m'], objects[i][key]['y_m'],
                              objects[i][key]['theta_rad']] for i in ids], dtype=torch.float32)
    s, t = pose(source, 'source_pose'), pose(target, 'target_pose')
    labels = {'source_xy': s[:, :2], 'source_angle': s[:, 2],
              'target_xy': t[:, :2], 'target_angle': t[:, 2],
              'size': torch.tensor([[source[i]['asset']['a_m'], source[i]['asset']['b_m']] for i in ids]),
              'color': torch.tensor([COLORS.index(source[i]['color']) for i in ids]),
              'shape': torch.tensor([SHAPES.index(source[i]['asset']['shape']) for i in ids])}
    return ids, labels


class S0Dataset(Dataset):
    def __init__(self, root, selection):
        self.root = Path(root).resolve()
        selection = json.loads(Path(selection).read_text()) if isinstance(selection, (str, Path)) else selection
        self.selection = selection
        if file_hash(self.root / 'manifest.jsonl') != selection['manifest_sha256']:
            raise ValueError('dataset manifest hash differs from S0 selection')
        provenance = json.loads((self.root / 'provenance.json').read_text())
        if (provenance.get('teacher_policy') != 'balanced_rows_v1.3' or
                provenance.get('language_policy') != 'english_only_v1'):
            raise ValueError('S0 requires the balanced English dataset')
        entries = [json.loads(l) for l in (self.root / 'manifest.jsonl').read_text().splitlines()]
        by_id = {e['sample_id']: e for e in entries}
        ids = selection['sample_ids']
        if len(ids) != 32 or len(set(ids)) != 32:
            raise ValueError('S0 requires exactly 32 distinct samples')
        self.records = []
        for sid in ids:
            entry = by_id[sid]
            path = self._path(entry['annotation'])
            if file_hash(path) != entry['annotation_sha256']:
                raise ValueError(f'{sid}: annotation hash mismatch')
            record = json.loads(path.read_text())
            if (record['sample_id'] != sid or record['split'] != 'train' or entry['split'] != 'train'
                    or not 2 <= len(record['source']['objects']) <= 4 or record['status'] != 'ok'):
                raise ValueError(f'{sid}: not an eligible S0 Train sample')
            image_path = record['source_render_dir'] + '/rgb.png'
            if file_hash(self._path(image_path)) != record['file_hashes'][image_path]:
                raise ValueError(f'{sid}: source image hash mismatch')
            supervision(record)  # Validate ID-based join before any model call.
            self.records.append(record)
        language = audit_english_records(self.records)
        if not language['valid']:
            raise ValueError(language['errors'])

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('dataset path escapes root')
        return path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self._path(record['source_render_dir'] + '/rgb.png')) as image:
            if image.size != (576, 384) or image.mode != 'RGB':
                raise ValueError('expected unmodified 576x384 RGB source')
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        ids, labels = supervision(record)
        return {'image': pixels, 'text': record['instruction']['text'], 'supervision': labels,
                'metadata': {'sample_id': record['sample_id'], 'object_ids': ids, 'record': record}}


class Collator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        # Never use truncation=True or the sentence-transformers pooling wrapper.
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[1] > self.max_length:
            raise ValueError('instruction exceeds 512 tokens; silent truncation forbidden')
        return Batch({'images': torch.stack([s['image'] for s in samples]),
                      'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask'].bool()},
                     [s['supervision'] for s in samples], [s['metadata'] for s in samples])


class FamilySampler(Sampler):
    """One shuffled family visit per epoch; cyclic shuffled instructions per family."""
    def __init__(self, records, seed):
        self.families = {}
        for index, record in enumerate(records):
            self.families.setdefault(record['scene_family_id'], []).append(index)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.families)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        names = sorted(self.families)
        for f in torch.randperm(len(names), generator=g).tolist():
            members = self.families[names[f]]
            cycle, offset = divmod(self.epoch, len(members))
            local = torch.Generator().manual_seed(self.seed + 100003 * (f + 1) + cycle)
            order = torch.randperm(len(members), generator=local).tolist()
            yield members[order[offset]]
