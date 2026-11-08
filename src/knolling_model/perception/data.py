"""Read-only, source-only perception view of the existing Pilot and Recovery data.

No teacher is invoked, no missing annotation is synthesized, and instruction
variants never multiply perception examples. Existing physical-envelope labels
are used for the reused RT-DETR criterion; mask boxes remain distinct labels.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.rendering import validate_render
from knolling_dataset.schema import COLORS, SHAPES, digest, scene_from_dict
from knolling_model.recovery.contracts import detector_labels
from knolling_model.recovery.data import assert_family_disjoint


def source_truth(record):
    """Read only observed source annotations, with no target/instruction dependency."""
    objects = sorted(record['source']['objects'], key=lambda o: o['object_id'])
    ids = [o['object_id'] for o in objects]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty source or duplicate object IDs')
    return ids, dict(
        source_xy=torch.tensor([[o['source_pose']['x_m'], o['source_pose']['y_m']] for o in objects]),
        source_angle=torch.tensor([o['source_pose']['theta_rad'] for o in objects]),
        size=torch.tensor([[o['asset']['a_m'], o['asset']['b_m']] for o in objects]),
        color=torch.tensor([COLORS.index(o['color']) for o in objects]),
        shape=torch.tensor([SHAPES.index(o['asset']['shape']) for o in objects]))


class SourceCatalog:
    def __init__(self, root, manifest_sha256, recovery_path, recovery_sha256, *, check_projection=True):
        self.root = Path(root).resolve()
        manifest = self.path('manifest.jsonl')
        if file_hash(manifest) != manifest_sha256 or file_hash(recovery_path) != recovery_sha256:
            raise ValueError('Pilot/Recovery provenance hash mismatch')
        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len({e['sample_id'] for e in entries}) != len(entries):
            raise ValueError('duplicate Pilot sample ID')
        families, records, rgbs, states, asset_splits = {}, [], {}, {}, {}
        for entry in entries:
            annotation = self.path(entry['annotation'])
            if file_hash(annotation) != entry['annotation_sha256']:
                raise ValueError('Pilot annotation hash mismatch')
            record = json.loads(annotation.read_text())
            family, split = record['scene_family_id'], record['split']
            if (record['sample_id'] != entry['sample_id'] or family != entry['scene_family_id']
                    or split != entry['split'] or record['source']['split'] != split
                    or record['source']['family_id'] != family or record['status'] != 'ok'
                    or split not in ('train', 'validation', 'test', 'development')):
                raise ValueError('Pilot source identity/split mismatch')
            source_truth(record)
            if family in families:
                first = families[family]
                if (first['source'] != record['source'] or first['split'] != split
                        or not self._source_annotation_equivalent(first, record)
                        or first['source_render_dir'] != record['source_render_dir']):
                    raise ValueError('family contains inconsistent source annotations')
            else:
                families[family] = record
                self._validate_source(record, check_projection)
                rgb = record['file_hashes'][record['source_render_dir']+'/rgb.png']
                # Ignore family/seed/IDs while comparing actual physical source states.
                state = digest(sorted([dict(asset=o['asset'], color=o['color'], pose=o['source_pose'])
                                       for o in record['source']['objects']], key=digest))
                for table, key in ((rgbs, rgb), (states, state)):
                    if key in table and table[key] != family:
                        raise ValueError('identical RGB/source state under different families')
                    table[key] = family
                for obj in record['source']['objects']:
                    af = obj['asset']['family_id']
                    if af in asset_splits and asset_splits[af] != split:
                        raise ValueError('asset family leaked across splits')
                    asset_splits[af] = split
            # Hash verification applies even to duplicate instruction variants.
            for name in ('rgb.png', 'instance_mask.png', 'render.json'):
                rel = record['source_render_dir']+'/'+name
                if file_hash(self.path(rel)) != record['file_hashes'][rel]:
                    raise ValueError('source RGB/mask/metadata hash mismatch')
            records.append(record)
        recovery = json.loads(Path(recovery_path).read_text())
        if recovery.get('schema') != 'recovery_dataset_v3':
            raise ValueError('unsupported Recovery annotation schema')
        recovery_records = recovery['records']
        for record in recovery_records:
            parent = families.get(record['scene_family_id'])
            if parent is None or parent['source'] != record['source'] or parent['split'] != record['split']:
                raise ValueError('Recovery source does not match verified Pilot source')
            rgb = parent['source_render_dir']+'/rgb.png'
            if record['source_render_dir'] != parent['source_render_dir'] or record['file_hashes'] != {rgb: parent['file_hashes'][rgb]}:
                raise ValueError('Recovery RGB provenance differs from Pilot')
        partitions = {s: [r for r in records if r['split'] == s]
                      for s in ('train', 'validation', 'test', 'development')}
        assert_family_disjoint(partitions)
        self.records = sorted(families.values(), key=lambda r: r['scene_family_id'])
        self.fingerprint = digest(['perception_sources_v1', manifest_sha256, recovery_sha256,
                                   [(r['scene_family_id'], r['split']) for r in self.records]])
        self.report = dict(schema='perception_annotation_audit_v1', valid=True,
            fingerprint=self.fingerprint, pilot_records=len(records), unique_sources=len(families),
            pilot_records_by_split=dict(Counter(r['split'] for r in records)),
            sources_by_split=dict(Counter(r['split'] for r in self.records)),
            objects_by_split={s: sum(len(r['source']['objects']) for r in v if r is families[r['scene_family_id']])
                              for s, v in partitions.items()},
            object_counts=dict(Counter(len(r['source']['objects']) for r in self.records)),
            recovery_records=len(recovery_records),
            recovery_unique_sources=len({r['scene_family_id'] for r in recovery_records}),
            recovery_new_sources=0,
            recovery_missing_duplicate_detection_fields=sum('objects' not in r for r in recovery_records),
            recovery_annotation_policy='source/RGB exact join to hash-verified Pilot; no generated labels',
            labels=['source_pose_xy_theta', 'intrinsic_major_minor_m', 'physical_envelope',
                    'mask_bbox_xyxy_px_exclusive_max', 'instance_mask', 'color', 'shape'],
            projection_checked=check_projection, train_validation_family_overlap=0,
            mask_use='audit only; no segmentation head', height_use='not predicted or consumed')

    @staticmethod
    def _source_annotation_equivalent(first, record):
        # Compare source fields only; target boxes, moves and locks vary by instruction.
        excluded = {'target_center_uv', 'target_center_px', 'target_mask_bbox_xyxy_px', 'locked', 'move_label'}
        left = [{k: v for k, v in o.items() if k not in excluded} for o in first['objects']]
        right = [{k: v for k, v in o.items() if k not in excluded} for o in record['objects']]
        if left != right:
            raise ValueError('duplicate source labels disagree')
        return True

    def path(self, relative):
        p = (self.root/relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('dataset path escapes root')
        return p

    def _validate_source(self, r, projection):
        directory = self.path(r['source_render_dir'])
        meta = json.loads((directory/'render.json').read_text())
        if (meta['camera_type'] != 'ORTHO' or meta['resolution'] != [576, 384]
                or meta['ortho_scale_m'] != .6 or meta['camera_position_m'] != [.3, .2, 1]):
            raise ValueError('BEV calibration mismatch')
        with Image.open(directory/'rgb.png') as image:
            if image.mode != 'RGB' or image.size != (576, 384):
                raise ValueError('unexpected source RGB')
        with Image.open(directory/'instance_mask.png') as image:
            mask = np.array(image)
        if mask.shape != (384, 576):
            raise ValueError('unexpected source mask')
        ids, truth = source_truth(r)
        labels = detector_labels([truth])[0]['boxes'].double().numpy()
        annotations = {o['object_id']: o for o in r['objects']}
        if len(annotations) != len(ids) or set(annotations) != set(ids):
            raise ValueError('missing/duplicate source object annotation')
        expected_instances = set()
        source_by_id = {o['object_id']: o for o in r['source']['objects']}
        for j, oid in enumerate(ids):
            a, src = annotations[oid], source_by_id[oid]
            instance = a['instance_id']
            expected_instances.add(instance)
            if meta['instance_ids'].get(str(instance)) != oid:
                raise ValueError('mask instance-to-object mapping mismatch')
            ys, xs = np.where(mask == instance)
            if not len(xs) or a['source_mask_bbox_xyxy_px'] != [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]:
                raise ValueError('missing/inconsistent mask bounding box')
            x, y, w, h = labels[j]
            env = [x-w/2, 1-y-h/2, x+w/2, 1-y+h/2]
            xy = truth['source_xy'][j].double().numpy()
            if (not np.allclose(env, a['source_envelope_xyxy_uv'], atol=1e-6, rtol=0)
                    or not np.allclose(xy/[.6, .4], a['source_center_uv'], atol=1e-6, rtol=0)
                    or not np.allclose([576*x-.5, 384*y-.5], a['source_center_px'], atol=1e-4, rtol=0)
                    or a['dimensions_m'] != [src['asset'][k] for k in ('a_m','b_m','h_m')]
                    or a['asset_id'] != src['asset']['asset_id']
                    or a['asset_family_id'] != src['asset']['family_id']
                    or a['shape'] != src['asset']['shape'] or a['color'] != src['color']
                    or not all(math.isfinite(v) for v in (*xy, *a['dimensions_m'], float(truth['source_angle'][j])))
                    or not a['dimensions_m'][0] >= a['dimensions_m'][1] > 0):
                raise ValueError('source pose/dimension/coordinate annotation mismatch')
        if len(expected_instances) != len(ids) or set(np.unique(mask)) != expected_instances | {0}:
            raise ValueError('mask instances incomplete or duplicated')
        if projection:
            result = validate_render(scene_from_dict(r['source']), 'source_pose', directory)
            if not result['valid']:
                raise ValueError(f'independent source RGB/mask/pose projection failed: {result["errors"]}')

    def dataset(self, split):
        return PerceptionDataset(self, split)


class PerceptionDataset(torch.utils.data.Dataset):
    def __init__(self, catalog, split):
        if split not in ('train', 'validation', 'test', 'development'):
            raise ValueError('unknown partition')
        self.catalog, self.split = catalog, split
        self.records = [r for r in catalog.records if r['split'] == split]
        if not self.records:
            raise ValueError('empty perception partition')
        self.fingerprint = digest([catalog.fingerprint, split])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(self.catalog.path(r['source_render_dir']+'/rgb.png')) as image:
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()/255
        ids, truth = source_truth(r)
        return dict(image=pixels, truth=truth, family_id=r['scene_family_id'], object_ids=ids)


def collate(samples):
    return dict(images=torch.stack([s['image'] for s in samples]), truths=[s['truth'] for s in samples],
                family_ids=[s['family_id'] for s in samples], object_ids=[s['object_ids'] for s in samples])
