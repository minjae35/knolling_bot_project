"""Offline Phase 1 size variants. No model/optimizer or image synthesis path."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from knolling_dataset.assets import make_asset, asset_split, geometry_fingerprint
from knolling_dataset.builder import file_hash
from knolling_dataset.schema import scene_from_dict, instruction_from_dict, digest
from knolling_dataset.teacher_v3 import solve, validate_layout, provenance, POLICY
from knolling_dataset.validator import validate_source, inferred_rows, polygon
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.arrangement.dataset_v3 import supervision_metadata, verify_contrast

SCHEMA = 'phase1_oracle_size_variants_v1'


class ValidationFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationFailure(message)


def check_record(record, parent, variant, forbidden, replay=True):
    """Independent exhaustive validation of a finished annotation, not sampling."""
    source = scene_from_dict(record['source'])
    target = scene_from_dict(record['target'])
    ins = instruction_from_dict(record['instruction'])
    original = scene_from_dict(parent['source'])
    require(record['split'] == source.split == target.split == parent['split'] == 'train', 'split')
    require(record['instruction'] == parent['instruction'], 'changed instruction')
    require(record['mode'] == parent['mode'] == 'full', 'mode')
    require(record['teacher_policy'] == POLICY, 'teacher policy')
    require(len(source.objects) == len(original.objects) and 2 <= len(source.objects) <= 4, 'count')
    require(validate_source(source).valid, 'invalid source geometry')
    require(validate_layout(source, target, ins, 'full').valid, 'invalid target/annotation')
    require(len(inferred_rows(target.objects)) == 1, 'outside Phase 1')
    require(record['supervision_v3'] == supervision_metadata(record), 'supervision metadata')
    if variant == 0:
        require(record == parent, 'original changed')
    else:
        require('source_render_dir' not in record and record.get('file_hashes') == {}, 'stale RGB reference')
        require(record.get('render_status') == 'unrendered_oracle_only', 'missing RGB prohibition')
        expected_family = parent['scene_family_id'] + f'__size{variant:02d}'
        require(record['scene_family_id'] == source.family_id == expected_family, 'variant family identity')
        require(record['sample_id'] == parent['sample_id'] + f'__size{variant:02d}', 'variant sample identity')
        require(record['parent_scene_family_id'] == parent['scene_family_id'], 'parent lineage')
    for obj, old in zip(source.objects, original.objects):
        require((obj.object_id, obj.color, obj.source_pose, obj.locked, obj.move_label) ==
                (old.object_id, old.color, old.source_pose, old.locked, old.move_label), 'changed non-size input')
        a = obj.asset
        require(a.shape == old.asset.shape and a.h_m == old.asset.h_m, 'changed shape class/height')
        mm = [round(v * 1000) for v in (a.a_m, a.b_m, a.h_m)]
        regenerated = make_asset(a.shape, *mm)
        require(digest(asdict(a)) == digest(asdict(regenerated)), 'asset dimensions/footprint/mass/ID mismatch')
        require(a.family_id == 'af_' + geometry_fingerprint(a)[:20], 'geometry fingerprint')
        require(asset_split(a.family_id) == 'train' and a.family_id not in forbidden, 'asset split leakage')
        if variant:
            require((a.a_m, a.b_m) != (old.asset.a_m, old.asset.b_m), 'unchanged variant dimensions')
    ids, tensors = recovery_supervision(record)
    require(ids == sorted(o.object_id for o in source.objects), 'loader identity')
    require(all(torch.isfinite(x).all() for x in tensors.values()), 'nonfinite supervision')
    require(not any(record['supervision_v3']['copy_required'].values()), 'unexpected Keep in Full')
    if replay:
        repeated = solve(source, ins)
        require(repeated.status == 'ok' and repeated.mode == 'full', 'teacher replay status')
        require(digest(asdict(repeated.target)) == digest(record['target']), 'teacher replay target')
        require(repeated.rows == record['supervision_v3']['rows'], 'teacher replay order')


def draw_asset(old, rng, forbidden, existing_train, limit=512):
    for _ in range(limit):
        a, b = rng.randint(40, 80), rng.randint(25, 50)
        if a / b < 1.3 or (a, b) == (round(old.a_m * 1000), round(old.b_m * 1000)):
            continue
        new = make_asset(old.shape, a, b, round(old.h_m * 1000))
        if (asset_split(new.family_id) == 'train' and new.family_id not in forbidden
                and new.family_id not in existing_train):
            return new
    raise ValidationFailure('bounded Train asset sampling exhausted')


def generate_family(job):
    """Reject unaccepted proposals; never ignore a finished-record validation error."""
    records, pairs, forbidden, existing_train, seed, variants, limit = job
    family = records[0]['scene_family_id']
    source = scene_from_dict(records[0]['source'])
    require(all(r['source'] == records[0]['source'] for r in records), 'parent family source mismatch')
    rng = random.Random(int(digest([SCHEMA, seed, family])[:16], 16))
    entries = [dict(parent_sample_id=r['sample_id'], family_split_group=family,
                    variant_index=0, input_mode='oracle_only', record=deepcopy(r)) for r in records]
    signatures = set()
    rejected = Counter()
    attempts = 0
    for variant in range(1, variants):
        for _ in range(limit):
            attempts += 1
            objects = tuple(replace(o, asset=draw_asset(o.asset, rng, forbidden, existing_train)) for o in source.objects)
            changed = replace(source, family_id=family + f'__size{variant:02d}', objects=objects)
            if not validate_source(changed).valid:
                rejected['source_geometry'] += 1
                continue
            appearance = {(o.asset.shape, o.asset.a_m, o.asset.b_m, o.color) for o in objects}
            signature = digest([asdict(o.asset) for o in objects])
            if len(appearance) != len(objects) or signature in signatures:
                rejected['duplicate_appearance_or_variant'] += 1
                continue
            candidate = []
            for parent in records:
                result = solve(changed, instruction_from_dict(parent['instruction']))
                if result.status != 'ok' or result.mode != 'full':
                    rejected['teacher_status'] += 1
                    break
                if len(inferred_rows(result.target.objects)) != 1:
                    rejected['not_single_row'] += 1
                    break
                r = deepcopy(parent)
                r.pop('source_render_dir', None)
                r.update(sample_id=parent['sample_id'] + f'__size{variant:02d}',
                         scene_family_id=changed.family_id, parent_scene_family_id=family,
                         source=asdict(changed), target=asdict(result.target), teacher_audit=result.audit,
                         file_hashes={}, render_status='unrendered_oracle_only')
                r['supervision_v3'] = supervision_metadata(r)
                candidate.append(r)
            if len(candidate) != len(records):
                continue
            by_parent = {r['sample_id']: new for r, new in zip(records, candidate)}
            try:
                for pair in pairs:
                    verify_contrast(*(by_parent[s] for s in pair['sample_ids']), pair['factor'])
            except ValueError:
                rejected['contrast_degenerate'] += 1
                continue
            # Any error here is fatal, never a sampling rejection.
            for r, parent in zip(candidate, records):
                check_record(r, parent, variant, forbidden, replay=False)
                entries.append(dict(parent_sample_id=parent['sample_id'], family_split_group=family,
                                    variant_index=variant, input_mode='oracle_only', record=r))
            signatures.add(signature)
            break
        else:
            raise ValidationFailure(f'{family}: variant {variant} exhausted {limit} proposals')
    return dict(family=family, entries=entries, proposals=attempts, rejected=dict(rejected))


def distribution(records):
    sizes = []
    assets, families = set(), set()
    cells, shapes = Counter(), Counter()
    for r in records:
        ins = r['instruction']
        cells[(len(r['source']['objects']), ins['axis_alignment'], round(ins['gap_m'] * 1000), ins['sort_direction'])] += 1
        for obj in r['source']['objects']:
            a = obj['asset']; assets.add(a['asset_id']); families.add(a['family_id']); shapes[a['shape']] += 1
            sizes.append([a['a_m'] * 1000, a['b_m'] * 1000])
    x = np.asarray(sizes)
    return dict(samples=len(records), asset_ids=len(assets), asset_families=len(families),
                unique_size_pairs=len(set(map(tuple, sizes))), shape_counts=dict(shapes),
                cells=[dict(count=k[0], axis=k[1], gap_mm=k[2], sort=k[3], samples=v) for k,v in sorted(cells.items())],
                size_mm={key:dict(minimum=float(x[:,i].min()), maximum=float(x[:,i].max()),
                    mean=float(x[:,i].mean()), std=float(x[:,i].std()),
                    quantiles=np.percentile(x[:,i], [5,25,50,75,95]).tolist()) for i,key in enumerate(('major','minor'))})


class OracleSizeVariants:
    """No RGB path, even for variant zero. Parent-first sampling preserves exposure."""
    def __init__(self, root):
        self.root = Path(root)
        result = json.loads((self.root / 'result.json').read_text())
        require(result['status'] == 'PASS', 'unvalidated dataset')
        require(file_hash(self.root / 'annotations.json') == result['annotation_sha256'], 'annotation integrity')
        bundle = json.loads((self.root / 'annotations.json').read_text())
        require(bundle['schema'] == SCHEMA and bundle['teacher'] == provenance(), 'dataset/teacher contract')
        self.entries = bundle['entries']
        self.parent_ids = bundle['parent_ids']
        self.variants = bundle['variants_per_parent']
        self.lookup = {(e['parent_sample_id'], e['variant_index']): e['record'] for e in self.entries}
        require(len(self.lookup) == len(self.entries) == len(self.parent_ids)*self.variants, 'coverage')

    def __len__(self):
        return len(self.parent_ids)

    def record_for_visit(self, parent_index, visit):
        require(visit >= 0, 'negative exposure')
        return deepcopy(self.lookup[self.parent_ids[parent_index], visit % self.variants])

    def get_rgb(self, *_):
        raise ValueError('Oracle-only size variants have no compatible RGB; rendering is required')

    def __getitem__(self, parent_index):
        r = self.record_for_visit(parent_index, 0)
        ids, truth = recovery_supervision(r)
        return dict(text=r['instruction']['text'], supervision=truth,
                    metadata=dict(sample_id=r['sample_id'], object_ids=ids, record=r))
