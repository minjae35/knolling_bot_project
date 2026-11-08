"""Versioned, offline counterfactual labels; reuses unchanged parent RGB files."""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from knolling_dataset.builder import _record, file_hash
from knolling_dataset.english import english_instruction, audit_english_records
from knolling_dataset.schema import scene_from_dict, instruction_from_dict, digest, canonical_json
from knolling_dataset.teacher_continuous import solve, POLICY
from knolling_dataset.validator import validate_layout
from knolling_model.evaluation import semantic_errors
from .data import recovery_supervision, unique_source_indices

SCHEMA = 'recovery_counterfactuals_v2'


def input_target_contract(records):
    """Identical RGB/text cannot acquire conflicting labels from two Teachers."""
    known = {}
    for r in records:
        key = (r['scene_family_id'], r['instruction']['text'])
        poses = {o['object_id']: o['target_pose'] for o in r['target']['objects']}
        signature = digest(poses)
        if key in known and known[key] != signature:
            raise ValueError('identical RGB/text has conflicting Teacher targets')
        known[key] = signature
    return known


def verify_pair(a, b):
    """Verify actual semantics/geometry rather than trusting pair metadata."""
    if (a['source'] != b['source'] or a['scene_family_id'] != b['scene_family_id']
            or a['split'] != b['split'] or a['source_render_dir'] != b['source_render_dir']
            or a['file_hashes'] != b['file_hashes']):
        raise ValueError('counterfactual must preserve source, RGB and split')
    ca, cb = a['counterfactual'], b['counterfactual']
    if ca != cb:
        raise ValueError('pair metadata disagreement')
    factor = ca['factor']
    changed = {k for k in a['instruction'] if a['instruction'][k] != b['instruction'][k]}-{'text', 'language'}
    wanted = {'selected_ids', 'locked_ids'} if factor == 'selection' else {factor}
    if factor not in ('axis_alignment', 'region', 'selection') or changed != wanted:
        raise ValueError('not a pure-factor contrast')
    if not audit_english_records([a, b])['valid']:
        raise ValueError('English text disagrees with semantics')
    for r in (a, b):
        source, target = scene_from_dict(r['source']), scene_from_dict(r['target'])
        ins = instruction_from_dict(r['instruction'])
        if r.get('teacher_policy') != POLICY or not validate_layout(source, target, ins, r['mode']).valid:
            raise ValueError('invalid counterfactual target or provenance')
    # Both targets must satisfy their own command and reject the opposite one.
    if not (semantic_errors(list(scene_from_dict(a['target']).objects), b)
            and semantic_errors(list(scene_from_dict(b['target']).objects), a)):
        raise ValueError('contrast has no bidirectional semantic effect')


def build_bundle(parent, destination, *, pair_limit=32, min_axis_pairs=8):
    """New directory only. Parent is a validated S0Dataset; no rendering or ML."""
    if pair_limit < min_axis_pairs+2 or min_axis_pairs < 1:
        raise ValueError('invalid counterfactual coverage request')
    destination = Path(destination)
    if destination.exists() or destination.resolve().is_relative_to(parent.root):
        raise ValueError('output must be a new directory outside the parent dataset')
    families = [parent.records[i] for i in unique_source_indices(parent.records)]
    candidates = {'axis_alignment': [], 'region': [], 'selection': []}
    for r in families:
        source, ins = scene_from_dict(r['source']), instruction_from_dict(r['instruction'])
        # For keep-all source families construct observable full-selection commands.
        if ins.keep_all:
            ins = replace(ins, keep_all=False, selected_ids=tuple(o.object_id for o in source.objects), locked_ids=())
        for region in ('full', 'left', 'right'):
            base = replace(ins, region=region)
            candidates['axis_alignment'].append((r, replace(base, axis_alignment='horizontal'),
                                                  replace(base, axis_alignment='vertical')))
        for region in ('left', 'right'):
            candidates['region'].append((r, replace(ins, region='full'), replace(ins, region=region)))
        all_ids = tuple(o.object_id for o in source.objects)
        for color in ('red', 'green', 'blue'):
            selected = tuple(o.object_id for o in source.objects if o.color == color)
            if selected and len(selected) != len(all_ids):
                candidates['selection'].append((r, replace(ins, selected_ids=all_ids, locked_ids=()),
                    replace(ins, selected_ids=selected, locked_ids=tuple(i for i in all_ids if i not in selected))))
    records, rejected, counts, seen = [], [], Counter(), set()
    def accept(factor, candidate):
        r, ia, ib = candidate
        source = scene_from_dict(r['source'])
        instructions = [english_instruction(source, i) for i in (ia, ib)]
        key = digest([r['scene_family_id'], factor, [asdict(i) for i in instructions]])
        if key in seen:
            return False
        seen.add(key)
        pair = []
        for side, ins in enumerate(instructions):
            result = solve(source, ins)
            if result.status != 'ok':
                rejected.append(dict(pair_id=key, factor=factor, reason=result.reason))
                return False
            record = _record(source, ins, result, f'cf2_{key[:20]}_{side}', 'recovery_counterfactual')
            rgb = r['source_render_dir']+'/rgb.png'
            record.update(source_render_dir=r['source_render_dir'], file_hashes={rgb: r['file_hashes'][rgb]},
                          teacher_policy=POLICY, teacher_audit=result.audit,
                          counterfactual=dict(pair_id=key, factor=factor))
            pair.append(record)
        try:
            verify_pair(*pair)
            input_target_contract(list(parent.records)+records+pair)
        except ValueError as exc:
            rejected.append(dict(pair_id=key, factor=factor, reason=str(exc)))
            return False
        records.extend(pair)
        counts[factor] += 1
        return True
    # Reserve axis coverage first, then round-robin other independent factors.
    queues = {k: iter(v) for k, v in candidates.items()}
    for candidate in queues['axis_alignment']:
        accept('axis_alignment', candidate)
        if counts['axis_alignment'] >= min_axis_pairs:
            break
    for factor in ('region', 'selection'):
        for candidate in queues[factor]:
            if accept(factor, candidate):
                break
    active = list(queues)
    while active and sum(counts.values()) < pair_limit:
        for factor in active[:]:
            try:
                accept(factor, next(queues[factor]))
            except StopIteration:
                active.remove(factor)
            if sum(counts.values()) >= pair_limit:
                break
    if sum(counts.values()) != pair_limit or counts['axis_alignment'] < min_axis_pairs or not all(counts[k] for k in candidates):
        raise ValueError(f'insufficient validated contrasts: {dict(counts)}')
    bundle = dict(schema=SCHEMA, parent_manifest_sha256=file_hash(parent.root/'manifest.jsonl'),
                  teacher_policy=POLICY, counts=dict(counts), records=records, rejected=rejected)
    payload = canonical_json(bundle)+'\n'
    destination.mkdir(parents=True, exist_ok=False)
    with (destination/'counterfactuals.json').open('x') as stream:
        stream.write(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    (destination/'bundle.sha256').write_text(sha+'\n')
    return dict(path=str(destination/'counterfactuals.json'), sha256=sha, pairs=dict(counts), samples=len(records))


class RecoverySuite(torch.utils.data.Dataset):
    """Legacy S0 + pinned contrasts, each with independently validated provenance."""
    def __init__(self, parent, bundle_path, expected_sha256, *, replay_teacher=True):
        path = Path(bundle_path)
        if file_hash(path) != expected_sha256:
            raise ValueError('counterfactual bundle hash mismatch')
        bundle = json.loads(path.read_text())
        if bundle['schema'] != SCHEMA or bundle['teacher_policy'] != POLICY:
            raise ValueError('unsupported recovery data contract')
        if bundle['parent_manifest_sha256'] != file_hash(parent.root/'manifest.jsonl'):
            raise ValueError('wrong parent dataset')
        self.parent, self.root = parent, parent.root
        # Parent records remain immutable and retain their original Teacher policy.
        self.records = list(parent.records)+bundle['records']
        input_target_contract(self.records)
        self.fingerprint = digest([bundle['parent_manifest_sha256'], parent.selection, expected_sha256])
        by_family = {r['scene_family_id']: r for r in parent.records}
        groups = {}
        for r in bundle['records']:
            base = by_family.get(r['scene_family_id'])
            if base is None or r['source'] != base['source'] or r['split'] != base['split']:
                raise ValueError('counterfactual changed parent source or family split')
            if r['source_render_dir'] != base['source_render_dir']:
                raise ValueError('counterfactual changed source RGB path')
            rgb = r['source_render_dir']+'/rgb.png'
            if r['file_hashes'] != {rgb: base['file_hashes'][rgb]} or file_hash(parent._path(rgb)) != r['file_hashes'][rgb]:
                raise ValueError('counterfactual RGB hash mismatch')
            if replay_teacher:
                result = solve(scene_from_dict(r['source']), instruction_from_dict(r['instruction']))
                if result.status != 'ok' or digest(asdict(result.target)) != digest(r['target']):
                    raise ValueError('counterfactual Teacher replay disagrees')
            groups.setdefault(r['counterfactual']['pair_id'], []).append(r)
        for pair in groups.values():
            if len(pair) != 2:
                raise ValueError('incomplete or duplicated counterfactual pair')
            verify_pair(*pair)
        counts = Counter(pair[0]['counterfactual']['factor'] for pair in groups.values())
        if dict(counts) != bundle['counts'] or counts['axis_alignment'] < 1:
            raise ValueError('counterfactual coverage metadata disagrees')
        if len({r['sample_id'] for r in self.records}) != len(self.records):
            raise ValueError('duplicate sample ID')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.parent._path(record['source_render_dir']+'/rgb.png')) as image:
            if image.mode != 'RGB' or image.size != (576, 384):
                raise ValueError('unexpected source image')
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()/255
        ids, labels = recovery_supervision(record)
        return dict(image=pixels, text=record['instruction']['text'], supervision=labels,
                    metadata=dict(sample_id=record['sample_id'], object_ids=ids, record=record))


def main():
    import argparse
    from knolling_model.data import S0Dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', default='artifacts/pilot_v1_3_en')
    parser.add_argument('--selection', default='configs/s0_samples.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs', type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(build_bundle(S0Dataset(args.parent, args.selection), args.output, pair_limit=args.pairs), indent=2))


if __name__ == '__main__':
    main()
