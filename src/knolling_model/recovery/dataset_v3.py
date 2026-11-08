"""Immutable v3 annotation overlay over the 25 existing verification families."""
from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from knolling_dataset.builder import _record, file_hash
from knolling_dataset.english import english_instruction, audit_english_records
from knolling_dataset.schema import scene_from_dict, instruction_from_dict, digest, canonical_json, angle_error
from knolling_dataset.teacher_v3 import solve, provenance, POLICY, validate_layout
from knolling_dataset.validator import inferred_rows
from knolling_model.evaluation import semantic_errors
from .data import recovery_supervision, unique_source_indices

SCHEMA = 'recovery_dataset_v3'


def supervision_metadata(record):
    source = {o.object_id:o for o in scene_from_dict(record['source']).objects}
    target = {o.object_id:o for o in scene_from_dict(record['target']).objects}
    ids = sorted(source)
    copy = {i:source[i].source_pose == target[i].target_pose for i in ids}
    task_move = {i:(math.hypot(target[i].target_pose.x_m-source[i].source_pose.x_m,
                            target[i].target_pose.y_m-source[i].source_pose.y_m) > .003 or
                    angle_error(target[i].target_pose.theta_rad,source[i].source_pose.theta_rad) > np.pi/60) for i in ids}
    selected = record['instruction']['selected_ids'] or ids
    center = [sum(getattr(target[i].target_pose,k) for i in selected)/len(selected) for k in ('x_m','y_m')]
    rows = inferred_rows([target[i] for i in record['instruction']['selected_ids']])
    row_ids = [[o.object_id for o in row] for row in rows]
    return dict(copy_required=copy, move_exact={i:not v for i,v in copy.items()}, move_task=task_move,
                structure_center_m=center, rows=row_ids,
                horizontal_neighbors=[[a,b] for row in row_ids for a,b in zip(row,row[1:])],
                vertical_neighbors=[[a,b] for a,b in zip(row_ids,row_ids[1:])])


def verify_contrast(a, b, factor):
    if a['source'] != b['source'] or a['file_hashes'] != b['file_hashes'] or a['split'] != b['split']:
        raise ValueError('contrast changed source/RGB/split')
    changed = {k for k in a['instruction'] if a['instruction'][k] != b['instruction'][k]}-{'text','language'}
    wanted = {'selected_ids','locked_ids'} if factor == 'selection' else {factor}
    if factor not in ('axis_alignment','region','selection','gap_m','sort_direction','group_by') or changed != wanted:
        raise ValueError('contrast is not a pure supported factor')
    if not (semantic_errors(list(scene_from_dict(a['target']).objects),b) and
            semantic_errors(list(scene_from_dict(b['target']).objects),a)):
        raise ValueError('contrast has no bidirectional effect')


def build_bundle(parent, old_suite, destination):
    destination = Path(destination)
    if destination.exists() or destination.resolve().is_relative_to(parent.root):
        raise ValueError('v3 requires a new output directory outside the parent')
    if len({r['scene_family_id'] for r in old_suite.records}) != 25:
        raise ValueError('verification release must use exactly the existing 25 families')
    records, aliases, pairs, rejected = {}, {}, [], []
    def add(old, instruction=None):
        scene = scene_from_dict(old['source'])
        ins = english_instruction(scene,instruction or instruction_from_dict(old['instruction']))
        result = solve(scene,ins)
        if result.status != 'ok':
            raise ValueError(f'{result.status}: {result.reason}')
        rgb = old['source_render_dir']+'/rgb.png'
        key = digest([old['file_hashes'][rgb],asdict(ins)])
        sid = 'v3_'+key[:24]
        record = _record(scene,ins,result,sid,'v3_verification')
        record.update(source_render_dir=old['source_render_dir'],file_hashes={rgb:old['file_hashes'][rgb]},
                      teacher_policy=POLICY,teacher_audit=result.audit)
        record['supervision_v3'] = supervision_metadata(record)
        if sid in records and records[sid] != record:
            raise ValueError('same observable input has inconsistent source/target')
        records[sid] = record
        return sid
    for r in old_suite.records:
        aliases[r['sample_id']] = add(r)
    groups = {}
    for r in old_suite.records:
        if 'counterfactual' in r:
            groups.setdefault(r['counterfactual']['pair_id'],[]).append(r)
    for pair in groups.values():
        a,b = [aliases[r['sample_id']] for r in pair]
        factor = pair[0]['counterfactual']['factor']
        verify_contrast(records[a],records[b],factor)
        pairs.append(dict(pair_id=digest([factor,a,b]),factor=factor,sample_ids=[a,b]))
    # Minimal extra contrasts, using only existing RGB, to cover the three other factors.
    for factor, values in [('gap_m',(.015,.030)),('sort_direction',('asc','desc')),('group_by',('none','color'))]:
        for index in unique_source_indices(parent.records):
            r = parent.records[index]
            scene, ins = scene_from_dict(r['source']),instruction_from_dict(r['instruction'])
            base = replace(ins,keep_all=False,selected_ids=tuple(o.object_id for o in scene.objects),locked_ids=())
            before = set(records)
            try:
                a,b = [add(r,replace(base,**{factor:v})) for v in values]
                verify_contrast(records[a],records[b],factor)
            except ValueError as exc:
                for sid in set(records)-before:del records[sid]
                rejected.append(dict(family=r['scene_family_id'],factor=factor,reason=str(exc)))
                continue
            pairs.append(dict(pair_id=digest([factor,a,b]),factor=factor,sample_ids=[a,b]))
            break
        else:
            raise ValueError(f'no verified {factor} contrast in existing families')
    bundle = dict(schema=SCHEMA,teacher=provenance(),parent_manifest_sha256=file_hash(parent.root/'manifest.jsonl'),
                  parent_selection=parent.selection,records=list(records.values()),pairs=pairs,aliases=aliases,
                  rejected=rejected,scope='train_verification_only_25_families',
                  counts=dict(records=len(records),families=25,pairs=dict(Counter(p['factor'] for p in pairs))))
    payload = canonical_json(bundle)+'\n'
    destination.mkdir(parents=True,exist_ok=False)
    path = destination/'annotations.json'
    with path.open('x') as stream:stream.write(payload)
    return dict(path=str(path),sha256=file_hash(path),counts=bundle['counts'])


class RecoveryDatasetV3(torch.utils.data.Dataset):
    def __init__(self, parent, path, expected_sha256):
        path = Path(path)
        if file_hash(path) != expected_sha256:
            raise ValueError('v3 annotation hash mismatch')
        bundle = json.loads(path.read_text())
        if bundle.get('schema') != SCHEMA or bundle.get('teacher') != provenance():
            raise ValueError('v3 teacher code/config contract mismatch')
        if (bundle['parent_manifest_sha256'] != file_hash(parent.root/'manifest.jsonl') or
                bundle['parent_selection'] != parent.selection):
            raise ValueError('v3 parent contract mismatch')
        self.parent, self.root = parent, parent.root
        self.records, self.pairs = bundle['records'], bundle['pairs']
        self.fingerprint = digest([SCHEMA,expected_sha256,parent.selection,bundle['teacher']])
        parents = {r['scene_family_id']:r for r in parent.records}
        seen, inputs = {}, {}
        for r in self.records:
            base = parents.get(r['scene_family_id'])
            if (base is None or r['source'] != base['source'] or r['split'] != base['split'] or
                    r['source_render_dir'] != base['source_render_dir'] or r['split'] != 'train'):
                raise ValueError('v3 source/family/split changed')
            rgb = r['source_render_dir']+'/rgb.png'
            if r['file_hashes'] != {rgb:base['file_hashes'][rgb]} or file_hash(parent._path(rgb)) != base['file_hashes'][rgb]:
                raise ValueError('v3 RGB hash mismatch')
            if r['sample_id'] in seen or r['teacher_policy'] != POLICY:
                raise ValueError('v3 duplicate sample or mixed teacher')
            source, target, ins = scene_from_dict(r['source']),scene_from_dict(r['target']),instruction_from_dict(r['instruction'])
            replay = solve(source,ins)
            if (replay.status != 'ok' or replay.mode != r['mode'] or digest(asdict(replay.target)) != digest(r['target'])
                    or not validate_layout(source,target,ins,r['mode']).valid
                    or r['supervision_v3'] != supervision_metadata(r)):
                raise ValueError('v3 target/supervision replay mismatch')
            key = (base['file_hashes'][rgb],ins.text)
            if key in inputs:
                raise ValueError('v3 duplicate observable input')
            inputs[key] = r['target']
            seen[r['sample_id']] = r
        if len({r['scene_family_id'] for r in self.records}) != 25 or not audit_english_records(self.records)['valid']:
            raise ValueError('v3 family coverage or language contract mismatch')
        if len({p['pair_id'] for p in self.pairs}) != len(self.pairs):
            raise ValueError('v3 duplicate contrast')
        for pair in self.pairs:
            a,b = [seen[sid] for sid in pair['sample_ids']]
            verify_contrast(a,b,pair['factor'])
        actual = dict(records=len(self.records),families=25,pairs=dict(Counter(p['factor'] for p in self.pairs)))
        if bundle['counts'] != actual or len(actual['pairs']) != 6:
            raise ValueError('v3 coverage mismatch')

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r = self.records[index]
        with Image.open(self.parent._path(r['source_render_dir']+'/rgb.png')) as image:
            if image.mode != 'RGB' or image.size != (576,384):raise ValueError('unexpected source image')
            pixels = torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255
        ids,truth = recovery_supervision(r)
        return dict(image=pixels,text=r['instruction']['text'],supervision=truth,
                    metadata=dict(sample_id=r['sample_id'],object_ids=ids,record=r))


def main():
    import argparse
    from knolling_model.data import S0Dataset
    from .counterfactuals import RecoverySuite
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/recovery_gpu_initial.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    c=json.loads(Path(args.config).read_text())
    parent=S0Dataset(c['dataset'],c['selection'])
    old=RecoverySuite(parent,c['counterfactuals']['path'],c['counterfactuals']['sha256'])
    result=build_bundle(parent,old,args.output)
    RecoveryDatasetV3(parent,result['path'],result['sha256'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
