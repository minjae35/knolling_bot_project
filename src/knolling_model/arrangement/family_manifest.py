"""Separate future-family pilot contract; never relaxes the 76/35 verification gate.

No data generation or training entrypoint. A pinned manifest fixes splits BEFORE
annotation creation. Callers select checkpoints on validation; test is final-only.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest, scene_from_dict, instruction_from_dict
from knolling_dataset.teacher_v3 import provenance, solve, validate_layout, POLICY
from knolling_dataset.english import audit_english_records
from .dataset_v3 import supervision_metadata, verify_contrast
from .data import recovery_supervision

SCHEMA='recovery_family_pilot_v1'
FACTORS={'axis_alignment','region','selection','gap_m','sort_direction','group_by'}
PARTITIONS=('train','validation','test')


def source_hash(record):
    # IDs/order/selection bookkeeping cannot disguise duplicate observations.
    objects=[dict(color=o['color'],pose=o['source_pose'],
                  asset={k:o['asset'][k] for k in ('a_m','b_m','h_m','shape','footprint')})
             for o in record['source']['objects']]
    return digest(sorted(objects,key=digest))


def coverage(records,pairs):
    by_id={r['sample_id']:r for r in records}
    families=defaultdict(set)
    for p in pairs:families[p['factor']].add(by_id[p['sample_ids'][0]]['scene_family_id'])
    return dict(records=len(records),families=len({r['scene_family_id'] for r in records}),
        modes=dict(Counter(r['mode'] for r in records)),
        object_counts=dict(Counter(str(len(r['source']['objects'])) for r in records)),
        pairs=dict(Counter(p['factor'] for p in pairs)),
        factor_families={k:len(v) for k,v in families.items()})


def validate_manifest(manifest,records,pairs):
    if manifest.get('schema')!=SCHEMA or manifest.get('teacher')!=provenance():
        raise ValueError('pilot schema/teacher mismatch')
    families=manifest['families'];by_id={r['sample_id']:r for r in records}
    if len(by_id)!=len(records) or len({p['pair_id'] for p in pairs})!=len(pairs):
        raise ValueError('duplicate pilot sample/pair')
    if set(families)!={r['scene_family_id'] for r in records}:raise ValueError('missing family coverage')
    totals=Counter((v['origin'],v['partition']) for v in families.values())
    if totals!=Counter({('verification','train'):25,('new','train'):40,('new','validation'):10,('new','test'):10}):
        raise ValueError('pilot requires 25 legacy train and new 40/10/10 families')
    sources={};rgbs={};observables=set()
    for r in records:
        f=families[r['scene_family_id']];rgb=r['source_render_dir']+'/rgb.png';rgb_hash=r['file_hashes'].get(rgb)
        # Scene/family IDs are bookkeeping, not observable geometry.
        source_digest=source_hash(r)
        if (r['split']!=f['partition'] or source_digest!=f['source_sha256'] or rgb_hash!=f['rgb_sha256']
                or r.get('teacher_policy')!=POLICY):raise ValueError('pilot source/split mismatch')
        for table,key in ((sources,source_digest),(rgbs,rgb_hash)):
            if key in table and table[key]!=r['scene_family_id']:raise ValueError('source/RGB duplicate across families')
            table[key]=r['scene_family_id']
        observable=(rgb_hash,r['instruction']['text'])
        if observable in observables:raise ValueError('duplicate or ambiguous observable input')
        observables.add(observable)
    for pair in pairs:
        if pair['factor'] not in FACTORS or len(pair['sample_ids'])!=2 or len(set(pair['sample_ids']))!=2:
            raise ValueError('invalid pilot contrast')
        a,b=[by_id[s] for s in pair['sample_ids']]
        if a['split']!=b['split'] or a['scene_family_id']!=b['scene_family_id']:
            raise ValueError('pair crosses family/partition')
        verify_contrast(a,b,pair['factor'])
    for split in PARTITIONS:
        subset=[r for r in records if r['split']==split];ids={r['sample_id'] for r in subset}
        contrasts=[p for p in pairs if set(p['sample_ids'])<=ids]
        actual=coverage(subset,contrasts)
        if (actual!=manifest['expected'][split] or set(actual['modes'])!={'full','partial','no_op'}
                or set(actual['object_counts'])!={'2','3','4'} or set(actual['pairs'])!=FACTORS
                or (split!='train' and min(actual['factor_families'].values(),default=0)<5)):
            raise ValueError('pilot expected counts/coverage quota mismatch')


class FamilyDataset(torch.utils.data.Dataset):
    """Immutable split view with replay-validated labels; GoalDataset can wrap it."""
    def __init__(self,root,manifest_path,expected_sha256,split):
        if split not in PARTITIONS:raise ValueError('unknown pilot partition')
        self.root=Path(root).resolve()
        if file_hash(manifest_path)!=expected_sha256:raise ValueError('pilot manifest hash mismatch')
        manifest=json.loads(Path(manifest_path).read_text())
        annotation=self._path(manifest['annotations']['path'])
        if file_hash(annotation)!=manifest['annotations']['sha256']:raise ValueError('pilot annotation hash mismatch')
        bundle=json.loads(annotation.read_text());records,pairs=bundle['records'],bundle['pairs']
        validate_manifest(manifest,records,pairs)
        if not audit_english_records(records)['valid']:raise ValueError('pilot language contract mismatch')
        for r in records:
            rgb=r['source_render_dir']+'/rgb.png'
            if file_hash(self._path(rgb))!=r['file_hashes'][rgb]:raise ValueError('pilot RGB hash mismatch')
            source,target,ins=scene_from_dict(r['source']),scene_from_dict(r['target']),instruction_from_dict(r['instruction'])
            replay=solve(source,ins)
            if (replay.status!='ok' or replay.mode!=r['mode'] or digest(asdict(replay.target))!=digest(r['target'])
                    or not validate_layout(source,target,ins,r['mode']).valid or r['supervision_v3']!=supervision_metadata(r)):
                raise ValueError('pilot Teacher replay mismatch')
        self.records=[r for r in records if r['split']==split];ids={r['sample_id'] for r in self.records}
        self.pairs=[p for p in pairs if set(p['sample_ids'])<=ids]
        self.manifest,self.split=manifest,split
        self.fingerprint=digest([SCHEMA,expected_sha256,split])

    def _path(self,path):
        p=(self.root/path).resolve()
        if not p.is_relative_to(self.root):raise ValueError('pilot path escapes root')
        return p

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(self._path(r['source_render_dir']+'/rgb.png')) as im:
            if im.mode!='RGB' or im.size!=(576,384):raise ValueError('unexpected pilot RGB')
            pixels=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/255
        ids,truth=recovery_supervision(r)
        return dict(image=pixels,text=r['instruction']['text'],supervision=truth,
                    metadata=dict(sample_id=r['sample_id'],object_ids=ids,record=r))


def breakdown(rows,records,checks):
    by_id={r['sample_id']:r for r in records}
    def summarize(key):
        groups=defaultdict(list)
        for r in rows:groups[str(key(by_id[r['sample_id']]))].append(r)
        return {k:dict(total=len(v),success=sum(bool(r['task_success']) for r in v),
                       rate=sum(bool(r['task_success']) for r in v)/len(v)) for k,v in groups.items()}
    factors={}
    for f in sorted(FACTORS):
        subset=[p for p in checks if p['factor']==f]
        factors[f]=dict(total=len(subset),success=sum(p['passed'] for p in subset),
            rate=sum(p['passed'] for p in subset)/len(subset) if subset else None)
    return dict(mode=summarize(lambda r:r['mode']),objects=summarize(lambda r:len(r['source']['objects'])),
                family=summarize(lambda r:r['scene_family_id']),factor=factors)


@torch.no_grad()
def evaluate_family_pilot(model,batches,*,dataset):
    """Separate descriptive held-out gate. It is NOT a production StageEngine gate."""
    from .evaluation_v4 import evaluate_v4
    if not isinstance(dataset,FamilyDataset):raise ValueError('validated pilot dataset required')
    metrics,rows=evaluate_v4(model,batches,stage='oracle_layout',
        expected_ids=[r['sample_id'] for r in dataset.records],pairs=dataset.pairs)
    report=breakdown(rows,dataset.records,metrics['contrasts']['checks'])
    goals=metrics['grounding'];complete=metrics['complete_set']
    goal_ok=complete and all(r['passed'] for r in goals['rows']) and all(p['passed'] for p in goals['contrasts'])
    goal_ok=goal_ok and len(goals['contrasts'])==len(dataset.pairs)
    collisions=sum(any('collision' in str(e) or 'overlap' in str(e) for e in r['errors']) for r in rows)
    passed=(complete and goal_ok and metrics['validity']['success']>=.95
        and all(v['rate']>=.9 for v in report['mode'].values())
        and all(v['rate'] is not None and v['rate']>=.9 for v in report['factor'].values())
        and collisions==0 and metrics['validity']['keep_violations']==0)
    return dict(schema='recovery_family_metrics_v1',partition=dataset.split,passed=bool(passed),
        dataset_fingerprint=dataset.fingerprint,grounding_passed=bool(goal_ok),collisions=collisions,
        metrics=metrics,breakdown=report,
        limitation='Family-level descriptive results; 10 test families do not establish deployment generalization.'),rows
