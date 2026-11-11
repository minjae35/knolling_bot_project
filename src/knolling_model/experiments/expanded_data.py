"""CPU generation of 4,000 immutable source families, five commands each."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from knolling_dataset.assets import build_registry,registry_dict
from knolling_dataset.scenes import generate_scene,SamplingFailure
from knolling_dataset.instructions import generate_instruction
from knolling_dataset.english import english_instruction,audit_english_records
from knolling_dataset.schema import digest,scene_from_dict,instruction_from_dict
from knolling_dataset.teacher_v3 import solve,validate_layout,provenance,POLICY
from knolling_dataset.rendering import make_job,render_batch,validate_render
from knolling_dataset.builder import file_hash,write_json
from knolling_model.recovery.dataset_v3 import supervision_metadata,verify_contrast
from knolling_model.recovery.data import recovery_supervision
from knolling_model.experiments.dataset_compatibility import verify_expanded_teacher

SEED=120912
REGISTRY_SEED=120913
SCHEMA='expanded_knolling_20000_v1'


def partition(i):
    # Twenty-family blocks balance all four contrast factors in every split.
    bucket=(i//20)%10
    return 'train' if bucket<8 else 'validation' if bucket==8 else 'test'


def prepare_family(index):
    registry=build_registry(REGISTRY_SEED,64);split=partition(index)
    family='expanded_'+digest([SCHEMA,SEED,index])[:24];rejections=Counter()
    for attempt in range(128):
        seed=SEED*1000000+index*128+attempt;rng=random.Random(seed)
        try:
            source=generate_scene(registry[split],seed,split,2+index%7,'partial',family)
            if len({o.color for o in source.objects})<2:raise ValueError('need distinct partial referents')
            # Region contrasts must be disjoint left/right; full vs right is
            # not bidirectional because a right layout also satisfies full.
            base=generate_instruction(source,seed+1,'full',region='left' if index%4==3 else 'full',axis_alignment='horizontal')
            color=rng.choice(sorted({o.color for o in source.objects}));ids=tuple(o.object_id for o in source.objects)
            selected=tuple(o.object_id for o in source.objects if o.color==color)
            partial=replace(base,selected_ids=selected,locked_ids=tuple(i for i in ids if i not in selected))
            keep=generate_instruction(source,seed+2,'keep_all')
            factor=('gap_m','sort_direction','group_by','region')[index%4]
            value=dict(gap_m=.030 if base.gap_m!=.030 else .015,
                sort_direction='desc' if base.sort_direction=='asc' else 'asc',
                group_by='none' if base.group_by=='color' else 'color',region='right')[factor]
            instructions=[base,replace(base,axis_alignment='vertical'),partial,keep,replace(base,**{factor:value})]
            records=[]
            for j,ins in enumerate(instructions):
                ins=english_instruction(source,ins);result=solve(source,ins)
                if result.status!='ok':raise ValueError(str(result.reason))
                if not validate_layout(source,result.target,ins,result.mode).valid:raise RuntimeError('Teacher accepted invalid target')
                r=dict(schema_version='1.2',sample_id=f'{family}_{j}',scene_family_id=family,split=split,
                    mode=result.mode,role='expanded_arrangement',status='ok',source=asdict(source),
                    target=asdict(result.target),instruction=asdict(ins),teacher_policy=POLICY,
                    source_render_dir=f'families/{index:04d}/source',file_hashes={},teacher_audit=result.audit)
                r['supervision_v3']=supervision_metadata(r);records.append(r)
            pairs=[]
            for j,f in ((1,'axis_alignment'),(2,'selection'),(4,factor)):
                verify_contrast(records[0],records[j],f)
                pairs.append(dict(pair_id=digest([family,f]),factor=f,sample_ids=[records[0]['sample_id'],records[j]['sample_id']]))
            audit=audit_english_records(records)
            if not audit['valid']:raise RuntimeError(str(audit['errors']))
            return dict(index=index,family=family,records=records,pairs=pairs,attempts=attempt+1,rejections=dict(rejections))
        except (SamplingFailure,ValueError) as exc:rejections[str(exc)]+=1
    raise RuntimeError(f'{family}: bounded generation failed: {dict(rejections)}')


def verify_family(root,bundle,replay=False):
    ids=[]
    for r in bundle['records']:
        ids.append(r['sample_id']);recovery_supervision(r)
        for path,sha in r['file_hashes'].items():
            if file_hash(root/path)!=sha:raise ValueError('immutable render changed: '+path)
        s,ins=scene_from_dict(r['source']),instruction_from_dict(r['instruction'])
        if not validate_layout(s,scene_from_dict(r['target']),ins,r['mode']).valid:raise ValueError('invalid target')
        if replay and digest(asdict(solve(s,ins).target))!=digest(r['target']):raise ValueError('Teacher replay mismatch')
    assert len(ids)==len(set(ids))==5
    byid={r['sample_id']:r for r in bundle['records']}
    for p in bundle['pairs']:verify_contrast(*(byid[s] for s in p['sample_ids']),p['factor'])


def generate_one(args):
    root,index=args;root=Path(root);folder=root/'families'/f'{index:04d}';done=folder/'family.json'
    if shutil.disk_usage(root).free<13*2**30:raise RuntimeError('disk safety floor')
    if done.exists():
        b=json.loads(done.read_text())
        assert all(r['split']==partition(index) for r in b['records']),'stale split family'
        verify_family(root,b);return b
    b=prepare_family(index);source=scene_from_dict(b['records'][0]['source'])
    folder.mkdir(parents=True,exist_ok=True)
    render_batch([make_job(source,'source_pose',folder/'source',b['family'])],folder/'work')
    validation=validate_render(source,'source_pose',folder/'source')
    if not validation['valid']:raise RuntimeError('render integrity failure: '+str(validation['errors']))
    hashes={str((folder/'source'/name).relative_to(root)):file_hash(folder/'source'/name)
            for name in ('rgb.png','instance_mask.png','render.json')}
    for r in b['records']:r['file_hashes']=hashes
    b['render_validation']=validation
    verify_family(root,b,replay=True);write_json(done,b);return b


def generate(root,pilot=False,workers=5):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    contract=dict(schema=SCHEMA,seed=SEED,registry_seed=REGISTRY_SEED,teacher=provenance(),families=4000,samples=20000,
        splits=dict(train=16000,validation=2000,test=2000),source_views_only=True,partition='twenty_family_blocks_v2')
    pin=root/'contract.json'
    if pin.exists() and json.loads(pin.read_text())!=contract:raise ValueError('generation contract changed')
    if not pin.exists():write_json(pin,contract);write_json(root/'asset_registry.json',registry_dict(build_registry(REGISTRY_SEED,64)))
    if not pilot and not (root/'pilot-pass.json').exists():raise ValueError('Pilot must pass before expansion')
    indices=list(range(20))+list(range(160,166))+list(range(180,186)) if pilot else list(range(4000))
    total=len(indices);started=time.monotonic();bundles=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(generate_one,(str(root),i)) for i in indices]
        for f in tqdm(as_completed(futures),total=total,desc='CPU source families'):
            bundles.append(f.result())
            if len(bundles)%16==0:write_json(root/'generation-progress.json',dict(completed=len(bundles),total=total,seconds=time.monotonic()-started))
    bundles.sort(key=lambda b:b['index']);records=[r for b in bundles for r in b['records']];pairs=[p for b in bundles for p in b['pairs']]
    assert len(records)==total*5 and len({r['sample_id'] for r in records})==len(records)
    seen={};rgbs={}
    for b in bundles:
        r=b['records'][0];state=digest([{k:v for k,v in o.items() if k!='object_id'} for o in r['source']['objects']])
        rgb=r['file_hashes'][r['source_render_dir']+'/rgb.png']
        assert state not in seen and rgb not in rgbs,'duplicate observable source'
        seen[state]=b['family'];rgbs[rgb]=b['family']
    counts=Counter(r['split'] for r in records)
    report=dict(passed=True,samples=len(records),families=total,split_counts=dict(counts),
        object_counts=dict(Counter(len(b['records'][0]['source']['objects']) for b in bundles)),
        modes=dict(Counter(r['mode'] for r in records)),contrast_factors=dict(Counter(p['factor'] for p in pairs)),
        split_contrast_factors={s:dict(Counter(p['factor'] for b in bundles if b['records'][0]['split']==s for p in b['pairs']))
                               for s in ('train','validation','test')},
        attempts=sum(b['attempts'] for b in bundles),seconds=time.monotonic()-started,teacher_replay=True,
        bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),free_bytes=shutil.disk_usage(root).free)
    if pilot:
        report['pilot_bytes']=sum(p.stat().st_size for i in indices for p in (root/'families'/f'{i:04d}').rglob('*') if p.is_file())
        report['estimated_20000_bytes_with_2x_margin']=report['pilot_bytes']/32*4000*2
        assert report['free_bytes']-report['estimated_20000_bytes_with_2x_margin']-2*2**30>=12*2**30
        write_json(root/'pilot-pass.json',report)
    else:
        assert dict(counts)==contract['splits'];write_json(root/'annotations.json',dict(contract=contract,records=records,pairs=pairs))
        report['annotation_sha256']=file_hash(root/'annotations.json');write_json(root/'complete.json',report)
    return report


class ExpandedDataset(torch.utils.data.Dataset):
    def __init__(self,root,split,limit=None):
        self.root=Path(root);done=json.loads((self.root/'complete.json').read_text());path=self.root/'annotations.json'
        annotation_sha256=file_hash(path)
        if annotation_sha256!=done['annotation_sha256']:raise ValueError('dataset hash mismatch')
        data=json.loads(path.read_text())
        self.provenance_match=verify_expanded_teacher(data['contract']['teacher'],annotation_sha256)
        self.records=[r for r in data['records'] if r['split']==split]
        if limit is not None:self.records=self.records[:limit]
        ids={r['sample_id'] for r in self.records};self.pairs=[p for p in data['pairs'] if set(p['sample_ids'])<=ids]
        self.fingerprint=digest([done['annotation_sha256'],split,limit])
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];path=self.root/r['source_render_dir']/'rgb.png'
        with Image.open(path) as im:pixels=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/255
        ids,t=recovery_supervision(r)
        return dict(image=pixels,text=r['instruction']['text'],supervision=t,metadata=dict(sample_id=r['sample_id'],object_ids=ids,record=r))
