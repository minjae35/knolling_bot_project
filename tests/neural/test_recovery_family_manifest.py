"""Metadata-only pilot fixtures; no new rendered/training dataset is generated."""
from copy import deepcopy

import pytest

from knolling_dataset.schema import digest
from knolling_dataset.teacher_v3 import provenance,POLICY
from knolling_model.recovery.family_manifest import (coverage,validate_manifest,source_hash,
    FACTORS,PARTITIONS,SCHEMA,FamilyDataset,evaluate_family_pilot)
from test_recovery_dataset_v3 import suite_v3
from test_recovery_evaluation_v4 import ExactV4Fixture,fixture_batches


def synthetic_manifest():
    families={};records=[];pairs=[]
    for i in range(85):
        split='train' if i<65 else 'validation' if i<75 else 'test';fid=f'f{i}'
        rgb=f'{fid}/rgb.png';rgbhash=digest(rgb)
        source={'objects':[dict(object_id=str(j),color='red',source_pose={'x_m':i*.001+j*.1,'y_m':.1,'theta_rad':0},
                asset=dict(a_m=.03,b_m=.02,h_m=.01,shape='box',footprint=[])) for j in range(2+i%3)]}
        for j,factor in enumerate(sorted(FACTORS)):
            members=[]
            for k in range(2):
                sid=f'{fid}/{factor}/{k}';members.append(sid)
                records.append(dict(sample_id=sid,scene_family_id=fid,split=split,source=source,source_render_dir=fid,
                    file_hashes={rgb:rgbhash},mode=('full','partial','no_op')[j%3],teacher_policy=POLICY,
                    instruction={'text':f'{factor}/{k}'}))
            pairs.append(dict(pair_id=f'{fid}/{factor}',factor=factor,sample_ids=members))
        families[fid]=dict(partition=split,origin='verification' if i<25 else 'new',
            source_sha256=source_hash(records[-1]),rgb_sha256=rgbhash)
    expected={}
    for split in PARTITIONS:
        subset=[r for r in records if r['split']==split];ids={r['sample_id'] for r in subset}
        expected[split]=coverage(subset,[p for p in pairs if set(p['sample_ids'])<=ids])
    return dict(schema=SCHEMA,teacher=provenance(),families=families,expected=expected),records,pairs


def test_pilot_partition_counts_hashes_coverage_fail_closed(monkeypatch):
    # Isolate manifest logic; real pure-pair validation is tested on the existing
    # Teacher bundle in test_recovery_dataset_v3, not bypassed by FamilyDataset.
    monkeypatch.setattr('knolling_model.recovery.family_manifest.verify_contrast',lambda *args:None)
    m,records,pairs=synthetic_manifest();validate_manifest(m,records,pairs)
    for change in ('split','missing_pair','counts','source_duplicate','rgb_duplicate','cross_pair','duplicate_id'):
        a,b,c=deepcopy((m,records,pairs))
        if change=='split':b[-1]['split']='train'
        elif change=='missing_pair':c.pop()
        elif change=='counts':a['expected']['test']['records']-=1
        elif change=='source_duplicate':
            for r in b:
                if r['scene_family_id']=='f84':r['source']=deepcopy(b[0]['source'])
            a['families']['f84']['source_sha256']=source_hash(b[0])
        elif change=='rgb_duplicate':
            for r in b:
                if r['scene_family_id']=='f84':r['file_hashes']['f84/rgb.png']=a['families']['f0']['rgb_sha256']
            a['families']['f84']['rgb_sha256']=a['families']['f0']['rgb_sha256']
        elif change=='cross_pair':c[-1]['sample_ids'][0]=b[0]['sample_id']
        else:b[-1]['sample_id']=b[0]['sample_id']
        with pytest.raises(ValueError):validate_manifest(a,b,c)


def test_source_hash_ignores_ids_and_permutation(suite_v3):
    record=deepcopy(suite_v3.records[0]);before=source_hash(record)
    record['source']['objects'].reverse()
    for i,o in enumerate(record['source']['objects']):o['object_id']=f'new{i}'
    assert source_hash(record)==before
    record['source']['objects'][0]['source_pose']['x_m']+=.01
    assert source_hash(record)!=before


def test_manifest_path_and_hash_guards(tmp_path):
    p=tmp_path/'manifest.json';p.write_text('{}')
    with pytest.raises(ValueError,match='hash'):FamilyDataset(tmp_path,p,'wrong','test')
    d=object.__new__(FamilyDataset);d.root=tmp_path
    with pytest.raises(ValueError,match='escapes'):d._path('../outside')


def test_separate_pilot_evaluator_and_existing_76_gate(suite_v3):
    # Validated dataset stand-in isolates evaluation from future data generation.
    d=object.__new__(FamilyDataset);d.records=suite_v3.records;d.pairs=suite_v3.pairs
    d.split='test';d.fingerprint='evaluation-fixture'
    m=ExactV4Fixture(d.records);b=fixture_batches(d.records)
    result,_=evaluate_family_pilot(m,b,dataset=d)
    assert result['passed'] and result['grounding_passed'] and result['collisions']==0
    assert set(result['breakdown']['factor'])==FACTORS
    assert sum(v['total'] for v in result['breakdown']['mode'].values())==76
    m.layout.goal_encoder.confidence=1.
    result,_=evaluate_family_pilot(m,b,dataset=d)
    assert not result['passed']
