from copy import deepcopy
from dataclasses import replace
import json

import pytest

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import scene_from_dict,instruction_from_dict,digest
from knolling_dataset.teacher_v3 import solve,validate_layout
from knolling_model.arrangement.dataset_v3 import RecoveryDatasetV3,build_bundle
from knolling_model.arrangement.counterfactuals import RecoverySuite


@pytest.fixture
def suite_v3(s0_dataset):
    c=json.load(open('configs/recovery_v3_execution.json'))
    return RecoveryDatasetV3(s0_dataset,c['data_v3']['path'],c['data_v3']['sha256'])


def test_v3_rebuild_replay_dedup_and_all_six_factors(s0_dataset,suite_v3,tmp_path):
    c=json.load(open('configs/recovery_gpu_initial.json'))
    old=RecoverySuite(s0_dataset,c['counterfactuals']['path'],c['counterfactuals']['sha256'])
    built=build_bundle(s0_dataset,old,tmp_path/'rebuild')
    expected=json.load(open('configs/recovery_v3_execution.json'))['data_v3']['sha256']
    assert built['sha256']==expected
    assert len(suite_v3)==76 and len(suite_v3.pairs)==35
    assert len({r['scene_family_id'] for r in suite_v3.records})==25
    assert len({p['factor'] for p in suite_v3.pairs})==6
    for r in suite_v3.records:
        scene=scene_from_dict(r['source']);ins=instruction_from_dict(r['instruction'])
        a=solve(scene,ins);b=solve(replace(scene,objects=tuple(reversed(scene.objects))),ins)
        pa={o.object_id:o.target_pose for o in a.target.objects};pb={o.object_id:o.target_pose for o in b.target.objects}
        assert pa==pb
        assert validate_layout(scene,a.target,ins,a.mode).valid
        for o in a.target.objects:
            if r['supervision_v3']['copy_required'][o.object_id]:assert o.target_pose==o.source_pose
    with pytest.raises(ValueError,match='new output'):build_bundle(s0_dataset,old,tmp_path/'rebuild')


def test_loader_rejects_rehashed_bad_data(s0_dataset,suite_v3,tmp_path):
    c=json.load(open('configs/recovery_v3_execution.json'));original=json.load(open(c['data_v3']['path']))
    for key in ('target','text','teacher','rgb','split','supervision','pair'):
        b=deepcopy(original);r=b['records'][0]
        if key=='target':r['target']['objects'][0]['target_pose']['x_m']+=.001
        elif key=='text':r['instruction']['text']='Keep everything.'
        elif key=='teacher':r['teacher_policy']='balanced_rows_v1.3'
        elif key=='rgb':r['file_hashes'][next(iter(r['file_hashes']))]='bad'
        elif key=='split':r['split']='test'
        elif key=='supervision':r['supervision_v3']['structure_center_m'][0]+=.01
        else:b['pairs'][0]['factor']='group_by'
        path=tmp_path/f'{key}.json';path.write_text(json.dumps(b))
        with pytest.raises(ValueError):RecoveryDatasetV3(s0_dataset,path,file_hash(path))
