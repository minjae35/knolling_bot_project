from copy import deepcopy
import subprocess

import pytest

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from knolling_model.arrangement.driver import read_config,save_exclusive,load_grounding_transfer,load_checkpoint
from knolling_model.arrangement.provenance import contract_hashes,load_evidence_transfer
from knolling_model.arrangement.storage import CheckpointStore
import torch


@pytest.mark.parametrize('split', [False, True])
def test_grounding_transfer_contracts_and_explicit_changes(tmp_path, split):
    source=read_config('configs/recovery_v4_gpu_grounding.json')
    target=read_config('configs/recovery_v4_integrated_review.json')
    # Teacher contract includes the aspect-ratio tolerance from b3a3d6b.
    # Model, loss, and evaluator contracts are unchanged since 7ac097a.
    commit=subprocess.check_output(['git','rev-parse','b3a3d6b'],text=True).strip()
    contracts=contract_hashes()
    assert contracts==contract_hashes(commit)
    metrics=dict(schema='recovery_metrics_v4',architecture='recovery_v4_goal_joint',stage='grounding',
        source_mode='oracle',evaluation_mode='free_rollout',complete_set=True,grounding={'passed':True})
    raw=dict(schema='recovery_checkpoint_v2',architecture='recovery_v4_goal_joint',evaluation_schema='recovery_metrics_v4',
        commit=commit,config=source,config_sha256=digest(source),dataset_fingerprint='fixture',goal_contract=source['goal_contract'],
        purpose='production',stage='grounding',status='passed',metrics=metrics,completed={'grounding':metrics})
    if split:
        raw['model']={'text_encoder.weight':torch.ones(2), 'layout.goal_encoder.weight':torch.zeros(2)}
    save = CheckpointStore(tmp_path/'checkpoint_store').save_checkpoint if split else save_exclusive
    path=tmp_path/'parent.pt';save(path,raw)
    grant=dict(sha256=file_hash(path),commit=commit,config_sha256=digest(source),contracts=contracts,
        allowed_config_changes=sorted(k for k in set(source)|set(target) if source.get(k)!=target.get(k)))
    before=file_hash(path)
    saved=load_grounding_transfer(path,target,'fixture',{'grounding_transfer':grant})
    assert saved['commit']==commit and file_hash(path)==before
    with pytest.raises(ValueError):load_checkpoint(path,target,'fixture','new-commit')
    for key,value in [('sha256','bad'),('contracts',{}),('allowed_config_changes',[])]:
        bad=deepcopy(grant);bad[key]=value
        with pytest.raises(ValueError):load_evidence_transfer(path,target,'fixture',bad)
    changed=deepcopy(target);changed['micro_pose']['max_updates']+=1
    bad=deepcopy(grant);bad['allowed_config_changes'].append('micro_pose')
    with pytest.raises(ValueError):load_evidence_transfer(path,changed,'fixture',bad)
    raw['status']='budget_exhausted';path2=tmp_path/'failed.pt';save(path2,raw)
    bad=deepcopy(grant);bad['sha256']=file_hash(path2)
    with pytest.raises(ValueError,match='passed'):load_evidence_transfer(path2,target,'fixture',bad)
