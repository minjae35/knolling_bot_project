"""Read-only replay of both pinned Best models and their RGB-to-layout bridge."""
import argparse
import json
from pathlib import Path
import torch

from knolling_model.current import load_current
from knolling_model.runner import seed_everything
from knolling_model.data import S0Dataset, Collator
from knolling_model.recovery.dataset_v3 import RecoveryDatasetV3
from knolling_model.recovery.goal_labels import GoalDataset
from knolling_model.recovery.alternating import evaluate_alternating
from knolling_model.recovery.storage import load_artifact, tensor_digest
from knolling_model.recovery.runtime import warm_text_cache
from knolling_model.perception import checkpoint
from knolling_model.perception.metrics import evaluate as evaluate_perception
from knolling_model.perception.pipeline import catalog


def run(manifest, output, device):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(manifest).read_text());seed_everything(1203,2)
    system=load_current(manifest,device=device)
    original={k:tensor_digest(getattr(system,k).state_dict()) for k in ('perception','arrangement')}
    review=json.loads(Path(spec['arrangement']['model_config']).read_text())
    dataset=GoalDataset(RecoveryDatasetV3(S0Dataset(review['dataset'],review['selection']),
        review['data_v3']['path'],review['data_v3']['sha256']))
    saved=load_artifact(spec['arrangement']['checkpoint'])
    assert dataset.fingerprint==saved['dataset_fingerprint'] and len(dataset)==76 and len(dataset.pairs)==35
    col=Collator(system.tokenizer)
    batches=[col([dataset[j] for j in range(i,min(i+8,len(dataset)))]).to(device) for i in range(0,len(dataset),8)]
    # Match the original Driver's token-only cache order/shapes exactly. Duplicate
    # commands can otherwise be re-encoded at different padded lengths.
    warm_text_cache(system.arrangement,dataset,col,8,device)
    metrics,rows=evaluate_alternating(system.arrangement,batches,stage='oracle_layout',
        expected_ids=[r['sample_id'] for r in dataset.records],pairs=dataset.pairs)
    reference=json.loads(Path(spec['arrangement']['evaluation']).read_text())
    evidence=metrics['integrated_evidence'];expected=reference['metrics']['integrated_evidence']
    assert evidence['successful_scenes']==expected['successful_scenes']
    assert evidence['successful_pairs']==expected['successful_pairs']
    assert len(evidence['successful_scenes'])==40 and len(evidence['successful_pairs'])==7
    prior={r['sample_id']:r for r in reference['rows']}
    exact=all(r['predicted_targets']==prior[r['sample_id']]['predicted_targets'] for r in rows)
    if device=='cuda':assert exact,'Arrangement CUDA replay differs'
    else:
        for r in rows:
            for oid,pose in r['predicted_targets'].items():
                for key,value in pose.items():
                    assert abs(value-prior[r['sample_id']]['predicted_targets'][oid][key])<1e-5
    (output/'arrangement.json').write_text(json.dumps(dict(metrics=metrics,rows=rows),indent=2))
    payload=checkpoint.read(spec['perception']['checkpoint'])
    data=catalog(payload['config']);assert data.fingerprint==payload['dataset_fingerprint']
    validation=data.dataset('validation')
    pm,frames=evaluate_perception(system.perception,validation,device=device,
        batch_size=payload['config']['batch_size'],threshold=payload['config']['threshold'])
    perception_exact=pm==payload['metrics']
    if device=='cuda':assert perception_exact,'Perception CUDA replay differs'
    else:
        for k in ('true_positives','false_positives','false_negatives','count_accuracy'):
            assert pm[k]==payload['metrics'][k]
    (output/'perception.json').write_text(json.dumps(dict(metrics=pm,frames=frames),indent=2))
    bridge=[]
    for i in range(len(validation)):
        sample=validation[i]
        try:
            # RGB and an ordinary command only; truth never enters this call.
            out=system.predict(sample['image'][None].to(device),
                ['Arrange all objects in horizontal rows with 20 mm gaps.'],[sample['family_id']])
        except ValueError as exc:
            if 'capacity exceeded' not in str(exc):raise
            bridge.append(dict(family_id=sample['family_id'],status='explicit_capacity_rejection',reason=str(exc)))
            continue
        layout=out['layout'];valid=layout['valid']
        assert torch.isfinite(layout['xy']).all() and torch.isfinite(layout['angle']).all()
        assert torch.equal(valid,out['objects'].valid)
        assert torch.equal(layout['query_index'],out['objects'].query_index)
        bridge.append(dict(family_id=sample['family_id'],status='forward_pass',count=int(valid.sum()),
            source_mode=out['provenance'][0]['source_mode'],
            target_xy=layout['xy'][valid].cpu().tolist(),target_angle=layout['angle'][valid].cpu().tolist()))
    assert any(r['status']=='forward_pass' for r in bridge)
    for key,value in original.items():assert tensor_digest(getattr(system,key).state_dict())==value
    result=dict(status='PASS',device=device,optimizer_steps=0,arrangement_validity=40,arrangement_counterfactual=7,
        arrangement_predictions_exact=exact,perception_metrics_exact=perception_exact,
        perception_count_accuracy=pm['count_accuracy'],perception_map_50_95=pm['map_50_95'],
        bridge=bridge,model_states_unchanged=True,
        limitation='Interface verification only; no End-to-End success or generalization claim')
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='configs/current_models.json')
    p.add_argument('--output',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=p.parse_args();run(args.manifest,args.output,args.device)
