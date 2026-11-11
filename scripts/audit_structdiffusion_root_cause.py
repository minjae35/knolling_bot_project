#!/usr/bin/env python3
"""CPU-only, read-only model audit. GT epsilon interventions are DIAGNOSTIC ONLY.

No optimizer, model save, architecture mutation on disk, or training entrypoint.
All generated evidence goes to a new root-cause artifact directory.
"""
import importlib.util
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest,scene_from_dict,instruction_from_dict
from knolling_dataset.teacher_v3 import solve,validate_layout
from knolling_model.experiments.comparison import setup,percept_config,write,config as protect
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.structdiffusion_xy import build,pack_context
from knolling_model.recovery.contracts import oracle_objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.storage import load_artifact,tensor_digest
from knolling_model.recovery.alternating import alternating_evaluation


class DiagnosticNoise(nn.Module):
    def __init__(self,learner,z0,abar,threshold):
        super().__init__();self.learner=learner;self.z0=z0;self.abar=abar;self.threshold=threshold
    def forward(self,t,features,goal,z,valid):
        if int(t[0])>=self.threshold:
            a=self.abar[t,None,None]
            return (z-a.sqrt()*self.z0)/(1-a).sqrt()
        return self.learner(t,features,goal,z,valid)


def position_stats(pred,target,active):
    error=(pred-target).norm(dim=-1)[active]*1000
    return dict(p95_mm=float(error.quantile(.95)),max_mm=float(error.max()),
                coordinate_rms_mm=float((pred-target)[active].square().mean().sqrt()*1000))


def mathematics(pose,target,valid,upstream):
    result={}
    for dtype in (torch.float32,torch.float64):
        z0=(target.to(dtype)-pose.center.to(dtype))/pose.scale.to(dtype)
        beta=pose.betas.to(dtype);alpha=1-beta;abar=alpha.cumprod(0)
        previous=torch.cat((abar.new_ones(1),abar[:-1]));noise=torch.randn(z0.shape,generator=torch.Generator().manual_seed(912),dtype=dtype)
        inverse=[];posterior=[]
        for i in range(pose.steps):
            z=abar[i].sqrt()*z0+(1-abar[i]).sqrt()*noise
            recovered=(z-(1-abar[i]).sqrt()*noise)/abar[i].sqrt()
            inverse.append(float(((recovered-z0)*pose.scale)[valid].abs().max()*1000))
            production=alpha[i].rsqrt()*(z-beta[i]*noise/(1-abar[i]).sqrt())
            independent=(previous[i].sqrt()*beta[i]/(1-abar[i]))*z0+(alpha[i].sqrt()*(1-previous[i])/(1-abar[i]))*z
            posterior.append(float(((production-independent)*pose.scale)[valid].abs().max()*1000))
        limits=(1.,1.) if dtype==torch.float32 else (1e-6,1e-6)
        result[str(dtype)]=dict(status='PASS' if max(inverse)<limits[0] and max(posterior)<limits[1] else 'FAIL',
            inverse_max_coordinate_mm=max(inverse),inverse_terminal_mm=inverse[-1],
            posterior_mean_max_coordinate_mm=max(posterior),limits_mm=limits)
    official=upstream.NoiseSchedule(200)
    result['schedule']=dict(cosine_matches_source=bool(torch.equal(pose.betas,upstream.cosine_beta_schedule(200))),
        default_schedule_matches=False,official_default='linear',current='cosine',
        terminal_beta=float(pose.betas[-1]),terminal_alpha_bar=float(pose.abar[-1]),
        epsilon_to_clean_gain=float(((1-pose.abar[-1])/pose.abar[-1]).sqrt()),
        first_reverse_epsilon_gain=float(pose.sqrt_recip_alphas[-1]*pose.betas[-1]/(1-pose.abar[-1]).sqrt()),
        official_terminal_alpha_bar=float(official.alphas_cumprod[-1]),
        official_first_reverse_epsilon_gain=float(official.sqrt_recip_alphas[-1]*official.betas[-1]/official.sqrt_one_minus_alphas_cumprod[-1]))
    # Execute upstream q_sample with the production cosine buffers.
    official.sqrt_alphas_cumprod=pose.abar.sqrt();official.sqrt_one_minus_alphas_cumprod=(1-pose.abar).sqrt()
    t=torch.full((len(target),),199,dtype=torch.long);noise=torch.randn_like(target)
    z0=(target-pose.center)/pose.scale
    a=upstream.q_sample(z0,t,official,noise)
    b=pose.abar[t,None,None].sqrt()*z0+(1-pose.abar[t,None,None]).sqrt()*noise
    result['upstream_q_sample_exact']=bool(torch.equal(a,b))
    return result


@torch.no_grad()
def trace(pose,ctx,target):
    steps=[];z0=(target-pose.center)/pose.scale;active=ctx['move']
    def capture(module,args,eps):
        t,features,goal,z,valid=args;i=int(t[0]);a=pose.abar[i]
        ideal=(z-a.sqrt()*z0)/(1-a).sqrt()
        clean=(z-(1-a).sqrt()*eps)/a.sqrt()
        steps.append(dict(t=i,alpha_bar=float(a),noise_error_rms=float((eps-ideal)[active].square().mean().sqrt()),
            noisy_state_rms=float(z[active].square().mean().sqrt()),
            implied_clean=position_stats(clean*pose.scale+pose.center,target,active),
            per_scene_max_xy_mm=[float(((clean*pose.scale+pose.center-target)[j][active[j]]).norm(dim=-1).max()*1000) for j in range(len(target))]))
    hook=pose.denoiser.register_forward_hook(capture)
    try:prediction=pose.sample(ctx)
    finally:hook.remove()
    return prediction,steps


def main():
    started=time.monotonic();setup(percept_config());torch.set_num_threads(2)
    assert not torch.cuda.is_available(),'Launch with CUDA_VISIBLE_DEVICES empty'
    root=Path('artifacts/structdiffusion_root_cause');root.mkdir(parents=True,exist_ok=True)
    if (root/'audit.json').exists():raise RuntimeError('audit already complete; no automatic repeat')
    c=json.loads(Path('configs/structdiffusion_xy.json').read_text())
    protect('configs/structural_comparison.json')
    run=json.loads(Path(c['output'],'result.json').read_text());checkpoint=run['best'];cp_hash=file_hash(checkpoint)
    train=ExpandedDataset(c['dataset'],'train');val=ExpandedDataset(c['dataset'],'validation')
    lo=torch.full((2,),float('inf'));hi=-lo;roundtrip=0.;count=0
    for r in train.records+val.records:
        ids,t=recovery_supervision(r)
        for key in ('source_xy','target_xy'):
            xy=t[key];assert torch.isfinite(xy).all();lo=torch.minimum(lo,xy.min(0).values);hi=torch.maximum(hi,xy.max(0).values)
            scale=xy.new_tensor([.3,.2]);z=(xy-scale)/scale
            roundtrip=max(roundtrip,float(((z*scale+scale)-xy).abs().max()*1000));count+=len(xy)
    data_audit=dict(records=18000,coordinates=count,xy_min_m=lo.tolist(),xy_max_m=hi.tolist(),
        normalize_roundtrip_max_mm=roundtrip,ids_joined_by='object_id',
        status='PASS' if roundtrip<.001 and (lo>=0).all() and (hi<=torch.tensor([.6,.4])).all() else 'FAIL')
    rows=run['validation']['oracle']['rows'];failed=[r for r in rows if r['mode']!='no_op' and not r['task_success']]
    def error(r):return max(r['target_position_mm'])
    chosen=[]
    for mode in ('full','partial'):
        subset=sorted([r for r in failed if r['mode']==mode],key=error,reverse=True)
        chosen+=subset[:2]+subset[len(subset)//2:len(subset)//2+1]
    chosen += [r for r in rows if r['mode']!='no_op' and r['task_success']][:3]
    records_by_id={r['sample_id']:r for r in val.records};records=[records_by_id[r['sample_id']] for r in chosen]
    model=build(c,'cpu');model.arrangement.load_state_dict(load_artifact(checkpoint)['model'],strict=True)
    before=tensor_digest(model.arrangement.state_dict());pose=model.layout.pose
    contexts=[];targets=[];object_ids=[];teacher=[]
    with alternating_evaluation(model.arrangement),torch.no_grad():
        for r in records:
            ids,t=recovery_supervision(r);obj=oracle_objects([t]);object_ids.append(ids)
            tok=model.tokenizer([r['instruction']['text']],padding=True,return_tensors='pt');mask=tok['attention_mask'].bool()
            goal=model.layout.goal_encoder(obj,model.encode_text(tok['input_ids'],mask),mask)
            ctx,_=pose.context(obj,goal['vector'],goal['selected']);contexts.append(ctx)
            targets.append(torch.cat((t['target_xy'],torch.zeros(8-len(ids),2)),0))
            source=scene_from_dict(r['source']);ins=instruction_from_dict(r['instruction'])
            teacher.append(dict(sample_id=r['sample_id'],target_replay_exact=digest(asdict(solve(source,ins).target))==digest(r['target']),
                physical_valid=validate_layout(source,scene_from_dict(r['target']),ins,r['mode']).valid))
        ctx=pack_context(contexts);target=torch.stack(targets);z0=(target-pose.center)/pose.scale
        spec=importlib.util.spec_from_file_location('pinned_noise','third_party/structdiffusion/src/StructDiffusion/diffusion/noise_schedule.py')
        upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
        math=mathematics(pose,target,ctx['valid'],upstream)
        predictions,trajectory=trace(pose,ctx,target)
        original=pose.denoiser;interventions={}
        try:
            for name,threshold in [('all_steps_true_noise',0),('only_terminal_true_noise',199)]:
                pose.denoiser=DiagnosticNoise(original,z0,pose.abar,threshold)
                p=pose.sample(ctx);interventions[name]=dict(metrics=position_stats(p,target,ctx['move']),
                    per_scene_max_mm=[float((p[j]-target[j])[ctx['move'][j]].norm(dim=-1).max()*1000) for j in range(len(target))])
        finally:pose.denoiser=original
        oracle_ok=interventions['all_steps_true_noise']['metrics']['max_mm']<.001
        interventions['all_steps_true_noise']['status']='PASS' if oracle_ok else 'FAIL'
        double_pose=deepcopy(pose).double()
        # Native upstream-style timestep sin/cos creates FP32 values. Permit
        # the diagnostic-only double network to consume those identical values.
        # This is mixed precision at that embedding, NOT full native FP64 support.
        double_pose.denoiser.time_embeddings.layers.register_forward_pre_hook(lambda module,args:(args[0].double(),))
        double_ctx={k:v.double() if v.is_floating_point() else v for k,v in ctx.items()}
        double_prediction=double_pose.sample(double_ctx)
        precision=dict(fp32=position_stats(predictions,target,ctx['move']),fp64=position_stats(double_prediction,target.double(),ctx['move']),
            xy_max_difference_mm=float((double_prediction-predictions).abs().max()*1000),
            scope='FP64 weights/attention/reverse algebra; unchanged FP32 sinusoidal embedding cast to FP64',
            native_double_supported=False)
        del double_pose
    cases=[]
    for j,(row,r,ids) in enumerate(zip(chosen,records,object_ids)):
        saved=torch.tensor([[row['predicted_targets'][oid]['x_m'],row['predicted_targets'][oid]['y_m']] for oid in ids])
        cases.append(dict(sample_id=r['sample_id'],command=r['instruction']['text'],mode=r['mode'],errors=row['errors'],
            saved_task_success=row['task_success'],object_ids=ids,source_xy_m=contexts[j]['source'][0].mul(pose.scale).add(pose.center).tolist(),
            target_xy_m=target[j,:len(ids)].tolist(),predicted_xy_m=predictions[j,:len(ids)].tolist(),
            saved_forward_max_difference_mm=float((predictions[j,:len(ids)]-saved).abs().max()*1000),
            saved_forward_allclose=bool(torch.allclose(predictions[j,:len(ids)],saved,atol=2e-4,rtol=2e-5))))
    assert tensor_digest(model.arrangement.state_dict())==before
    assert file_hash(checkpoint)==cp_hash and file_hash(Path(c['dataset'])/'annotations.json')==c['dataset_sha256']
    protect('configs/structural_comparison.json')
    write(root/'audit.json',dict(device='cpu',optimizer_steps=0,seconds=time.monotonic()-started,checkpoint=checkpoint,checkpoint_sha256=cp_hash,
        dataset=data_audit,math=math,teacher=teacher,interventions=interventions,precision=precision,cases=cases,trajectory=trajectory,
        preserved_weights_exact=True,overfit='NOT_RUN: a sampling failure mechanism is already isolated; stop before further training'))
    print('CPU AUDIT COMPLETE',time.monotonic()-started,flush=True)


if __name__=='__main__':main()
