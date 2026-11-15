"""Learned Move/Keep routing; exact source copy uses predictions, never labels."""
import torch
from torch import nn
from torch.nn import functional as F
from .relative_task import TrainableSplitPose

POLICY = 'neural_keep_router_v1'


class NeuralKeepPose(TrainableSplitPose):
    def __init__(self, split):
        super().__init__(split)
        d=self.input.out_features
        self.move_keep_head=nn.Sequential(nn.Linear(2*d+26,d),nn.GELU(),nn.Linear(d,1)).to(self.input.weight.device)
        nn.init.zeros_(self.move_keep_head[-1].weight);nn.init.zeros_(self.move_keep_head[-1].bias)

    def forward(self, objects, goal, selected):
        hidden=self.input(torch.cat((objects.features(),selected[...,None]),-1))
        for block in self.blocks:hidden=block(hidden,goal,objects.valid)
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            raw=self.output(hidden.float())[...,:2]
            proposal=(raw*raw.new_tensor([.3,.2])+raw.new_tensor([.3,.2])).masked_fill(~objects.valid[...,None],0)
            feature=self.output[0](hidden.float()).masked_fill(~objects.valid[...,None],0)
            pooled=feature.sum(1)/objects.valid.sum(1).clamp_min(1)[:,None]
            # The current arrangement matters: selected objects can already be
            # correctly arranged. Predicted selection alone is not a Keep label.
            inputs=torch.cat((feature,pooled[:,None].expand_as(feature),objects.features(),
                goal[:,None].expand(-1,feature.shape[1],-1),torch.logit(selected.clamp(1e-6,1-1e-6))[...,None]),-1)
            logits=self.move_keep_head(inputs).squeeze(-1).masked_fill(~objects.valid,0)
        direction=self.orientation_pose(objects,goal,selected)
        move=(logits>=0)&objects.valid  # Zero initialization preserves the old all-Move forward.
        xy=torch.where(move[...,None],proposal,objects.xy).masked_fill(~objects.valid[...,None],0)
        angle=torch.where(move,direction['angle'],objects.angle).masked_fill(~objects.valid,0)
        return dict(direction,xy=xy,angle=angle,move_logit=logits,move_mask=move,
                    proposal_xy=proposal,proposal_angle=direction['angle'])


def attach(model):
    if not isinstance(model.layout.pose,NeuralKeepPose):model.layout.pose=NeuralKeepPose(model.layout.pose)


def labels(metadata, device):
    n=max(len(m['object_ids']) for m in metadata)
    target=torch.zeros(len(metadata),n,device=device)
    for i,m in enumerate(metadata):
        for j,oid in enumerate(m['object_ids']):target[i,j]=not m['record']['supervision_v3']['copy_required'][oid]
    return target


def loss(output, target):
    valid=output['valid']
    values=F.binary_cross_entropy_with_logits(output['move_logit'],target,reduction='none')
    return (values.masked_fill(~valid,0).sum(-1)/valid.sum(-1).clamp_min(1)).mean()


def metrics(output,target):
    mask=output['valid'];p=output['move_mask'][mask];y=target[mask].bool()
    return dict(correct=int((p==y).sum()),total=int(mask.sum()),
                false_keep=int((~p&y).sum()),false_move=int((p&~y).sum()))


def adopt(current,baseline):
    a,b=current['integrated_evidence'],baseline['integrated_evidence']
    return (len(a['successful_scenes'])>len(b['successful_scenes'])
        and len(a['successful_pairs'])>len(b['successful_pairs'])
        and set(b['successful_scenes'])<=set(a['successful_scenes'])
        and current['validity']['keep_violations']<=baseline['validity']['keep_violations']
        and current['target_angle_p95_deg']<=baseline['target_angle_p95_deg']+1e-6
        and all(a['angle_errors_deg'][k]<=2+1e-6 for k,v in b['angle_errors_deg'].items() if v<=2))
