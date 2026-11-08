"""Task foreground RT-DETR with isolated denoising and native detection losses.

Uses the pinned HF modules, not its detection-loss wrapper: in transformers
4.57.3 that wrapper splits DN only for auxiliary losses, leaving DN in the final
Hungarian input. Here every normal stage excludes DN. A broadcast self-attention
mask also prevents normal queries from reading DN ground truth.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.model import PhysicalSourceHeads
from knolling_model.rtdetr import RTDetrPerception
from .contracts import detector_labels


def install_foreground_heads(detector, num_denoising=100):
    """Keep pretrained geometry/vision; explicitly initialize new task classifiers."""
    if getattr(detector.config, 'recovery_foreground_v1', False):
        raise ValueError('foreground conversion already applied; do not reset trained heads')
    config = detector.config
    if config.d_model != 256 or config.learn_initial_query or not config.with_box_refine:
        raise ValueError('recovery requires 256-dim proposal queries with box refinement')
    config.num_labels = 1
    config.id2label, config.label2id = {0: 'object'}, {'object': 0}
    config.num_denoising = num_denoising
    config.auxiliary_loss = True
    config.weight_loss_vfl, config.weight_loss_bbox, config.weight_loss_giou = 1., 5., 2.
    config.matcher_class_cost, config.matcher_bbox_cost, config.matcher_giou_cost = 2., 5., 2.
    config.use_focal_loss = True
    config.recovery_foreground_v1 = True
    reference = next(detector.parameters())
    def head():
        value = nn.Linear(config.d_model, 1).to(reference)
        nn.init.normal_(value.weight, std=.01)
        nn.init.constant_(value.bias, math.log(.01/.99))
        return value
    detector.model.enc_score_head = head()
    detector.class_embed = nn.ModuleList(head() for _ in detector.class_embed)
    detector.model.decoder.class_embed = detector.class_embed
    detector.model.denoising_class_embed = nn.Embedding(2, config.d_model, padding_idx=1).to(reference)
    nn.init.normal_(detector.model.denoising_class_embed.weight, std=.02)
    with torch.no_grad():
        detector.model.denoising_class_embed.weight[1].zero_()
    return detector


class RecoveryPerception(RTDetrPerception):
    def __init__(self, detector, num_denoising=100):
        import transformers
        if transformers.__version__ != '4.57.3':
            raise ValueError('reviewed adapter requires transformers==4.57.3')
        super().__init__(install_foreground_heads(detector, num_denoising))
        self.detector.model.denoising_class_embed.requires_grad_(True)
        self.physical = PhysicalSourceHeads()

    def forward(self, images):
        return self._forward(images, None)

    def supervised(self, images, truths):
        """Training-only DN path. Layout inference never accepts this argument."""
        if not self.training:
            raise ValueError('denoising supervision is forbidden in evaluation')
        return self._forward(images, detector_labels(truths))

    def _forward(self, images, labels):
        if images.ndim != 4 or images.shape[1:] != (3, 384, 576):
            raise ValueError('expected Bx3x384x576 RGB')
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError('RGB must be finite in [0,1]')
        from transformers.models.rt_detr.modeling_rt_detr import (
            get_contrastive_denoising_training_group, inverse_sigmoid)
        pixels = F.interpolate(images, self.image_size, mode='bilinear', align_corners=False)
        model = self.detector.model
        features = model.backbone(pixels, torch.ones_like(pixels[:, 0]))
        projected = [model.encoder_input_proj[i](f) for i, (f, _) in enumerate(features)]
        encoded = model.encoder(projected, return_dict=True).last_hidden_state
        sources = [model.decoder_input_proj[i](f) for i, f in enumerate(encoded)]
        shapes_list = tuple(f.shape[-2:] for f in sources)
        shapes = torch.tensor(shapes_list, device=pixels.device, dtype=torch.long)
        start = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
        memory = torch.cat([f.flatten(2).transpose(1, 2) for f in sources], 1)
        anchors, valid = model.generate_anchors(shapes_list, device=pixels.device, dtype=pixels.dtype)
        proposals = model.enc_output(memory*valid.to(memory.dtype))
        logits = model.enc_score_head(proposals)
        boxes = model.enc_bbox_head(proposals) + anchors
        indices = logits.squeeze(-1).topk(model.config.num_queries, dim=1).indices
        def select(t):
            return t.gather(1, indices[..., None].expand(-1, -1, t.shape[-1]))
        tokens, references = select(proposals), select(boxes).sigmoid()
        stages = [dict(tokens=tokens, boxes=references, logits=select(logits))]
        dn_meta, mask, dn_count = None, None, 0
        if labels is not None and model.config.num_denoising > 0:
            dn_tokens, dn_boxes, mask, dn_meta = get_contrastive_denoising_training_group(
                labels, num_classes=1, num_queries=model.config.num_queries,
                class_embed=model.denoising_class_embed,
                num_denoising_queries=model.config.num_denoising,
                label_noise_ratio=model.config.label_noise_ratio,
                box_noise_scale=model.config.box_noise_scale)
            if dn_meta is not None:
                dn_count = dn_meta['dn_num_split'][0]
                tokens = torch.cat((dn_tokens, tokens), 1)
                references = torch.cat((dn_boxes.sigmoid(), references), 1)
                # Pinned HF attention expands this 2D mask to B,1,Q,Q itself.
        dn_stages = []
        for i, layer in enumerate(model.decoder.layers):
            tokens = layer(tokens, position_embeddings=model.decoder.query_pos_head(references),
                encoder_hidden_states=memory, reference_points=references.unsqueeze(2),
                spatial_shapes=shapes, spatial_shapes_list=shapes_list, level_start_index=start,
                encoder_attention_mask=mask, output_attentions=False)[0]
            references = (self.detector.bbox_embed[i](tokens)+inverse_sigmoid(references)).sigmoid()
            scores = self.detector.class_embed[i](tokens)
            stages.append(dict(tokens=tokens[:, dn_count:], boxes=references[:, dn_count:],
                               logits=scores[:, dn_count:]))
            if dn_count:
                dn_stages.append(dict(logits=scores[:, :dn_count], pred_boxes=references[:, :dn_count]))
        final = stages[-1]
        source = {'uv': torch.stack((final['boxes'][..., 0], 1-final['boxes'][..., 1]), -1),
                  'objectness_logits': final['logits'].squeeze(-1), **self.physical(final['tokens'])}
        return dict(source=source, stages=stages, dn_stages=dn_stages, dn_meta=dn_meta)

    def detection_loss(self, output, truths):
        from transformers.loss.loss_rt_detr import RTDetrLoss
        stages = output['stages']
        def prediction(stage):
            return dict(logits=stage['logits'], pred_boxes=stage['boxes'])
        packed = prediction(stages[-1])
        packed['auxiliary_outputs'] = [prediction(s) for s in stages[:-1]]
        if output['dn_meta'] is not None:
            packed['dn_auxiliary_outputs'] = output['dn_stages']
            packed['denoising_meta_values'] = output['dn_meta']
        criterion = RTDetrLoss(self.detector.config).to(stages[-1]['boxes'].device)
        labels = detector_labels(truths)
        losses = criterion(packed, labels)
        # Separate final source-only assignment for physical and layout labels.
        matches = criterion.matcher(prediction(stages[-1]), labels)
        source = output['source']
        physical = source['uv'].sum()*0
        count = sum(len(i) for i, _ in matches)
        from knolling_model.coordinates import angle_delta
        for b, (pred, gt) in enumerate(matches):
            if not len(pred):
                continue
            truth = truths[b]
            physical = physical + F.smooth_l1_loss(source['size'][b, pred]/.005,
                                                  truth['size'][gt]/.005, reduction='sum')
            delta = angle_delta(source['angle'][b, pred], truth['source_angle'][gt])/(torch.pi/18)
            physical = physical + F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction='sum')
            for key in ('color', 'shape'):
                physical = physical + .1*F.cross_entropy(source[key+'_logits'][b, pred],
                                                          truth[key][gt], reduction='sum')
        losses['physical'] = physical/max(count, 1)
        return losses, matches
