"""Pinned RT-DETR-R18 perception; all 300 query rows remain in the joint graph.

Uses the pretrained HF modules, with equivalent deterministic sampling and
without the stock proposal/refinement stop-gradients. Native encoder top-300
proposal initialization remains; no decoded object is filtered before layout.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .deterministic_sampling import DeterministicDeformableAttention


def load_detector(pins, local_files_only=False):
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import RTDetrForObjectDetection
    if transformers.__version__ != '4.57.3':
        raise ValueError('RT-DETR adapter requires reviewed transformers==4.57.3')
    snapshot = Path(snapshot_download(pins['detector_model'], revision=pins['detector_revision'],
        token=False, local_files_only=local_files_only,
        allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors']))
    digest = hashlib.sha256((snapshot/'model.safetensors').read_bytes()).hexdigest()
    if digest != pins['detector_sha256']:
        raise ValueError('RT-DETR pretrained weight SHA-256 mismatch')
    processor = json.loads((snapshot/'preprocessor_config.json').read_text())
    if (processor['size'] != {'height': 640, 'width': 640} or processor['do_normalize']
            or not processor['do_rescale'] or processor['rescale_factor'] != 1/255):
        raise ValueError('unexpected RT-DETR pretrained image preprocessing')
    detector, loading = RTDetrForObjectDetection.from_pretrained(snapshot, local_files_only=True,
        output_loading_info=True, attn_implementation='eager')
    if any(loading.values()):
        raise ValueError(f'incomplete RT-DETR checkpoint loading: {loading}')
    config = detector.config
    if (config.d_model != 256 or config.num_queries != 300 or config.decoder_layers != 3
            or config.backbone_config.depths != [2, 2, 2, 2]
            or config.learn_initial_query or not config.with_box_refine
            or config.num_feature_levels != 3):
        raise ValueError('expected full RT-DETR-R18, 300 queries, three pretrained decoder layers')
    detector.pretrained_loading_info = loading
    return detector


class RTDetrPerception(nn.Module):
    def __init__(self, detector, image_size=(640, 640)):
        super().__init__()
        from transformers.models.rt_detr.modeling_rt_detr import RTDetrMultiscaleDeformableAttention
        self.detector = detector
        self.image_size = tuple(image_size)
        if self.image_size != (640, 640):
            raise ValueError('reviewed RT-DETR preprocessing is 640x640, RGB [0,1], no normalization')
        for module in detector.modules():
            if isinstance(module, RTDetrMultiscaleDeformableAttention):
                module.attn = DeterministicDeformableAttention()
        # Denoising queries contain GT and are deliberately absent from this joint
        # model. Retain their pretrained weights for provenance, never optimize them.
        if hasattr(detector.model, 'denoising_class_embed'):
            detector.model.denoising_class_embed.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # S0's micro batches must not change pretrained normalization statistics.
        # Affine parameters remain trainable; train/eval use the same running stats.
        for module in self.detector.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 384, 576):
            raise ValueError('expected original Bx3x384x576 RGB [0,1]')
        pixels = F.interpolate(images, size=self.image_size, mode='bilinear', align_corners=False)
        model = self.detector.model
        mask = torch.ones(pixels.shape[0], *self.image_size, device=pixels.device)
        features = model.backbone(pixels, mask)
        projected = [model.encoder_input_proj[i](feature) for i, (feature, _) in enumerate(features)]
        encoded = model.encoder(projected, return_dict=True).last_hidden_state
        sources = [model.decoder_input_proj[i](feature) for i, feature in enumerate(encoded)]
        shapes_list = tuple(feature.shape[-2:] for feature in sources)
        shapes = torch.tensor(shapes_list, device=pixels.device, dtype=torch.long)
        start = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
        memory = torch.cat([feature.flatten(2).transpose(1, 2) for feature in sources], 1)
        anchors, valid = model.generate_anchors(shapes_list, device=pixels.device, dtype=pixels.dtype)
        proposals = model.enc_output(memory*valid.to(memory.dtype))
        proposal_logits = model.enc_score_head(proposals)
        proposal_boxes = model.enc_bbox_head(proposals) + anchors
        indices = proposal_logits.max(-1).values.topk(model.config.num_queries, dim=1).indices
        def select(tensor):
            return tensor.gather(1, indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1]))
        # These retain gradients, unlike the stock RT-DETR training forward.
        tokens = select(proposals)
        references = select(proposal_boxes).sigmoid()
        stages = [{'tokens': tokens, 'boxes': references, 'logits': select(proposal_logits)}]
        from transformers.models.rt_detr.modeling_rt_detr import inverse_sigmoid
        for i, layer in enumerate(model.decoder.layers):
            tokens = layer(tokens, position_embeddings=model.decoder.query_pos_head(references),
                encoder_hidden_states=memory, reference_points=references.unsqueeze(2),
                spatial_shapes=shapes, spatial_shapes_list=shapes_list,
                level_start_index=start, output_attentions=False)[0]
            references = (self.detector.bbox_embed[i](tokens) + inverse_sigmoid(references)).sigmoid()
            stages.append({'tokens': tokens, 'boxes': references,
                           'logits': self.detector.class_embed[i](tokens)})
        return stages
