"""v1.3 integrated RGB/text → source instances → three target refinements."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import FeaturePyramidNetwork, FrozenBatchNorm2d

from .coordinates import wrap_pi


def position_encoding(height, width, device, dtype, dimension=256):
    # Image y-down coordinates, converted to the workspace's y-up convention.
    y, x = torch.meshgrid((torch.arange(height, device=device) + .5) / height,
                          (torch.arange(width, device=device) + .5) / width, indexing='ij')
    frequencies = 10000 ** (torch.arange(dimension // 4, device=device) / (dimension // 4))
    def encode(values):
        phase = values.reshape(-1, 1) * (2 * math.pi) / frequencies
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
    return torch.cat((encode(x), encode(1 - y)), -1).to(dtype)


class VisualBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
                             norm_layer=FrozenBatchNorm2d)
        self.body.avgpool = nn.Identity()
        self.body.fc = nn.Identity()
        self.fpn = FeaturePyramidNetwork([128, 256, 512], 256)
        self.level = nn.Parameter(torch.randn(3, 256) * .02)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 384, 576):
            raise ValueError('v1.3 requires Bx3x384x576 RGB')
        x = (images - self.mean) / self.std  # No resize/crop or target-aware transforms.
        b = self.body
        x = b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x)))))
        c3 = b.layer2(x)
        c4 = b.layer3(c3)
        c5 = b.layer4(c4)
        maps = self.fpn({'8': c3, '16': c4, '32': c5})
        memory, positions = [], []
        for level, feature in enumerate(maps.values()):
            memory.append(feature.flatten(2).transpose(1, 2))
            positions.append(position_encoding(*feature.shape[-2:], feature.device, feature.dtype) + self.level[level])
        return torch.cat(memory, 1), torch.cat(positions, 0).unsqueeze(0)


class DecoderBlock(nn.Module):
    """Pre-LN self-attention, cross-attention, FFN; never masks object queries."""
    def __init__(self, dropout=0.):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(3))
        self.self_attn = nn.MultiheadAttention(256, 8, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(256, 8, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(dropout), nn.Linear(1024, 256))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, query_position=None, memory_position=None, memory_padding_mask=None):
        z = self.norms[0](x)
        q = z if query_position is None else z + query_position
        x = x + self.drop(self.self_attn(q, q, z, need_weights=False)[0])
        z = self.norms[1](x)
        q = z if query_position is None else z + query_position
        key = memory if memory_position is None else memory + memory_position
        x = x + self.drop(self.cross_attn(q, key, memory, key_padding_mask=memory_padding_mask,
                                        need_weights=False)[0])
        return x + self.drop(self.ffn(self.norms[2](x)))


def mlp(output):
    return nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, output))


class SourceHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = mlp(2)
        self.size = mlp(2)
        self.direction = mlp(2)
        self.objectness = nn.Linear(256, 1)
        self.color, self.shape = nn.Linear(256, 3), nn.Linear(256, 3)
        nn.init.zeros_(self.size[-1].weight)
        with torch.no_grad():
            self.size[-1].bias.copy_(torch.tensor([math.log(math.expm1(.035 - 1e-4)),
                                                 math.log(math.expm1(.060 - .035))]))
        nn.init.zeros_(self.direction[-1].weight)
        with torch.no_grad():
            self.direction[-1].bias.copy_(torch.tensor([1., 0.]))

    def forward(self, tokens):
        sizes = self.size(tokens)
        b = F.softplus(sizes[..., 0]) + 1e-4
        a = b + F.softplus(sizes[..., 1])
        raw_direction = self.direction(tokens)
        # Only stabilize the undefined zero vector; no orientation-norm loss.
        near_zero = raw_direction.square().sum(-1) < 1e-12
        c = torch.where(near_zero, raw_direction[..., 0] + 1e-6, raw_direction[..., 0])
        angle = .5 * torch.atan2(raw_direction[..., 1], c)
        return {'uv': self.center(tokens).sigmoid(), 'size': torch.stack((a, b), -1),
                'angle': angle, 'objectness_logits': self.objectness(tokens).squeeze(-1),
                'color_logits': self.color(tokens), 'shape_logits': self.shape(tokens)}


class ArrangementDecoder(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.blocks = nn.ModuleList(DecoderBlock(dropout) for _ in range(3))
        # Source uv(2), ab(2), direction(2), q(1), previous target uv(2), direction(2).
        self.geometry = nn.ModuleList(nn.Sequential(nn.Linear(11, 256), nn.GELU(), nn.Linear(256, 256)) for _ in range(3))
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(3))
        self.residuals = nn.ModuleList(mlp(3) for _ in range(3))
        for head in self.residuals:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, objects, source, text, mask):
        uv, angle, tokens = source['uv'], source['angle'], objects
        source_features = torch.cat((source['uv'], source['size'] / source['size'].new_tensor([.6, .4]),
                                     torch.stack(((2 * source['angle']).cos(), (2 * source['angle']).sin()), -1),
                                     source['objectness_logits'].sigmoid().unsqueeze(-1)), -1)
        stages = []
        for k, alpha in enumerate((1., .5, .25)):
            geometry = torch.cat((source_features, uv, torch.stack(((2 * angle).cos(), (2 * angle).sin()), -1)), -1)
            tokens = self.blocks[k](tokens + self.geometry[k](geometry), text, memory_padding_mask=~mask)
            residual = self.residuals[k](self.norms[k](tokens))
            uv = (torch.logit(uv.clamp(1e-4, 1 - 1e-4)) + alpha * residual[..., :2]).sigmoid()
            angle = wrap_pi(angle + math.pi / 2 * residual[..., 2].tanh())
            stages.append({'uv': uv, 'angle': angle})
        return stages


class KnollingModel(nn.Module):
    def __init__(self, text_encoder, pretrained_backbone=True, dropout=0.):
        super().__init__()
        self.backbone = VisualBackbone(pretrained_backbone)
        self.queries = nn.Parameter(torch.randn(12, 256) * .02)
        self.object_blocks = nn.ModuleList(DecoderBlock(dropout) for _ in range(4))
        self.object_norm = nn.LayerNorm(256)
        self.source_heads = SourceHeads()
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.text_projection = nn.Linear(384, 256)
        self.arrangement = ArrangementDecoder(dropout)

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()  # Frozen weights AND deterministic pretrained dropout.
        return self

    def forward(self, images, input_ids, attention_mask):
        if (input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or
                input_ids.shape[0] != images.shape[0] or input_ids.shape[1] > 512 or
                not attention_mask.bool().any(-1).all()):
            raise ValueError('invalid or truncated text batch')
        memory, position = self.backbone(images)
        query_position = self.queries.unsqueeze(0).expand(images.shape[0], -1, -1)
        objects = torch.zeros_like(query_position)
        for block in self.object_blocks:
            objects = block(objects, memory, query_position, position)
        objects = self.object_norm(objects)
        source = self.source_heads(objects)
        with torch.no_grad():
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text = self.text_projection(text_features)
        stages = self.arrangement(objects, source, text, attention_mask.bool())
        return {'source': source, 'targets': stages, 'object_tokens': objects}


def load_pretrained(config_path='configs/pretrained.json', local_files_only=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    config = json.loads(Path(config_path).read_text())
    # Resolve a pinned snapshot first. Passing a remote ID directly to this
    # tokenizer version can make an extra Hub probe even in local-files-only mode.
    snapshot = snapshot_download(config['text_model'], revision=config['text_revision'],
                                 token=False, local_files_only=local_files_only,
                                 allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                                 'tokenizer_config.json', 'special_tokens_map.json',
                                                 'sentencepiece.bpe.model'])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    text_encoder = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True)
    if text_encoder.config.hidden_size != 384 or text_encoder.config.max_position_embeddings != 512:
        raise ValueError('unexpected text encoder configuration')
    if local_files_only and not (Path(torch.hub.get_dir()) / 'checkpoints' / 'resnet18-f37072fd.pth').is_file():
        raise FileNotFoundError('ResNet-18 weights are not cached; prepare assets online first')
    return KnollingModel(text_encoder, pretrained_backbone=True), tokenizer


@torch.no_grad()
def decode_predictions(output):
    """No GT count, sorting, NMS, parser or layout repair. Keys are query indices."""
    predictions = []
    for b in range(output['source']['uv'].shape[0]):
        source, target = output['source'], output['targets'][-1]
        indices = torch.where(source['objectness_logits'][b].sigmoid() >= .5)[0]
        predictions.append({'query_index': indices,
                            **{k: v[b, indices] for k, v in source.items()},
                            'objectness': source['objectness_logits'][b, indices].sigmoid(),
                            'color_probabilities': source['color_logits'][b, indices].softmax(-1),
                            'shape_probabilities': source['shape_logits'][b, indices].softmax(-1),
                            'target_uv': target['uv'][b, indices], 'target_angle': target['angle'][b, indices]})
    return predictions
