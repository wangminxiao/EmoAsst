"""
Based on the implementation of EmotionCLIP model. 
https://github.com/Xeaver/EmotionCLIP.git
"""
from __future__ import annotations
import json
import os.path as osp
import json
import itertools
import collections.abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .pre_audio import AudioHead


# From PyTorch internals
def _ntuple(n: int):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(itertools.repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)



#####################################
#            CLIP Moduels           #
#####################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers: list[int], output_dim: int, heads: int, image_size: int = 224, width: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        # FIXME support for non-transformer
        pass

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
            self, image_size: int, patch_size: int, width: int, layers: int, heads: int, mlp_ratio: float,
            output_dim: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        self.avg_pool = nn.AvgPool2d(kernel_size=self.patch_size)

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x)  # shape = [*, width, grid, grid], mean = -0.02, std = 0.9
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12


class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            raise NotImplementedError('timm models not currently supported')
            from .timm_model import TimmModel
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size
            )
            act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            self.visual = VisualTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=embed_dim,
                act_layer=act_layer,
            )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        text: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        image_features = self.encode_image(image, image_mask)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()


#####################################
#            EmotionCLIP            #
#####################################

class VisualHead(nn.Module):
    def __init__(
        self, 
        temporal_fusion: str = 'transformer',
        embed_dim: int = 512,
        video_len: int = 8,
        transformer_nhead: Optional[int] = 8,
        transformer_nlayer: Optional[int] = 6,
        transformer_dim_feedforward: Optional[int] = 2048
    ) -> None:
        super().__init__()
        self.temporal_fusion = temporal_fusion
        self.embed_dim = embed_dim
        self.video_len = video_len
        
        # create visual head
        if self.temporal_fusion == 'mean':
            self.fusion_layer = None
        elif self.temporal_fusion == 'conv_1d':
            self.fusion_layer = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                         kernel_size=3, stride=1, padding=0, groups=self.embed_dim, bias=False)
        elif self.temporal_fusion == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=transformer_nhead, dim_feedforward=transformer_dim_feedforward, activation='gelu', batch_first=True)
            norm_layer = nn.LayerNorm(self.embed_dim)
            self.fusion_layer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=transformer_nlayer, norm=norm_layer)
            self.frame_position_embedding = nn.Embedding(self.video_len, self.embed_dim)
            self.register_buffer('position_ids', torch.arange(self.video_len, dtype=torch.long))
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, video_len, embed_dim]
        if self.temporal_fusion == 'mean':
            pass
        elif self.temporal_fusion == 'conv_1d':
            x = x.permute(0, 2, 1)
            x = self.fusion_layer(x)
            x = x.permute(0, 2, 1)
        elif self.temporal_fusion == 'transformer':
            pos = self.frame_position_embedding(self.position_ids)
            # FIXME works in torch 1.11 but raises error in torch 1.12. Seems to be bugs in AMP.
            x = self.fusion_layer(x + pos) + x
        return x.mean(dim=1)


class EmotionCLIP(nn.Module):
    def __init__(
        self,
        backbone_config: str = '/home/minxiao/workspace/EmoAsst/model/model_configs/ViT-B-32.json',
        backbone_checkpoint: Optional[str] = None,
        temporal_fusion: str = 'transformer',
        video_len: int = 8,
        head_nlayer: Optional[int] = 6,
        reset_logit_scale: bool = False,
    ):
        assert osp.isfile(backbone_config), f'{backbone_config} not found.'
        if backbone_checkpoint:
            assert osp.isfile(backbone_checkpoint), f'{backbone_checkpoint} not found.'
        assert temporal_fusion in ('mean', 'conv_1d', 'transformer'), f'invalid argument {temporal_fusion=}'
        
        super().__init__()
        self.temporal_fusion = temporal_fusion
        self.video_len = video_len
        self.head_nlayer = head_nlayer

         # init backbone
        with open(backbone_config, mode='r') as f:
            clip_config = json.load(f)
        self.backbone = CLIP(**clip_config)

        # load pretrained weights
        if backbone_checkpoint:
            self.backbone.load_state_dict(torch.load(backbone_checkpoint), strict=False)

        # create visual head
        self.visual_head = VisualHead(
            temporal_fusion=temporal_fusion, 
            embed_dim=self.backbone.visual.output_dim, 
            video_len=video_len,
            transformer_nlayer=head_nlayer
        )

        # create audio model
        self.audio_model = AudioHead()

        self.logit_scale = self.backbone.logit_scale
        if reset_logit_scale:
            nn.init.constant_(self.logit_scale, np.log(1 / 0.07)) 

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        frames = video.reshape(-1, C, H, W)
        x = self.backbone.encode_image(frames)
        x = x.reshape(B, T, -1)
        x = self.visual_head(x)
        return x

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_image(image)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_text(text)

    def encode_audio(self, audio: torch.Tensor, other_feature: torch.Tensor) -> torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
        return self.audio_model(audio, other_feature)
    
    def forward(self, video: torch.Tensor, text: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        video: [B, T, 3, H, W]
        text: [B, 77]
        """
        # [B, C]
        video_features = self.encode_video(video)
        text_features = self.encode_text(text)
        audio_features = self.encode_audio(audio, text_features)

        # normalized features
        video_features = F.normalize(video_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        audio_features = F.normalize(audio_features, dim=1)

        return video_features, text_features, audio_features, self.logit_scale.exp()