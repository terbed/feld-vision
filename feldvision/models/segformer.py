from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from feldvision.config import ModelConfig
from feldvision.models.registry import register_model


def segformer_b2_config(num_classes: int) -> SegformerConfig:
    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        decoder_hidden_size=768,
        semantic_loss_ignore_index=255,
        num_labels=num_classes,
    )


@dataclass(frozen=True)
class SegformerComponents:
    encoder: SegformerModel
    decode_head: SegformerDecodeHead
    config: SegformerConfig


def _components(
    model_config: ModelConfig,
    num_classes: int,
    hf_config: SegformerConfig | None,
) -> SegformerComponents:
    if model_config.pretrained:
        pretrained = SegformerForSemanticSegmentation.from_pretrained(
            model_config.pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        return SegformerComponents(
            encoder=pretrained.segformer,
            decode_head=pretrained.decode_head,
            config=pretrained.config,
        )
    config = copy.deepcopy(hf_config) if hf_config is not None else segformer_b2_config(num_classes)
    config.num_labels = num_classes
    return SegformerComponents(
        encoder=SegformerModel(config),
        decode_head=SegformerDecodeHead(config),
        config=config,
    )


def _hidden_states(encoder: SegformerModel, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
    output = encoder(image, output_hidden_states=True, return_dict=True)
    if output.hidden_states is None:
        raise RuntimeError("SegFormer encoder did not return stage hidden states")
    return output.hidden_states


def _resize_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


class SegformerSingleStream(nn.Module):
    def __init__(self, components: SegformerComponents) -> None:
        super().__init__()
        self.encoder = components.encoder
        self.decode_head = components.decode_head

    def forward(
        self,
        detail: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        logits = self.decode_head(_hidden_states(self.encoder, detail))
        return _resize_logits(logits, detail)


class SegformerDualStream(nn.Module):
    def __init__(
        self,
        components: SegformerComponents,
        *,
        shared_encoder: bool,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        self.detail_encoder = components.encoder
        self.context_encoder = None if shared_encoder else copy.deepcopy(components.encoder)
        self.decode_head = components.decode_head
        self.fusion = nn.ModuleList(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1)
            for hidden_size in components.config.hidden_sizes
        )

    def forward(
        self,
        detail: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("dual-stream SegFormer requires a context tensor")
        detail_features = _hidden_states(self.detail_encoder, detail)
        context_encoder = self.detail_encoder if self.shared_encoder else self.context_encoder
        if context_encoder is None:
            raise RuntimeError("separate context encoder is not initialized")
        context_features = _hidden_states(context_encoder, context)
        fused = tuple(
            projection(torch.cat((detail_feature, context_feature), dim=1))
            for projection, detail_feature, context_feature in zip(
                self.fusion,
                detail_features,
                context_features,
                strict=True,
            )
        )
        logits = self.decode_head(fused)
        return _resize_logits(logits, detail)


@register_model("segformer_b2_single")
def build_single(
    *,
    config: ModelConfig,
    num_classes: int,
    hf_config: SegformerConfig | None = None,
) -> nn.Module:
    return SegformerSingleStream(_components(config, num_classes, hf_config))


@register_model("segformer_b2_shared_context")
def build_shared_context(
    *,
    config: ModelConfig,
    num_classes: int,
    hf_config: SegformerConfig | None = None,
) -> nn.Module:
    return SegformerDualStream(
        _components(config, num_classes, hf_config),
        shared_encoder=True,
    )


@register_model("segformer_b2_separate_context")
def build_separate_context(
    *,
    config: ModelConfig,
    num_classes: int,
    hf_config: SegformerConfig | None = None,
) -> nn.Module:
    return SegformerDualStream(
        _components(config, num_classes, hf_config),
        shared_encoder=False,
    )
