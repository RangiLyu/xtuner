# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import nn

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.rms_norm import RMSNorm

from .dense_decoder_layer import DenseMLP


class Gemma4DecoderLayer(nn.Module):
    """Gemma 4 text decoder layer."""

    layer_scalar: torch.Tensor

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered", "gemma4"] = "gemma4",
        attention_config: MHAConfig,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
        layer_type: Literal["full_attention", "sliding_attention"],
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.self_attn = attention_config.build(
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            rope_scaling_cfg=None,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )
        self.mlp = DenseMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
            hidden_act=hidden_act,
            float8_cfg=float8_cfg,
        )

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings[self.layer_type],
            seq_ctx=seq_ctx,
        )
        hidden_states = self.post_attention_layernorm(attn_outputs["projected_output"])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states
