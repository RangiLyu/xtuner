# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import nn

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rms_norm import RMSNorm


class Gemma4MLP(nn.Module):
    """Gemma4 MLP with SwiGLU activation."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        hidden_act: str,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.gate_proj = build_linear(hidden_size, intermediate_size, bias=bias, float8_cfg=float8_cfg)
        self.up_proj = build_linear(hidden_size, intermediate_size, bias=bias, float8_cfg=float8_cfg)
        self.down_proj = build_linear(intermediate_size, hidden_size, bias=bias, float8_cfg=float8_cfg)
        self.act_fn = _get_act_fn(hidden_act)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def _get_act_fn(hidden_act: str):
    """Get activation function for Gemma4 MLP.

    Gemma4 uses gelu_pytorch_tanh, which maps to torch.nn.functional.gelu with tanh.
    """
    from xtuner.v1.ops.act_fn import get_act_fn

    if hidden_act == "gelu_pytorch_tanh":
        hidden_act = "gelu"
    return get_act_fn(hidden_act)


class Gemma4DecoderLayer(nn.Module):
    """Gemma4-specific decoder layer.

    Gemma4 uses a unique post-attention + pre/post-feedforward layernorm structure
    with a per-layer scalar, different from the standard Llama-style decoder layer:

    1. input_layernorm -> self_attn -> post_attention_layernorm -> +residual
    2. pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> +residual
    3. output *= layer_scalar
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered"] = "default",
        attention_config: MHAConfig,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # Attention block
        self.self_attn = attention_config.build(
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            rope_scaling_cfg=None,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

        # Feedforward block with pre/post layernorms (Gemma4-specific)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.mlp = Gemma4MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
            hidden_act=hidden_act,
            float8_cfg=float8_cfg,
        )
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

        # Per-layer scalar (Gemma4-specific)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
        hidden_states = attn_outputs["projected_output"]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feedforward block (Gemma4-specific: pre + post layernorms)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Apply layer scalar (Gemma4-specific)
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states