# Copyright (c) OpenMMLab. All rights reserved.
import math
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field
from torch import nn
from typing_extensions import Self, override

from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.base import DEFAULT_FLOAT8_CFG, TorchCompileOption, TransformerConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.gemma4 import Gemma4DecoderLayer
from xtuner.v1.module.rms_norm import RMSNorm
from xtuner.v1.module.rope import Gemma4TextRotaryEmbedding

from .dense import Dense


GEMMA4_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.gemma4.Gemma4DecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


class Gemma4ScaledWordEmbedding(nn.Embedding):
    """Embedding with Gemma 4's dtype-rounded sqrt(hidden_size) scale."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = math.sqrt(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(input_ids)
        embed_scale = torch.tensor(
            self.scalar_embed_scale,
            dtype=self.weight.dtype,
            device=embeddings.device,
        )
        return embeddings * embed_scale


class Gemma4Dense(Dense):
    config: "Gemma4DenseConfig"

    def __init__(self, config: "Gemma4DenseConfig"):
        # Trainer's backward-compatibility path may replace lm_loss_cfg before
        # building the model. Logit soft-capping is part of the Gemma 4
        # architecture, so restore it independently of that training option.
        config.lm_loss_cfg.logit_softcap = config.final_logit_softcapping
        super().__init__(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type="gemma4")
        self._init_load_spec()

    def build_embeddings(self, config: TransformerConfig) -> nn.Embedding:
        return Gemma4ScaledWordEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def build_rotary_embedding(self, config: TransformerConfig) -> Gemma4TextRotaryEmbedding:
        with torch.device("cpu"):
            return Gemma4TextRotaryEmbedding(config)

    def build_layers(self, config: TransformerConfig) -> nn.ModuleDict:
        assert isinstance(config, Gemma4DenseConfig)
        layers = nn.ModuleDict()
        for layer_idx, layer_type in enumerate(config.layers_type):
            attention_config = config.attention if layer_type == "sliding_attention" else config.full_attention
            layers[str(layer_idx)] = Gemma4DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mlp_bias=config.mlp_bias,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                rms_norm_type="gemma4",
                attention_config=attention_config,
                generate_config=config.generate_config,
                float8_cfg=config.float8_cfg,
                layer_type=layer_type,
                layer_idx=layer_idx,
            )
        return layers

    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if key.startswith("norm."):
            key = key.replace("norm.", "model.norm.")
        return [key]

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return GEMMA4_COMPILE_CFG


class Gemma4DenseConfig(TransformerConfig):
    """XTuner text-only configuration for Gemma 4 Unified dense models."""

    bos_token_id: int | None = 2
    initializer_range: float = 0.02
    use_cache: bool = True
    use_bidirectional_attention: Literal["all", "vision"] | None = "vision"
    attention_k_eq_v: bool = True
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int | None = None
    enable_moe_block: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None
    final_logit_softcapping: float | None = None
    lm_loss_cfg: CELossConfig = CELossConfig()
    gemma4_layer_types: list[Literal["full_attention", "sliding_attention"]]
    sliding_rope_parameters: dict[str, Any] = Field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 10_000.0}
    )
    full_rope_parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        }
    )
    full_attention: MHAConfig

    def model_post_init(self, _context: Any) -> None:
        if len(self.gemma4_layer_types) != self.num_hidden_layers:
            raise ValueError("gemma4_layer_types must contain one entry per decoder layer.")
        if self.num_kv_shared_layers != 0:
            raise ValueError("Gemma 4 KV sharing is not supported by the text-only XTuner port yet.")
        if self.use_double_wide_mlp:
            raise ValueError("Gemma 4 double-wide MLP layers are not supported by the dense text-only port.")
        if self.use_bidirectional_attention == "all":
            raise ValueError("The text-only Gemma 4 port supports causal attention only.")
        if self.enable_moe_block:
            raise ValueError("Gemma 4 MoE blocks are outside the dense text-only port.")

    def build(self) -> Gemma4Dense:
        return Gemma4Dense(self)

    @property
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention"]]:
        return self.gemma4_layer_types

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        try:
            from transformers import AutoConfig
            from transformers.models.gemma4_unified import (
                Gemma4UnifiedConfig,
                Gemma4UnifiedTextConfig,
            )
        except ImportError as exc:
            raise RuntimeError("Gemma 4 requires a transformers version that provides gemma4_unified.") from exc

        source_config = AutoConfig.from_pretrained(hf_path)
        is_unified_checkpoint = isinstance(source_config, Gemma4UnifiedConfig)
        hf_config = source_config.text_config if is_unified_checkpoint else source_config
        if not isinstance(hf_config, Gemma4UnifiedTextConfig):
            raise TypeError(f"Expected Gemma4UnifiedTextConfig, got {type(hf_config)!r}.")

        rope_parameters = hf_config.rope_parameters
        if not isinstance(rope_parameters, dict):
            raise ValueError("Gemma 4 text config must define per-layer-type rope_parameters.")

        full_num_key_value_heads = (
            hf_config.num_global_key_value_heads if hf_config.attention_k_eq_v else hf_config.num_key_value_heads
        )
        if full_num_key_value_heads is None:
            full_num_key_value_heads = hf_config.num_key_value_heads

        return cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            eos_token_id=hf_config.eos_token_id,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            num_hidden_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            rms_norm_type="default",
            hidden_act=hf_config.hidden_activation,
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=hf_config.head_dim,
                dropout=hf_config.attention_dropout,
                qkv_bias=hf_config.attention_bias,
                qk_norm=True,
                v_norm=True,
                rms_norm_eps=hf_config.rms_norm_eps,
                rms_norm_type="gemma4",
                scaling=1.0,
                sliding_window=hf_config.sliding_window,
            ),
            full_attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=full_num_key_value_heads,
                head_dim=hf_config.global_head_dim or hf_config.head_dim,
                dropout=hf_config.attention_dropout,
                qkv_bias=hf_config.attention_bias,
                qk_norm=True,
                v_norm=True,
                rms_norm_eps=hf_config.rms_norm_eps,
                rms_norm_type="gemma4",
                scaling=1.0,
                k_eq_v=hf_config.attention_k_eq_v,
            ),
            tie_word_embeddings=hf_config.tie_word_embeddings,
            model_type="gemma4_unified_text",
            initializer_range=hf_config.initializer_range,
            use_cache=hf_config.use_cache,
            use_bidirectional_attention=hf_config.use_bidirectional_attention,
            attention_k_eq_v=hf_config.attention_k_eq_v,
            num_kv_shared_layers=hf_config.num_kv_shared_layers,
            use_double_wide_mlp=hf_config.use_double_wide_mlp,
            hidden_size_per_layer_input=getattr(hf_config, "hidden_size_per_layer_input", 0),
            vocab_size_per_layer_input=getattr(hf_config, "vocab_size_per_layer_input", hf_config.vocab_size),
            enable_moe_block=getattr(hf_config, "enable_moe_block", False),
            num_experts=getattr(hf_config, "num_experts", None),
            top_k_experts=getattr(hf_config, "top_k_experts", None),
            moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", None),
            final_logit_softcapping=hf_config.final_logit_softcapping,
            gemma4_layer_types=list(hf_config.layer_types),
            sliding_rope_parameters=dict(rope_parameters["sliding_attention"]),
            full_rope_parameters=dict(rope_parameters["full_attention"]),
            lm_loss_cfg=CELossConfig(logit_softcap=hf_config.final_logit_softcapping),
            hf_key_mapping={r"^model\.": "model.language_model."} if is_unified_checkpoint else None,
        )

    @property
    def hf_config(self):
        try:
            from transformers.models.gemma4_unified import Gemma4UnifiedTextConfig
        except ImportError as exc:
            raise RuntimeError("Gemma 4 requires a transformers version that provides gemma4_unified.") from exc

        return Gemma4UnifiedTextConfig(
            architectures=["Gemma4UnifiedForCausalLM"],
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_key_value_heads,
            head_dim=self.attention.head_dim,
            hidden_activation=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_parameters={
                "sliding_attention": self.sliding_rope_parameters,
                "full_attention": self.full_rope_parameters,
            },
            attention_bias=self.attention.qkv_bias,
            attention_dropout=self.attention.dropout,
            sliding_window=self.attention.sliding_window,
            layer_types=self.gemma4_layer_types,
            final_logit_softcapping=self.final_logit_softcapping,
            use_bidirectional_attention=self.use_bidirectional_attention,
            num_global_key_value_heads=self.full_attention.num_key_value_heads,
            global_head_dim=self.full_attention.head_dim,
            attention_k_eq_v=self.attention_k_eq_v,
            num_kv_shared_layers=self.num_kv_shared_layers,
            use_double_wide_mlp=self.use_double_wide_mlp,
            hidden_size_per_layer_input=self.hidden_size_per_layer_input,
            vocab_size_per_layer_input=self.vocab_size_per_layer_input,
            enable_moe_block=self.enable_moe_block,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            moe_intermediate_size=self.moe_intermediate_size,
            dtype=torch.bfloat16,
        )


class Gemma4Dense12BConfig(Gemma4DenseConfig):
    vocab_size: int = 262_144
    max_position_embeddings: int = 262_144
    eos_token_id: int = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    num_hidden_layers: int = 48
    hidden_size: int = 3840
    intermediate_size: int = 15_360
    rms_norm_eps: float = 1e-6
    rms_norm_type: Literal["default", "zero_centered"] = "default"
    hidden_act: str = "gelu_pytorch_tanh"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        qk_norm=True,
        v_norm=True,
        rms_norm_eps=1e-6,
        rms_norm_type="gemma4",
        scaling=1.0,
        sliding_window=1024,
    )
    full_attention: MHAConfig = MHAConfig(
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=512,
        qk_norm=True,
        v_norm=True,
        rms_norm_eps=1e-6,
        rms_norm_type="gemma4",
        scaling=1.0,
        k_eq_v=True,
    )
    tie_word_embeddings: bool = True
    model_type: str | None = "gemma4_unified_text"
    gemma4_layer_types: list[Literal["full_attention", "sliding_attention"]] = Field(
        default_factory=lambda: [
            "sliding_attention" if (layer_idx + 1) % 6 else "full_attention" for layer_idx in range(48)
        ]
    )
    vocab_size_per_layer_input: int | None = 262_144
    final_logit_softcapping: float | None = 30.0
    lm_loss_cfg: CELossConfig = CELossConfig(logit_softcap=30.0)
    hf_key_mapping: dict[str, str] | None = {r"^model\.": "model.language_model."}
