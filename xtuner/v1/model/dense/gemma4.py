import re
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from torch import nn
from typing_extensions import Self

from transformers.models.gemma4_unified import Gemma4UnifiedTextConfig as HFGemma4UnifiedTextConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.gemma4 import Gemma4DecoderLayer, Gemma4MLP
from xtuner.v1.module.rope import RopeParametersConfig

from .dense import Dense


class Gemma4Dense(Dense):
    def _build_decoder_layer(
        self,
        config: TransformerConfig,
        attention_config: MHAConfig,
        layer_type: Literal["full_attention", "sliding_attention"] | None,
        layer_idx: int,
    ) -> nn.Module:
        """Build a Gemma4-specific decoder layer with pre/post feedforward layernorms."""
        return Gemma4DecoderLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            mlp_bias=config.mlp_bias,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            rms_norm_type=config.rms_norm_type,
            attention_config=attention_config,
            generate_config=config.generate_config,
            float8_cfg=config.float8_cfg,
            layer_type=layer_type,
            layer_idx=layer_idx,
        )

    def to_hf_key_list(self, key: str) -> list[str]:
        # Gemma4 HF weights are stored under model.language_model.<...> while
        # XTuner's Gemma4Dense uses model.<...> directly. The deployment-dependent
        # prefix remap (model. -> model.language_model.) is handled by the config's
        # hf_key_mapping, not here.
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        else:
            return [key]


class Gemma4DenseConfig(TransformerConfig):
    use_sliding_window: bool = True
    bos_token_id: int
    max_window_layers: int | None = None
    # Gemma4 HF weights nest the language model under model.language_model.<...>.
    # This remap is deployment-dependent (a standalone Gemma4 checkpoint uses model.<...>),
    # so it belongs in hf_key_mapping, not to_hf_key_list.
    hf_key_mapping: dict[str, str] | None = {r"^model\.": "model.language_model."}

    def build(self) -> Gemma4Dense:
        return Gemma4Dense(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)

        # The HF checkpoint may be saved as either:
        # 1. Gemma4UnifiedTextConfig (the text-only config)
        # 2. Gemma4UnifiedConfig (the multi-modal config with text_config sub-config)
        if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
            hf_config = hf_config.text_config

        assert isinstance(hf_config, HFGemma4UnifiedTextConfig), (
            f"Expected HFGemma4UnifiedTextConfig, got {type(hf_config)}"
        )

        # Gemma4's rope_parameters is a dict keyed by layer type (e.g. 'sliding_attention',
        # 'full_attention'), each containing its own rope_type and rope_theta. We pick the
        # full_attention config as the primary rope config for XTuner's unified RotaryEmbedding.
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            # Prefer full_attention's rope config; fall back to sliding_attention if only that exists.
            full_attn_rope = rope_parameters.get("full_attention", rope_parameters.get("sliding_attention", {}))
            rope_theta = full_attn_rope.get("rope_theta", 10000.0)
            hf_rope_type = full_attn_rope.get("rope_type", "default")
            partial_rotary_factor = full_attn_rope.get("partial_rotary_factor", 1.0)
        else:
            rope_theta = 10000.0
            hf_rope_type = "default"
            partial_rotary_factor = 1.0

        # Gemma4 uses 'proportional' rope type for full attention layers, which XTuner's
        # RotaryEmbedding doesn't directly support. Map it to 'default' and rely on
        # partial_rotary_factor for the proportional behavior.
        rope_type = "default" if hf_rope_type == "proportional" else hf_rope_type

        rope_parameters_cfg = RopeParametersConfig(
            rope_theta=rope_theta,
            rope_type=rope_type,
            partial_rotary_factor=partial_rotary_factor,
        )

        # Gemma4 uses sliding_window attention on some layers and full attention on others.
        # The layer_types field tells us which layers are which.
        layer_types = getattr(hf_config, "layer_types", None)
        use_sliding_window = layer_types is not None and "sliding_attention" in set(layer_types)

        # Determine max_window_layers: the number of sliding attention layers before the first full attention
        max_window_layers = None
        if use_sliding_window and layer_types is not None:
            for idx, lt in enumerate(layer_types):
                if lt == "full_attention":
                    max_window_layers = idx
                    break

        # Gemma4 uses a global_head_dim for full attention layers that differs from head_dim.
        # XTuner's MHA uses a single head_dim for all layers, so we use the standard head_dim.
        head_dim = getattr(hf_config, "head_dim", 256)

        # Gemma4 uses qk_norm (q_norm and k_norm in the attention)
        qk_norm = True

        # Gemma4 uses gelu_pytorch_tanh as the hidden activation
        hidden_act = getattr(hf_config, "hidden_activation", "gelu_pytorch_tanh")

        # Map gelu_pytorch_tanh to xtuner's expected act name
        if hidden_act == "gelu_pytorch_tanh":
            hidden_act = "gelu"

        config = cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            pad_token_id=getattr(hf_config, "pad_token_id"),
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            num_hidden_layers=hf_config.num_hidden_layers,
            max_window_layers=max_window_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            rms_norm_type="zero_centered",  # Gemma4 uses zero-centered RMSNorm
            rope_parameters_cfg=rope_parameters_cfg,
            hidden_act=hidden_act,
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=head_dim,
                sliding_window=getattr(hf_config, "sliding_window", 1024),
                qk_norm=qk_norm,
                qkv_bias=getattr(hf_config, "attention_bias", False),
                rms_norm_eps=hf_config.rms_norm_eps,
                rms_norm_type="zero_centered",
            ),
            use_sliding_window=use_sliding_window,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )

        return config

    @property
    def hf_config(self) -> HFGemma4UnifiedTextConfig:
        """Check if the configuration can be saved in HuggingFace format."""
        return HFGemma4UnifiedTextConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            max_window_layers=self.max_window_layers,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_parameters=self.rope_parameters,
            hidden_activation=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_key_value_heads,
            head_dim=self.attention.head_dim,
            sliding_window=self.attention.sliding_window,
            use_sliding_window=self.use_sliding_window,
            tie_word_embeddings=self.tie_word_embeddings,
            dtype=torch.bfloat16,
        )


class Gemma4Dense2BConfig(Gemma4DenseConfig):
    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    bos_token_id: int = 2
    pad_token_id: int | None = 0
    eos_token_id: int = 1
    num_hidden_layers: int = 18
    hidden_size: int = 2048
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-06
    rms_norm_type: str = "zero_centered"
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=10000.0, rope_type="default")
    )
    hidden_act: str = "gelu"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        sliding_window=1024,
        qk_norm=True,
        qkv_bias=False,
        rms_norm_eps=1e-06,
        rms_norm_type="zero_centered",
    )
    use_sliding_window: bool = True
    tie_word_embeddings: bool = True