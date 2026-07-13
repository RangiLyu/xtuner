# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class Gemma4TextRotaryEmbedding(nn.Module):
    """The two RoPE profiles used by Gemma 4 text layers."""

    def __init__(self, config, device: torch.device | str | None = None):
        super().__init__()
        hf_config = config.hf_config
        self.max_seq_len_cached = hf_config.max_position_embeddings
        self.original_max_seq_len = hf_config.max_position_embeddings
        self.layer_types = set(hf_config.layer_types)

        for layer_type in self.layer_types:
            rope_params = hf_config.rope_parameters[layer_type]
            rope_type = rope_params["rope_type"]
            if rope_type == "default":
                rope_init_fn = self._compute_default_rope_parameters
            else:
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]

            rope_init_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_kwargs["head_dim_key"] = "global_head_dim"

            inv_freq, attention_scaling = rope_init_fn(hf_config, **rope_init_kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def _compute_default_rope_parameters(
        config,
        device: torch.device | str | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = config.head_dim or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def _forward_profile(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {layer_type: self._forward_profile(x, position_ids, layer_type) for layer_type in self.layer_types}
