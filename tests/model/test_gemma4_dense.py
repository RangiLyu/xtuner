import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from packaging.version import Version
from safetensors.torch import save_file

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import __version__ as transformers_version
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Gemma4Dense12BConfig, get_model_config, get_model_config_from_hf
from xtuner.v1.model.dense.gemma4 import Gemma4DenseConfig
from xtuner.v1.module.attention import MHAConfig


GEMMA4_12B_PATH = os.getenv("GEMMA4_12B_PATH")
HAS_GEMMA4 = Version(transformers_version) >= Version("5.13.0")

if GEMMA4_12B_PATH and torch.cuda.is_available():
    from xtuner._testing import DeterministicDDPTestCase
else:
    DeterministicDDPTestCase = unittest.TestCase


def _tiny_config(*, unified_checkpoint: bool = False) -> Gemma4DenseConfig:
    return Gemma4DenseConfig(
        vocab_size=64,
        max_position_embeddings=128,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        attention=MHAConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            qk_norm=True,
            v_norm=True,
            rms_norm_eps=1e-6,
            rms_norm_type="gemma4",
            scaling=1.0,
            sliding_window=4,
            attn_impl="eager_attention",
        ),
        full_attention=MHAConfig(
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            qk_norm=True,
            v_norm=True,
            rms_norm_eps=1e-6,
            rms_norm_type="gemma4",
            scaling=1.0,
            k_eq_v=True,
            attn_impl="eager_attention",
        ),
        tie_word_embeddings=True,
        model_type="gemma4_unified_text",
        gemma4_layer_types=["sliding_attention", "full_attention"],
        vocab_size_per_layer_input=64,
        final_logit_softcapping=30.0,
        lm_loss_cfg=CELossConfig(logit_softcap=30.0),
        compile_cfg=False,
        hf_load_key_mapping={r"^model\.": "model.language_model."} if unified_checkpoint else None,
    )


@unittest.skipUnless(HAS_GEMMA4, "transformers >= 5.13.0 is required for Gemma 4")
class TestGemma4DenseCPU(unittest.TestCase):
    def test_12b_config_meta_shapes_and_dispatch(self):
        from transformers.models.gemma4_unified import Gemma4UnifiedConfig

        cfg = Gemma4Dense12BConfig(compile_cfg=False)
        self.assertIsInstance(get_model_config("gemma4-12b"), Gemma4Dense12BConfig)
        self.assertEqual(
            cfg.gemma4_layer_types,
            ["sliding_attention" if (layer_idx + 1) % 6 else "full_attention" for layer_idx in range(48)],
        )
        self.assertEqual(
            [i for i, layer_type in enumerate(cfg.layers_type) if layer_type == "full_attention"],
            list(range(5, 48, 6)),
        )

        # Trainer's legacy loss_cfg override must not remove an architectural
        # softcap from the model or its training loss path.
        cfg.lm_loss_cfg = CELossConfig()
        with torch.device("meta"):
            model = cfg.build()
        self.assertEqual(cfg.lm_loss_cfg.logit_softcap, 30.0)
        self.assertEqual(model.lm_head.logit_softcap, 30.0)

        state = model.state_dict()
        self.assertEqual(state["layers.0.self_attn.q_proj.weight"].shape, (4096, 3840))
        self.assertEqual(state["layers.0.self_attn.k_proj.weight"].shape, (2048, 3840))
        self.assertEqual(state["layers.0.self_attn.v_proj.weight"].shape, (2048, 3840))
        self.assertEqual(state["layers.5.self_attn.q_proj.weight"].shape, (8192, 3840))
        self.assertEqual(state["layers.5.self_attn.k_proj.weight"].shape, (512, 3840))
        self.assertNotIn("layers.5.self_attn.v_proj.weight", state)
        self.assertIsNone(model.layers["5"].self_attn.v_proj)
        self.assertNotIn("layers.5.self_attn.v_norm.weight", state)
        self.assertEqual(len({spec.hf_keys[0] for spec in model.load_spec_mapping.values()}), 666)
        self.assertEqual(len({spec.load_hf_keys[0] for spec in model.load_spec_mapping.values()}), 666)
        self.assertEqual(model.load_spec_mapping["embed_tokens.weight"].hf_keys, ["model.embed_tokens.weight"])
        self.assertEqual(
            model.load_spec_mapping["embed_tokens.weight"].load_hf_keys,
            ["model.language_model.embed_tokens.weight"],
        )

        hf_config = cfg.hf_config
        self.assertEqual(hf_config.architectures, ["Gemma4UnifiedForCausalLM"])
        self.assertEqual(hf_config.layer_types, cfg.gemma4_layer_types)
        self.assertEqual(hf_config.global_head_dim, 512)
        self.assertEqual(hf_config.num_global_key_value_heads, 1)
        self.assertEqual(hf_config.final_logit_softcapping, 30.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            top_config = Gemma4UnifiedConfig(text_config=hf_config)
            top_config.save_pretrained(tmpdir)
            dispatched = get_model_config_from_hf(Path(tmpdir))
        self.assertIsInstance(dispatched, Gemma4DenseConfig)
        self.assertEqual(dispatched.hf_load_key_mapping, {r"^model\.": "model.language_model."})

    def test_scaled_embedding_and_rope_bitwise(self):
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedForCausalLM,
            Gemma4UnifiedTextRotaryEmbedding,
        )

        cfg = _tiny_config()
        hf_config = cfg.hf_config
        hf_config._attn_implementation = "eager"
        torch.manual_seed(3)
        hf_model = Gemma4UnifiedForCausalLM(hf_config)
        xtuner_model = cfg.build()
        xtuner_model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight)

        input_ids = torch.tensor([[2, 11, 12, 13]])
        torch.testing.assert_close(
            xtuner_model.embed_tokens(input_ids),
            hf_model.model.embed_tokens(input_ids),
            rtol=0.0,
            atol=0.0,
        )

        x = torch.randn(1, 7, cfg.hidden_size, dtype=torch.bfloat16)
        position_ids = torch.arange(7).unsqueeze(0)
        actual = xtuner_model.rotary_emb(x, position_ids)
        expected_rotary = Gemma4UnifiedTextRotaryEmbedding(hf_config)
        for layer_type in ("sliding_attention", "full_attention"):
            expected = expected_rotary(x, position_ids, layer_type)
            torch.testing.assert_close(actual[layer_type][0], expected[0], rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual[layer_type][1], expected[1], rtol=0.0, atol=0.0)

    def test_tiny_forward_backward_bitwise(self):
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM

        with mock.patch.dict(os.environ, {"XTUNER_HF_IMPL": "true"}):
            cfg = _tiny_config()
            hf_config = cfg.hf_config
            hf_config._attn_implementation = "eager"
            torch.manual_seed(7)
            hf_model = Gemma4UnifiedForCausalLM(hf_config).eval()
            xtuner_model = cfg.build().eval()

            with torch.no_grad():
                hf_state = hf_model.state_dict()
                for name, tensor in xtuner_model.state_dict().items():
                    tensor.copy_(hf_state[xtuner_model.to_hf_key_list(name)[0]])

            input_ids = torch.tensor([[2, 11, 12, 13, 14, 15, 16]])
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
            expected_logits = hf_model(input_ids=input_ids, use_cache=False).logits
            actual_logits = xtuner_model(seq_ctx).logits
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)

            hf_inputs = hf_model.model.embed_tokens(input_ids).detach().requires_grad_(True)
            xtuner_inputs = xtuner_model.embed_tokens(input_ids).detach().requires_grad_(True)
            expected = hf_model(inputs_embeds=hf_inputs, use_cache=False).logits
            actual = xtuner_model(seq_ctx.copy(input_ids=None, inputs_embeds=xtuner_inputs)).logits
            expected.sum().backward()
            actual.sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            torch.testing.assert_close(xtuner_inputs.grad, hf_inputs.grad, rtol=0.0, atol=0.0)

    def test_unified_load_and_standalone_strict_reload(self):
        from transformers.models.gemma4_unified import Gemma4UnifiedConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM

        cfg = _tiny_config(unified_checkpoint=True)
        torch.manual_seed(11)
        source_model = cfg.build()
        with torch.no_grad():
            for tensor in source_model.state_dict().values():
                if tensor.is_floating_point():
                    tensor.uniform_(-0.1, 0.1)
        expected_state = {name: tensor.detach().clone() for name, tensor in source_model.state_dict().items()}
        checkpoint = {
            source_model.load_spec_mapping[name].load_hf_keys[0]: tensor.clone()
            for name, tensor in expected_state.items()
        }
        checkpoint["model.visual.placeholder"] = torch.ones(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "unified"
            output_dir = tmpdir / "text"
            source_dir.mkdir()
            Gemma4UnifiedConfig(text_config=cfg.hf_config).save_pretrained(source_dir)
            save_file(checkpoint, source_dir / "model.safetensors")

            receiver = cfg.build()
            loaded, unloaded, missing = receiver.from_hf(source_dir, strict=False)
            self.assertEqual(len(loaded), len(expected_state))
            self.assertFalse(unloaded)
            self.assertFalse(missing)
            for name, tensor in receiver.state_dict().items():
                torch.testing.assert_close(tensor, expected_state[name], rtol=0.0, atol=0.0)

            receiver.save_hf(output_dir)
            exported_config = AutoConfig.from_pretrained(output_dir)
            self.assertEqual(exported_config.model_type, "gemma4_unified_text")
            self.assertEqual(exported_config.architectures, ["Gemma4UnifiedForCausalLM"])
            hf_reloaded = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True)
            self.assertIsInstance(hf_reloaded, Gemma4UnifiedForCausalLM)

            reload_config = Gemma4DenseConfig.from_hf(output_dir)
            reload_config.compile_cfg = False
            self.assertIsNone(reload_config.hf_load_key_mapping)
            reloaded = reload_config.build()
            _, unloaded, missing = reloaded.from_hf(output_dir, strict=True)
            self.assertFalse(unloaded)
            self.assertFalse(missing)
            for name, tensor in reloaded.state_dict().items():
                torch.testing.assert_close(tensor, expected_state[name], rtol=0.0, atol=0.0)


@unittest.skipUnless(
    HAS_GEMMA4 and GEMMA4_12B_PATH and torch.cuda.is_available(),
    "GEMMA4_12B_PATH and CUDA are required for Gemma 4 checkpoint parity",
)
class TestGemma4DenseCheckpoint(DeterministicDDPTestCase):
    def test_sliding_decoder_layer_bitwise_parity(self):
        self._check_decoder_layer_bitwise_parity("cuda", 0)

    def test_full_decoder_layer_bitwise_parity(self):
        self._check_decoder_layer_bitwise_parity("cuda", 5)

    def _check_decoder_layer_bitwise_parity(self, device, layer_idx):
        from transformers.models.gemma4_unified import Gemma4UnifiedConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )
        from xtuner.v1.utils import HFCheckpointLoader

        self.create_pg(device)
        with self.hf_impl():
            loader = HFCheckpointLoader(GEMMA4_12B_PATH)
            with torch.device("meta"):
                cfg = Gemma4DenseConfig.from_hf(GEMMA4_12B_PATH)
                cfg.compile_cfg = False
                model = cfg.build()
            xtuner_layer = model.layers[str(layer_idx)]
            self.materialize_submodule(model, xtuner_layer, loader)
            self.materialize_submodule(model, model.norm, loader)
            self.materialize_submodule(model, model.lm_head, loader)
            xtuner_layer.layer_scalar.copy_(
                loader.load(f"model.language_model.layers.{layer_idx}.layer_scalar").to("cuda")
            )
            model.rotary_emb.to("cuda")

            hf_config = Gemma4UnifiedConfig.from_pretrained(GEMMA4_12B_PATH)
            hf_config._attn_implementation = "eager"
            hf_config.text_config._attn_implementation = "eager"
            with torch.device("meta"):
                hf_model = Gemma4UnifiedForConditionalGeneration(hf_config).eval()
            hf_layer = hf_model.model.language_model.layers[layer_idx]
            hf_norm = hf_model.model.language_model.norm
            hf_lm_head = hf_model.lm_head
            self.materialize_submodule(hf_model, hf_layer, loader)
            self.materialize_submodule(hf_model, hf_norm, loader)
            self.materialize_submodule(hf_model, hf_lm_head, loader)
            hf_layer.layer_scalar.copy_(
                loader.load(f"model.language_model.layers.{layer_idx}.layer_scalar").to("cuda")
            )

            seq_len = 16
            input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda")
            seq_ctx = SequenceContext.from_input_ids((input_ids,))
            position_embeddings = model.rotary_emb(
                torch.empty(1, seq_len, cfg.hidden_size, device="cuda", dtype=torch.bfloat16),
                seq_ctx.position_ids,
            )
            layer_type = cfg.layers_type[layer_idx]
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device="cuda", dtype=torch.bfloat16),
                diagonal=1,
            )[None, None]
            labels = torch.randint(0, cfg.vocab_size, (seq_len,), device="cuda")
            base = torch.randn(1, seq_len, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

            hf_inputs = base.clone().requires_grad_(True)
            expected_output = hf_layer(
                hf_inputs,
                shared_kv_states={},
                position_embeddings=position_embeddings[layer_type],
                attention_mask=attention_mask,
            )
            expected_logits = hf_lm_head(hf_norm(expected_output))
            expected_logits = torch.tanh(expected_logits / 30.0) * 30.0
            expected_loss = F.cross_entropy(expected_logits.float().reshape(-1, cfg.vocab_size), labels)
            expected_loss.backward()

            xtuner_inputs = base.clone().requires_grad_(True)
            actual_output = xtuner_layer(xtuner_inputs, position_embeddings, seq_ctx)
            _, (actual_logits, _) = model.lm_head(model.norm(actual_output))
            actual_loss = F.cross_entropy(actual_logits.reshape(-1, cfg.vocab_size), labels)
            actual_loss.backward()

        torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(xtuner_inputs.grad, hf_inputs.grad, rtol=0.0, atol=0.0)
