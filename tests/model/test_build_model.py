from unittest import TestCase

from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE


class TestBuildModel(TestCase):
    def test_build_mha_with_shared_kv_and_weightless_v_norm(self):
        attention = MHAConfig(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            qk_norm=True,
            v_norm=True,
            k_eq_v=True,
            scaling=1.0,
            attn_impl="eager_attention",
        ).build(hidden_size=8, layer_type="full_attention")

        self.assertIsNone(attention.v_proj)
        self.assertNotIn("v_norm.weight", attention.state_dict())
        self.assertEqual(attention.scaling, 1.0)

    def test_build_moe(self):
        from xtuner.v1.model import Qwen3MoEConfig

        router_config = GreedyRouterConfig(
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        )
        attention_config = MHAConfig(
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            qk_norm=True,
        )
        config = Qwen3MoEConfig(
            vocab_size=151936,
            max_position_embeddings=4096,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_hidden_layers=48,
            hidden_size=2048,
            intermediate_size=6144,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            n_routed_experts=128,
            n_shared_experts=0,
            num_experts_per_tok=8,
            first_k_dense_replace=0,
            hidden_factor=1.0,
            moe_intermediate_size=768,
            router=router_config,
        )
        model = config.build()
        self.assertIsInstance(model, Qwen3MoE)
