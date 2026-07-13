# Gemma 4 text-tower integration design

## 1. Scope and source of truth

Target checkpoint: [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it).

- Top-level architecture: `Gemma4UnifiedForConditionalGeneration`.
- Top-level `model_type`: `gemma4_unified`.
- Text sub-config `model_type`: `gemma4_unified_text`.
- Reference implementation: Transformers
  `models/gemma4_unified/modeling_gemma4_unified.py` (validated with Transformers
  5.13.0 in the user-provided environment).
- Checkpoint dtype: bf16, suitable for training.

The first integration supports only the language tower. Vision/audio embedding,
multimodal token replacement, and full compose-model parity are deferred. The text
tower must nevertheless load its weights from the released conditional-generation
checkpoint.

## 2. Review of commit `94b6cc80`

The draft has the correct high-level bucket (Dense text tower) and correctly notices
the four-norm decoder block and persistent `layer_scalar`. It is not yet loadable or
numerically equivalent to the 12B reference.

### Blocking architecture gaps

1. `get_model_config_from_hf` dispatches only `gemma4_unified_text`, while the released
   checkpoint reports `gemma4_unified`. Calling the entry point on the target repo
   currently raises `ValueError: Unsupported model type: gemma4_unified`.
2. The exact `layer_types` list is periodic (`sliding` layers 0-4, `full` layer 5,
   repeated every six layers). The draft converts it to a single boundary and produces
   `[full x5, sliding x43]`, so every full-attention layer is assigned the wrong profile.
3. The two profiles have different shapes and behavior:

   | Property | Sliding attention | Full attention |
   |---|---:|---:|
   | query heads | 16 | 16 |
   | head dim | 256 | 512 (`global_head_dim`) |
   | KV heads | 8 | 1 (`num_global_key_value_heads`) |
   | K/V projection | separate | K is reused as V (`attention_k_eq_v`) |
   | attention scale | 1.0 | 1.0 |
   | RoPE | default, theta 10,000, full head | proportional, theta 1,000,000, first 25% |

   The draft uses one `MHAConfig` (`head_dim=256`, 8 KV heads, separate V, scale
   `head_dim**-0.5`) for all layers. Comparing meta models finds 40 parameter-shape
   mismatches and eight XTuner `v_proj` weights that do not exist in HF.
4. HF normalizes Q, K, and V; V normalization has no trainable scale. Existing MHA only
   normalizes Q/K. The full-attention K=V path also needs the same unnormalized projection
   to feed two different normalization operations.
5. Gemma 4 computes two RoPE profiles. The draft keeps only the full profile, maps
   `proportional` to `default`, and passes `rope_scaling_cfg=None` into attention. This
   both loses proportional semantics and disables partial-rotary application.

### Blocking numerical gaps

- Embeddings are multiplied by `sqrt(hidden_size)` after rounding the scale to the
  embedding dtype. The draft uses a plain `nn.Embedding`.
- Gemma 4 RMSNorm uses the checkpoint weight directly (default semantics), not
  `(1 + weight)`. The draft selects `zero_centered`, changing every norm output.
- HF uses `gelu_pytorch_tanh`; the draft rewrites it to exact `gelu`.
- HF attention uses an explicit scale of `1.0`; existing XTuner MHA derives
  `head_dim**-0.5`.
- The 12B checkpoint applies final logit soft-capping with value `30.0`; it is not
  represented in the model or CE-loss path.

### Config, export, and delivery gaps

- `hf_config` does not round-trip `layer_types`, the two nested RoPE profiles,
  `global_head_dim`, `num_global_key_value_heads`, `attention_k_eq_v`,
  `attention_dropout`, `final_logit_softcapping`, or the optional KV-sharing and
  double-wide-MLP fields. It currently emits a flat invalid `rope_parameters` dict.
- The only size alias is `gemma4-2b`, but the requested checkpoint is the 12B variant
  (48 layers, hidden size 3840, intermediate size 15360).
- The draft has no design doc, parity test, save/load test, engine regression, or
  drop-in CI training config.
- `gemma4.py` imports the HF config class at module import time even though this repo's
  declared Transformers pins predate Gemma 4. The import must be lazy/version-guarded,
  or the supported dependency must be updated deliberately.

## 3. Proposed decomposition

### 3.1 Model/config seam: `xtuner/v1/model/dense/gemma4.py`

`Gemma4Dense(Dense)` remains a dense language model and owns only the behavior that the
generic Dense body cannot express:

- `build_embeddings`: build a scaled word embedding matching HF dtype rounding.
- `build_layers`: select a sliding or full `MHAConfig` from the exact per-layer
  `layer_types` list and build `Gemma4DecoderLayer`.
- `build_rotary_embedding`: build a Gemma-specific two-profile rotary module. Its
  forward returns both profiles; each decoder layer selects its own profile. This keeps
  the generic Dense forward intact.
- `to_hf_key_list`: emit standalone text-model keys (`model.*`), including tied
  embedding handling. VLM nesting is deployment metadata, not structural naming.
- Apply final logit soft-capping through a small LM-head/loss hook shared by eager and
  chunked CE so training and inference use identical logits.

`Gemma4DenseConfig(TransformerConfig)` stores all architecture-affecting fields:

- the exact `layer_types` list;
- `sliding_attention` and `full_attention` MHA configs;
- both RoPE parameter dictionaries;
- `final_logit_softcapping`, embedding scale, `attention_k_eq_v`,
  `num_kv_shared_layers`, and `use_double_wide_mlp`;
- a built-in HF config round trip using `Gemma4UnifiedTextConfig`.

`from_hf` accepts either a top-level `Gemma4UnifiedConfig` or a standalone
`Gemma4UnifiedTextConfig`, unwraps the text config when needed, and validates the model
type. `Gemma4Dense12BConfig` hard-codes the released 12B dimensions and is registered as
`gemma4-12b`.

The package dispatch accepts both `gemma4_unified` and `gemma4_unified_text`. Imports of
Gemma 4 Transformer classes are lazy and report a clear minimum-version error.

### 3.2 Module seam

`Gemma4DecoderLayer` contains:

- input, post-attention, pre-feedforward, and post-feedforward RMSNorms with default
  weight semantics;
- generic MHA configured for the layer's profile;
- a gated MLP with exact `gelu_pytorch_tanh`;
- persistent `layer_scalar`.

Extend generic MHA only with reusable primitives required by the HF implementation:

- explicit attention `scaling` override;
- optional K=V projection sharing;
- optional V RMS normalization without a trainable scale.

No Gemma model-name branch should be added to MHA or Dense. If these options make MHA
unreasonably complex, contain them in a `Gemma4Attention` subclass while still reusing
the existing attention op, projection builder, rotary op, and cache-independent
training path.

The two-profile rotary behavior stays in a Gemma-specific module. It should use the HF
`proportional` initialization formula directly and set partial-rotary application on
the full-attention module; it should not coerce `proportional` into `default` or create
the deprecated `RopeScalingConfig` merely to carry one flag.

### 3.3 Checkpoint contract

The released checkpoint stores text tensors below `model.language_model.*`, while a
standalone `Gemma4UnifiedForCausalLM` stores them below `model.*`.

Baseline contract:

1. Load the released conditional checkpoint directly with `strict=False`, selecting
   only text-tower tensors and mapping `model.* -> model.language_model.*` on load.
2. Export a standalone text-only HF checkpoint with `Gemma4UnifiedTextConfig`,
   `architectures=["Gemma4UnifiedForCausalLM"]`, and standard `model.*` tensor names.
3. Reload that exported checkpoint strictly and require byte-equal text tensors.

This requires load-prefix mapping and save-prefix mapping to be directional. Prefer a
small generalization of the load/save spec over a Gemma name check in base classes. If
that refactor is judged too invasive for the baseline, the fallback is a streaming
checkpoint-extraction script that rewrites the released language-tower keys into a
standalone text checkpoint before training (the same trade-off used by the Step-3.5
reference PR for an incompatible storage layout).

## 4. Baseline validation

Add `tests/model/test_gemma4_dense.py`, based on
`tests/model/test_qwen3_5_dense.py`, using `GEMMA4_12B_PATH`.

Required cases:

1. Config/shape test (CPU/meta): top-level dispatch, exact field mapping, exact
   `layer_types`, profile shapes, no `v_proj` on full layers, and `hf_config` round trip.
2. Scaled embedding and both RoPE profiles: bitwise against HF.
3. Decoder-layer parity for one sliding layer (0) and one full layer (5): materialize
   only the selected layer + norm + LM head; assert layer output, soft-capped CE loss,
   and `dL/dx` are bitwise under `XTUNER_HF_IMPL`.
4. Weight mapping/load: every expected text tensor maps to the same shape; full layers
   do not request `v_proj`; released checkpoint loads with only vision/audio keys left
   unused.
5. FSDP forward parity within the established tolerance.
6. Text-only `save_hf` round trip: exported config loads as
   `Gemma4UnifiedForCausalLM`, and all text tensors are byte-equal.
7. Engine regression: a short real train trace, including forward, soft-capped chunked
   CE, backward, FSDP reduce, optimizer step, and a recorded loss trajectory.

Because 12B is a large model, the mandatory bitwise guarantee is the selective
single-layer forward/backward test. Whole-model forward is an integration test only if
it fits the available GPU setup; otherwise record it as a multi-GPU manual result.

## 5. Drop-in training config

Add `ci/config/gemma4_dense12B.py`:

- `GEMMA4_12B_PATH` and dataset paths come from environment variables;
- use `Gemma4Dense12BConfig`;
- use `CELossConfig(mode="chunk")` with Gemma logit soft-capping enabled;
- start with bf16, eager, data-parallel FSDP only;
- use `strict_load=False` only for the released multimodal checkpoint; a standalone
  exported text checkpoint must load strictly.

Run about 50 steps on real hardware and record the loss curve in the PR description.

## 6. Stacked implementation plan

1. **Design/review baseline**: this document; no behavior changes.
2. **Shared primitives**: MHA scale/K=V/V-norm support and soft-capped LM loss, each with
   focused unit tests. Revert draft shared changes that are not needed after the final
   decomposition.
3. **Gemma 4 text architecture**: scaled embedding, dual RoPE, decoder, exact config,
   12B size config, key mapping, registration, and dependency/version guard.
4. **Baseline parity tests**: config/shape, both decoder profiles forward+backward,
   checkpoint loading, and standalone save/reload.
5. **Training delivery**: CI config, engine/loss trace, documentation, lint, and local
   baseline commit.
6. **Optimizations after baseline is committed**: SP, `torch.compile`, fp8, and
   activation offload, each compared with the committed bf16 eager baseline. EP is not
   applicable to this dense model. Add intra-layer micro-batch coverage if the Dense
   path supports it.
7. **Deferred compose support**: vision/audio towers, multimodal token injection, and
   full `Gemma4UnifiedForConditionalGeneration` parity.

## 7. Acceptance criteria

- Target config dispatches successfully from the released checkpoint.
- Zero parameter-name or shape mismatches for the text tower.
- Sliding and full decoder layers match HF bitwise for output, loss, and input gradient
  under the parity environment.
- Standalone text export is loadable by Transformers and byte-equal on round trip.
- The real training smoke test completes and has a plausible descending loss trace.
- Each optimization reports its numerical delta relative to the committed baseline.
