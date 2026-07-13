# Text-only SFT smoke config for google/gemma-4-12B-it.
#
# Usage (single node, 8 GPUs):
#
#   export GEMMA4_12B_PATH=/path/to/gemma-4-12B-it
#   export ALPACA_PATH=/path/to/alpaca
#   torchrun --nproc-per-node=8 -m xtuner.v1.train.cli.sft --config ci/config/gemma4_dense12B.py
#
# The released checkpoint is a Gemma4UnifiedForConditionalGeneration checkpoint.
# XTuner loads only model.language_model.* and ignores the vision/audio tensors.
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Gemma4Dense12BConfig
from xtuner.v1.train import TrainerConfig


GEMMA4_12B_PATH = os.environ["GEMMA4_12B_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


loss_cfg = CELossConfig(mode="chunk", logit_softcap=30.0)
model_cfg = Gemma4Dense12BConfig(compile_cfg=False, lm_loss_cfg=loss_cfg)
optim_cfg = AdamWConfig(lr=6e-5)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    torch_compile=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=8192),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=8192)


trainer = TrainerConfig(
    load_from=GEMMA4_12B_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=GEMMA4_12B_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir="/tmp/gemma4_dense12B",
    seed=0,
    strict_load=False,
)
