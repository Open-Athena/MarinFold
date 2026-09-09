# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave next-token training runner for exp157.

This is the regular CE/control-style path for issue #157: contacts-v1,
unmasked next-token loss, exp117-ish recipe (LR 3.162e-3, wd 0.2, cosine,
10% warmup). ``EXP157_POSITION_MODE=fixed`` replaces the learned input rows
for ``<p0>`` ... ``<p1999>`` with fixed RoPE/sinusoidal residue-location
vectors; ``EXP157_POSITION_MODE=rope_delta`` adds a zero-initialized learned
per-position residual to those vectors; ``EXP157_POSITION_MODE=learned`` is the
matched learned-embedding control.
"""

import math
import os

from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config

from fray.types import ResourceConfig

from contacts_v1_train_common import (
    CONTACTS_V1_NUM_POSITION_TOKENS,
    CONTACTS_V1_P0_TOKEN_ID,
    CONTACTS_V1_S3_PREFIX,
    CONTACTS_V1_TOKENIZER,
    PROTEIN_RESOURCES_H100,
)
from dispatch_train import dispatch_training_run
from fixed_position_model import (
    FixedResiduePositionLlamaConfig,
    FixedResiduePositionQwen3Config,
    ResiduePositionEmbeddingSpec,
)

SEQ_LEN = 8192
EPOCHS = 16
TRAIN_TOKENS = 4_672_623_743
LEARNING_RATE = float(os.environ.get("EXP157_LR", "3.162e-3"))
WEIGHT_DECAY = float(os.environ.get("EXP157_WEIGHT_DECAY", "0.2"))
WARMUP = float(os.environ.get("EXP157_WARMUP", "0.1"))
TRAIN_BATCH = int(os.environ.get("EXP157_TRAIN_BATCH", "16"))
NUM_TRAIN_STEPS = int(os.environ.get("EXP157_MAX_STEPS", "20"))
MODEL_SIZE = os.environ.get("EXP157_MODEL_SIZE", "1_5b")
MODEL_FAMILY = os.environ.get("EXP157_MODEL_FAMILY", "llama")
POSITION_MODE = os.environ.get("EXP157_POSITION_MODE", "fixed")
POSITION_DELTA_L2_WEIGHT = float(os.environ.get("EXP157_POSITION_DELTA_L2_WEIGHT", "0.0"))
GPU_VARIANT = os.environ.get("EXP157_GPU_VARIANT", "H100")
GPU_COUNT = int(os.environ.get("EXP157_GPU_COUNT", "8"))
GPU_REPLICAS = int(os.environ.get("EXP157_GPU_REPLICAS", "1"))
TARGET_CLUSTER = os.environ.get("EXP157_TARGET_CLUSTER") or None
STEPS_PER_EVAL = int(os.environ.get("EXP157_STEPS_PER_EVAL", str(NUM_TRAIN_STEPS)))
_MAX_EVAL_BATCHES = os.environ.get("EXP157_MAX_EVAL_BATCHES", "1")
MAX_EVAL_BATCHES = None if _MAX_EVAL_BATCHES.lower() in {"", "none", "full", "all"} else int(_MAX_EVAL_BATCHES)
INITIALIZE_FROM_CHECKPOINT_PATH = os.environ.get("EXP157_INITIALIZE_FROM_CHECKPOINT_PATH") or None
IRIS_PRIORITY = os.environ.get("EXP157_IRIS_PRIORITY", "batch")

_ATTN = os.environ.get("EXP157_ATTN", "jax_flash").upper()
ATTN_BACKEND = AttentionBackend[_ATTN] if _ATTN else None
_GRAD_CKPT = os.environ.get("EXP157_GRAD_CKPT", "1") != "0"


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def _verify_position_token_span() -> None:
    """Fail loudly if the tokenizer no longer maps <p0> to the expected id."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CONTACTS_V1_TOKENIZER)
    actual = int(tokenizer.convert_tokens_to_ids("<p0>"))
    if actual != CONTACTS_V1_P0_TOKEN_ID:
        raise ValueError(f"expected <p0> id {CONTACTS_V1_P0_TOKEN_ID}, got {actual}")
    last = int(tokenizer.convert_tokens_to_ids(f"<p{CONTACTS_V1_NUM_POSITION_TOKENS - 1}>"))
    expected_last = CONTACTS_V1_P0_TOKEN_ID + CONTACTS_V1_NUM_POSITION_TOKENS - 1
    if last != expected_last:
        raise ValueError(f"expected <p1999> id {expected_last}, got {last}")


MODEL_SHAPES = {
    "tiny": dict(hidden_dim=512, intermediate_dim=2048, num_heads=8, num_kv_heads=2, num_layers=4),
    "1_5b": dict(hidden_dim=2048, intermediate_dim=8192, num_heads=32, num_kv_heads=8, num_layers=24),
}
if MODEL_SIZE not in MODEL_SHAPES:
    raise ValueError(f"EXP157_MODEL_SIZE must be one of {sorted(MODEL_SHAPES)}, got {MODEL_SIZE!r}")
if MODEL_FAMILY not in {"llama", "qwen3"}:
    raise ValueError("EXP157_MODEL_FAMILY must be 'llama' or 'qwen3'")
if POSITION_MODE not in {"fixed", "rope_delta", "learned"}:
    raise ValueError("EXP157_POSITION_MODE must be one of 'fixed', 'rope_delta', or 'learned'")

_common_model_kwargs = dict(
    max_seq_len=SEQ_LEN,
    **MODEL_SHAPES[MODEL_SIZE],
    rope=Llama3RotaryEmbeddingsConfig(),
    attn_backend=ATTN_BACKEND,
    gradient_checkpointing=_GRAD_CKPT,
)
_position_embedding = ResiduePositionEmbeddingSpec(
    start_token_id=CONTACTS_V1_P0_TOKEN_ID,
    num_tokens=CONTACTS_V1_NUM_POSITION_TOKENS,
    trainable_delta=POSITION_MODE == "rope_delta",
    delta_l2_weight=POSITION_DELTA_L2_WEIGHT if POSITION_MODE == "rope_delta" else 0.0,
)
if POSITION_MODE in {"fixed", "rope_delta"} and MODEL_FAMILY == "qwen3":
    protein_llama_model = FixedResiduePositionQwen3Config(**_common_model_kwargs, position_embedding=_position_embedding)
elif POSITION_MODE in {"fixed", "rope_delta"}:
    protein_llama_model = FixedResiduePositionLlamaConfig(**_common_model_kwargs, position_embedding=_position_embedding)
elif MODEL_FAMILY == "qwen3":
    protein_llama_model = Qwen3Config(**_common_model_kwargs)
else:
    protein_llama_model = LlamaConfig(**_common_model_kwargs)

_DEFAULT_CPU = "32" if GPU_COUNT >= 4 else "8"
_DEFAULT_RAM = "256g" if GPU_COUNT >= 4 else "64g"
RESOURCES = ResourceConfig.with_gpu(
    GPU_VARIANT,
    count=GPU_COUNT,
    cpu=int(os.environ.get("EXP157_CPU") or _DEFAULT_CPU),
    ram=os.environ.get("EXP157_RAM") or _DEFAULT_RAM,
    disk=os.environ.get("EXP157_DISK") or "256g",
    replicas=GPU_REPLICAS,
    target_cluster=TARGET_CLUSTER,
)

def _l2_tag(weight: float) -> str:
    if weight == 0.0:
        return ""
    return f"-l2{weight:.0e}".replace("e-0", "em").replace("e+0", "e")


RUN_SUFFIX = os.environ.get("EXP157_RUN_SUFFIX", "smoke20-r1")
RUN_NAME = (
    f"exp157-cv1-{MODEL_SIZE}-e{EPOCHS}-lr{_lr_tag(LEARNING_RATE).replace('-', 'm')}-"
    f"wd0p2-bs{TRAIN_BATCH}-{MODEL_FAMILY}-{POSITION_MODE}-position{_l2_tag(POSITION_DELTA_L2_WEIGHT)}-{RUN_SUFFIX}"
)

_env_vars = {"WANDB_ENTITY": "open-athena"}
if os.environ.get("WANDB_API_KEY"):
    _env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]


def main() -> None:
    _verify_position_token_span()
    steps_per_epoch = math.ceil(TRAIN_TOKENS / (TRAIN_BATCH * SEQ_LEN))
    print(
        f"[exp157] {MODEL_FAMILY} {POSITION_MODE}-position next-token run: run={RUN_NAME} "
        f"gpu={GPU_VARIANT}x{GPU_COUNT} replicas={GPU_REPLICAS} target_cluster={TARGET_CLUSTER} priority={IRIS_PRIORITY} "
        f"batch={TRAIN_BATCH} seq={SEQ_LEN} steps={NUM_TRAIN_STEPS} "
        f"delta_l2={POSITION_DELTA_L2_WEIGHT:g} "
        f"({steps_per_epoch} steps/epoch at this batch; full e{EPOCHS} would be "
        f"{steps_per_epoch * EPOCHS})"
    )
    output_path = f"{CONTACTS_V1_S3_PREFIX}/checkpoints/{RUN_NAME}"
    job = dispatch_training_run(
        run_name=RUN_NAME,
        model_config=protein_llama_model,
        learning_rate=LEARNING_RATE,
        num_train_steps=NUM_TRAIN_STEPS,
        train_batch_size=TRAIN_BATCH,
        seq_len=SEQ_LEN,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        output_path=output_path,
        resources=RESOURCES,
        env_vars=_env_vars,
        wandb_name=RUN_NAME,
        tags=(
            "protein",
            "contacts-v1",
            "llama",
            MODEL_SIZE,
            MODEL_FAMILY,
            "unmasked",
            f"{POSITION_MODE}-position",
            f"position-delta-l2-{POSITION_DELTA_L2_WEIGHT:g}",
            "coreweave",
            f"{GPU_VARIANT.lower()}x{GPU_COUNT}n{GPU_REPLICAS}",
            f"bs{TRAIN_BATCH}",
            f"lr{_lr_tag(LEARNING_RATE)}",
        ),
        steps_per_eval=STEPS_PER_EVAL,
        max_eval_batches=MAX_EVAL_BATCHES,
        initialize_from_checkpoint_path=INITIALIZE_FROM_CHECKPOINT_PATH,
        priority=IRIS_PRIORITY,
        wait=True,
    )
    print(f"[exp157] child job completed: {job}")


if __name__ == "__main__":
    main()
