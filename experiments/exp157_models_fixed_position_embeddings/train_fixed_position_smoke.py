# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave next-token training smoke for fixed residue-position embeddings.

This is the regular CE/control-style arm for issue #157: contacts-v1, unmasked
next-token loss, exp117-ish recipe (LR 3.162e-3, wd 0.2, cosine, 10% warmup),
but with the learned input rows for ``<p0>`` ... ``<p1999>`` replaced by fixed
RoPE/sinusoidal residue-location vectors.
"""

import math
import os

from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

from fray.types import ResourceConfig

from contacts_v1_train_common import (
    CONTACTS_V1_NUM_POSITION_TOKENS,
    CONTACTS_V1_P0_TOKEN_ID,
    CONTACTS_V1_S3_PREFIX,
    CONTACTS_V1_TOKENIZER,
    PROTEIN_RESOURCES_H100,
)
from dispatch_train import dispatch_training_run
from fixed_position_model import FixedResiduePositionLlamaConfig, ResiduePositionEmbeddingSpec

SEQ_LEN = 8192
EPOCHS = 16
TRAIN_TOKENS = 4_672_623_743
LEARNING_RATE = float(os.environ.get("EXP157_LR", "3.162e-3"))
WEIGHT_DECAY = float(os.environ.get("EXP157_WEIGHT_DECAY", "0.2"))
WARMUP = float(os.environ.get("EXP157_WARMUP", "0.1"))
TRAIN_BATCH = int(os.environ.get("EXP157_TRAIN_BATCH", "16"))
NUM_TRAIN_STEPS = int(os.environ.get("EXP157_MAX_STEPS", "20"))
MODEL_SIZE = os.environ.get("EXP157_MODEL_SIZE", "1_5b")
GPU_COUNT = int(os.environ.get("EXP157_GPU_COUNT", "8"))
STEPS_PER_EVAL = int(os.environ.get("EXP157_STEPS_PER_EVAL", str(NUM_TRAIN_STEPS)))
MAX_EVAL_BATCHES = int(os.environ.get("EXP157_MAX_EVAL_BATCHES", "1"))

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

protein_llama_fixed_positions = FixedResiduePositionLlamaConfig(
    max_seq_len=SEQ_LEN,
    **MODEL_SHAPES[MODEL_SIZE],
    rope=Llama3RotaryEmbeddingsConfig(),
    attn_backend=ATTN_BACKEND,
    gradient_checkpointing=_GRAD_CKPT,
    position_embedding=ResiduePositionEmbeddingSpec(
        start_token_id=CONTACTS_V1_P0_TOKEN_ID,
        num_tokens=CONTACTS_V1_NUM_POSITION_TOKENS,
    ),
)

RESOURCES = (
    PROTEIN_RESOURCES_H100
    if GPU_COUNT == 8
    else ResourceConfig.with_gpu("H100", count=GPU_COUNT, cpu=8, ram="64g", disk="256g", replicas=1)
)

RUN_SUFFIX = os.environ.get("EXP157_RUN_SUFFIX", "smoke20-r1")
RUN_NAME = (
    f"exp157-cv1-{MODEL_SIZE}-e{EPOCHS}-lr{_lr_tag(LEARNING_RATE).replace('-', 'm')}-"
    f"wd0p2-bs{TRAIN_BATCH}-fixed-position-{RUN_SUFFIX}"
)

_env_vars = {"WANDB_ENTITY": "open-athena"}
if os.environ.get("WANDB_API_KEY"):
    _env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]


def main() -> None:
    _verify_position_token_span()
    steps_per_epoch = math.ceil(TRAIN_TOKENS / (TRAIN_BATCH * SEQ_LEN))
    print(
        f"[exp157] fixed-position next-token smoke: run={RUN_NAME} "
        f"batch={TRAIN_BATCH} seq={SEQ_LEN} steps={NUM_TRAIN_STEPS} "
        f"({steps_per_epoch} steps/epoch at this batch; full e{EPOCHS} would be "
        f"{steps_per_epoch * EPOCHS})"
    )
    output_path = f"{CONTACTS_V1_S3_PREFIX}/checkpoints/{RUN_NAME}"
    job = dispatch_training_run(
        run_name=RUN_NAME,
        model_config=protein_llama_fixed_positions,
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
            "unmasked",
            "fixed-position",
            "coreweave",
            f"bs{TRAIN_BATCH}",
            f"lr{_lr_tag(LEARNING_RATE)}",
        ),
        steps_per_eval=STEPS_PER_EVAL,
        max_eval_batches=MAX_EVAL_BATCHES,
        wait=True,
    )
    print(f"[exp157] child job completed: {job}")


if __name__ == "__main__":
    main()
