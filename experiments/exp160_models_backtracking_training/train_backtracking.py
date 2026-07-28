# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Training worker: full fine-tune of exp120 on the 50:50 backtracking mix (#160).

Runs on a **v5p-32 TPU slice in us-east5-a** (see ``dispatch_train.py`` for the
availability evidence behind that zone). Assembles a ``TrainLmOnPodConfig`` via
``marinfold_models.defaults.build_train_lm_on_pod_config`` and runs it in the
current process — Levanter/JAX handle the 4 VMs of the slice.

Notes that are easy to get wrong:

- **``resources`` must be a real ``ResourceConfig``, not ``None``.**
  ``run_levanter_train_lm`` reads ``config.resources.device`` (to log the shape
  and to pick the GCS region check), so a ``None`` would ``AttributeError`` the
  moment training started. The earlier CoreWeave revision passed ``None`` and
  never got far enough to find out.
- **Every GCS path must be in us-east5.** With a TPU device,
  ``check_gcs_paths_same_region`` hard-fails on any cross-region path, which is
  exactly the co-location rule we want enforced. Corpus, token cache, init
  checkpoint and output all live under ``gs://marin-us-east5`` (a regional
  bucket in US-EAST5, same region as the us-east5-a slice).
- **Tokenizer comes from the bucket, not a HF model repo.** The superset
  tokenizer (crops/ccoord vocab, 3849 tokens, `<retract>` at 3848) is published
  next to the corpus, per the "tokenizer lives beside the data" convention. We
  download it to a local dir at startup and hand Levanter that path, which
  avoids minting a new HF model repo.
- **Unmasked loss, `pack=True`** — same as #117. contacts-v1 has no
  `<distance>` statements to mask. This keeps us off the
  ``PrebuiltLmDatasetFormat`` path that marin 0.2.57 would otherwise force for
  per-token weights.
- **`auto_build_caches=True`** so the token cache is built on the training
  workers from the parquet URLs; there is no separate tokenize step.
- **Warm start, not resume**: ``initialize_from_checkpoint_path`` takes model
  weights only — fresh step 0, fresh optimizer state, fresh LR schedule.
  Levanter's own rolling checkpoints (10-minute cadence, set in
  ``build_train_lm_on_pod_config``) are what make the run survive preemption —
  the v5p pool is ``capacity_type: preemptible`` whatever band we submit at.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# `from fray import ResourceConfig` (exp120's spelling) no longer works: this
# fray line re-exports nothing at the package root.
from fray.types import ResourceConfig
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig

from marinfold_models.defaults import build_train_lm_on_pod_config

SEQ_LEN = 8192

# exp75/exp117/exp120 architecture (Qwen3 1.47B: exp44 dims + Llama3 rope),
# copied verbatim from exp120's ``MODEL_CONFIG``. It MUST match the checkpoint
# we warm-start from — in particular the **Llama3 rope**, which carries no
# parameters, so getting it wrong loads the weights without complaint and then
# trains a differently-positioned model. (Also note the field is
# ``max_seq_len``; ``seq_len=`` is a TypeError, not a silent default.)
MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# The pod shape, restated for marin's benefit (see the module docstring). Sized
# to the ct5p-hightpu-4t VM: v5p-32 is 4 VMs x 4 chips. Disk holds the 17.7 GiB
# init checkpoint plus the token cache.
TPU_TYPE = "v5p-32"
TPU_ZONE = "us-east5-a"


def build_resources(tpu_type: str, zone: str) -> ResourceConfig:
    return ResourceConfig.with_tpu(
        tpu_type, slice_count=1, cpu=200, ram="400g", disk="200g", zone=zone
    )


def fetch_tokenizer(remote_dir: str, local: Path) -> Path:
    """Download the superset tokenizer to a local dir Levanter can load.

    AutoTokenizer cannot read an object-store URI directly, hence the copy.
    ``gs://`` routes through gcsfs on the TPU VM's service account.
    """
    import fsspec

    local.mkdir(parents=True, exist_ok=True)
    fs, _ = fsspec.core.url_to_fs(remote_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        fs.get(f"{remote_dir.rstrip('/')}/{name}", str(local / name))
    return local


def build_data_config(corpus: str, tokenizer: str) -> LmDataConfig:
    """Tokenize-on-the-fly config over the mix's parquet shards."""
    source = UrlDatasetSourceConfig(
        train_urls=[f"{corpus}/train/*.parquet"],
        validation_urls=[],
        cache_dir=f"{corpus}/tokenized/mix50",
        format=TextLmDatasetFormat(text_key="document"),
    )
    component = DatasetComponent(
        source=source,
        cache_dir=source.cache_dir,
        format=source.format,
        # pack=True: never concat-and-split — a partial protein document is
        # nonsensical. No loss mask: every token contributes.
        pack=True,
        split="train",
    )
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        auto_build_caches=True,
        shuffle=True,
        block_cross_document_attention=True,
        components={"backtracking-mix50": component},
        train_weights={"backtracking-mix50": 1.0},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init", required=True, help="Levanter checkpoint dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="bucket dir holding the superset tokenizer, or a local path")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.2)
    ap.add_argument("--train-batch-size", type=int, default=128)
    ap.add_argument("--num-train-steps", type=int, required=True)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--tpu-type", default=TPU_TYPE)
    ap.add_argument("--tpu-zone", default=TPU_ZONE)
    args = ap.parse_args()

    tokenizer = args.tokenizer
    if tokenizer.startswith(("s3://", "gs://", "hf://")):
        tokenizer = str(fetch_tokenizer(tokenizer, Path("/tmp/exp160_tokenizer")))
    print(f"tokenizer -> {tokenizer}", flush=True)

    model = MODEL_CONFIG
    optimizer = AdamConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        lr_schedule="cosine",
    )
    config = build_train_lm_on_pod_config(
        run_name=args.run_name,
        model=model,
        optimizer=optimizer,
        data=build_data_config(args.corpus, tokenizer),
        resources=build_resources(args.tpu_type, args.tpu_zone),
        output_path=f"{args.output}/{args.run_name}",
        num_train_steps=args.num_train_steps,
        train_batch_size=args.train_batch_size,
        seq_len=SEQ_LEN,
        data_seed=args.data_seed,
        # Must be set HERE, on the pod config — marin overrides the inner
        # LmDataConfig's flag with this one, whose default is False. There is
        # no separate tokenize step for this corpus, so the workers build the
        # cache from the parquet URLs on first run.
        auto_build_caches=True,
        # Warm start: weights only, fresh optimizer/LR/step-0.
        initialize_from_checkpoint_path=args.init,
        wandb_project="MarinFold",
        wandb_name=args.run_name,
        tags=("protein", "contacts-v1", "backtracking", "qwen3", "1.5b",
              "unmasked", "tpu", "v5p"),
        env_vars={"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")},
    )

    from marin.training.training import run_levanter_train_lm

    run_levanter_train_lm(config)


if __name__ == "__main__":
    main()
