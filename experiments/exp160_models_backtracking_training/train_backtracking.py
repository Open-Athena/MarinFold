# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Training worker: full fine-tune of exp120 on the 50:50 backtracking mix (#160).

Invoked by ``dispatch_train.py`` inside the GPU job. Assembles a
``TrainLmOnPodConfig`` via ``marinfold_models.defaults.build_train_lm_on_pod_config``
and runs it, following exp108's GPU wiring.

Notes that are easy to get wrong:

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
  workers from the parquet URLs; there is no separate tokenize step to route
  through GCS (which CoreWeave cannot reach anyway).
- **Warm start, not resume**: ``initialize_from_checkpoint_path`` takes model
  weights only — fresh step 0, fresh optimizer state, fresh LR schedule.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig

from marinfold_models.defaults import build_train_lm_on_pod_config

# exp75/exp117/exp120 architecture (Qwen3 1.47B: exp44 dims + Llama3 rope).
SEQ_LEN = 8192


def fetch_tokenizer(bucket_dir: str, local: Path) -> Path:
    """Download the superset tokenizer from the bucket to a local dir."""
    import subprocess

    local.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        subprocess.run(
            ["hf", "buckets", "cp", f"{bucket_dir}/{name}", str(local / name)],
            check=True,
        )
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
    args = ap.parse_args()

    tokenizer = args.tokenizer
    if tokenizer.startswith("hf://"):
        tokenizer = str(fetch_tokenizer(tokenizer, Path("/tmp/exp160_tokenizer")))
    print(f"tokenizer -> {tokenizer}", flush=True)

    model = Qwen3Config(
        seq_len=SEQ_LEN,
        hidden_dim=2048,
        intermediate_dim=8192,
        num_layers=24,
        num_heads=32,
        num_kv_heads=8,
    )
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
        resources=None,  # set by the dispatcher's JobRequest; unused on-pod
        output_path=f"{args.output}/{args.run_name}",
        num_train_steps=args.num_train_steps,
        train_batch_size=args.train_batch_size,
        seq_len=SEQ_LEN,
        data_seed=args.data_seed,
        # Warm start: weights only, fresh optimizer/LR/step-0.
        initialize_from_checkpoint_path=args.init,
        wandb_project="MarinFold",
        wandb_name=args.run_name,
        tags=("protein", "contacts-v1", "backtracking", "qwen3", "1.5b",
              "unmasked", "coreweave"),
        env_vars={"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")},
    )

    from marin.training.training import run_levanter_train_lm

    run_levanter_train_lm(config)


if __name__ == "__main__":
    main()
