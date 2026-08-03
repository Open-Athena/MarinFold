# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch Levanter recomputation of exp117 contacts-v1-val loss."""

import argparse
import base64
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "recompute_eval_loss_levanter.py"
IRIS = os.environ.get("IRIS_BIN", "/Users/zack/projects/agent_workspaces/beta/.venv-iris/bin/iris")
MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/Users/zack/projects/agent_workspaces/repos/marin-beta"))
MODEL = os.environ.get(
    "EXP188_MODEL",
    "gs://marin-us-central1/protein-structure/MarinFold/exp169/models/exp117_e16_final_step35679",
)
TOKENIZER = os.environ.get("EXP188_TOKENIZER")
NATIVE_CHECKPOINT = os.environ.get("EXP188_NATIVE_CHECKPOINT")
CACHE_DIR = os.environ.get("EXP188_CACHE_DIR", "gs://marin-us-east5/tokenized/contacts-v1-val/2026.07.13.1")
RAW_VALIDATION_URLS = [url for url in os.environ.get("EXP188_RAW_VALIDATION_URLS", "").split(",") if url]
TEXT_KEY = os.environ.get("EXP188_TEXT_KEY")
EVAL_MODE = os.environ.get("EXP188_EVAL_MODE", "tagged")
VOCAB_SIZE = os.environ.get("EXP188_VOCAB_SIZE")
MODEL_ARCH = os.environ.get("EXP188_MODEL_ARCH")
ROPE_TYPE = os.environ.get("EXP188_ROPE_TYPE")
ROPE_THETA = os.environ.get("EXP188_ROPE_THETA")
ROPE_FACTOR = os.environ.get("EXP188_ROPE_FACTOR")
MAX_SEQ_LEN = os.environ.get("EXP188_MAX_SEQ_LEN")
HIDDEN_DIM = os.environ.get("EXP188_HIDDEN_DIM")
INTERMEDIATE_DIM = os.environ.get("EXP188_INTERMEDIATE_DIM")
NUM_HEADS = os.environ.get("EXP188_NUM_HEADS")
NUM_KV_HEADS = os.environ.get("EXP188_NUM_KV_HEADS")
NUM_LAYERS = os.environ.get("EXP188_NUM_LAYERS")
HEAD_DIM = os.environ.get("EXP188_HEAD_DIM")
USE_QK_NORM = os.environ.get("EXP188_USE_QK_NORM") in {"1", "true", "True"}
ATTN_BACKEND = os.environ.get("EXP188_ATTN_BACKEND")
LOSS_WEIGHT_MODE = os.environ.get("EXP188_LOSS_WEIGHT_MODE")
PADDING_TARGET_LOSS = os.environ.get("EXP188_PADDING_TARGET_LOSS")
ALLOW_CROSS_DOCUMENT_ATTENTION = os.environ.get("EXP188_ALLOW_CROSS_DOCUMENT_ATTENTION") in {"1", "true", "True"}
OUT_PREFIX = os.environ.get("EXP188_PREFIX", "gs://marin-us-central1/protein-structure/MarinFold/exp188")
PRIORITY = os.environ.get("EXP188_PRIORITY", "interactive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--job-suffix", default="full")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--tpu", default="v5p-8")
    args = parser.parse_args()

    script_b64 = base64.b64encode(SCRIPT.read_bytes()).decode()
    limit = f" --max-eval-batches {args.max_eval_batches}" if args.max_eval_batches is not None else ""
    tokenizer = f" --tokenizer {TOKENIZER}" if TOKENIZER else ""
    checkpoint = f" --checkpoint-path {NATIVE_CHECKPOINT}" if NATIVE_CHECKPOINT else ""
    vocab_size = f" --vocab-size {VOCAB_SIZE}" if VOCAB_SIZE else ""
    cross_doc_attention = " --allow-cross-document-attention" if ALLOW_CROSS_DOCUMENT_ATTENTION else ""
    model_shape = "".join(
        f" {flag} {value}"
        for flag, value in [
            ("--model-arch", MODEL_ARCH),
            ("--rope-type", ROPE_TYPE),
            ("--rope-theta", ROPE_THETA),
            ("--rope-factor", ROPE_FACTOR),
            ("--max-seq-len", MAX_SEQ_LEN),
            ("--hidden-dim", HIDDEN_DIM),
            ("--intermediate-dim", INTERMEDIATE_DIM),
            ("--num-heads", NUM_HEADS),
            ("--num-kv-heads", NUM_KV_HEADS),
            ("--num-layers", NUM_LAYERS),
            ("--head-dim", HEAD_DIM),
            ("--attn-backend", ATTN_BACKEND),
        ]
        if value
    )
    if USE_QK_NORM:
        model_shape += " --use-qk-norm"
    loss_weight_mode = f" --loss-weight-mode {LOSS_WEIGHT_MODE}" if LOSS_WEIGHT_MODE else ""
    padding_target_loss = f" --padding-target-loss {PADDING_TARGET_LOSS}" if PADDING_TARGET_LOSS else ""
    raw_validation_urls = "".join(f" --raw-validation-url {url}" for url in RAW_VALIDATION_URLS)
    text_key = f" --text-key {TEXT_KEY}" if TEXT_KEY else ""
    output = f"{OUT_PREFIX}/levanter_recompute/{args.job_suffix}.json"
    command_text = f"""
set -euo pipefail
mkdir -p /tmp/exp188
echo {script_b64} | base64 -d > /tmp/exp188/recompute_eval_loss_levanter.py
exec uv run --no-sync python /tmp/exp188/recompute_eval_loss_levanter.py \
  --model {MODEL}{tokenizer} \
  --cache-dir {CACHE_DIR}{raw_validation_urls}{text_key} \
  --eval-mode {EVAL_MODE} \
  --output {output}{limit}{checkpoint}{vocab_size}{cross_doc_attention}{model_shape}{loss_weight_mode}{padding_target_loss}
""".strip()
    job_name = f"exp188-levanter-loss-exp117-{args.job_suffix}"
    command = [
        IRIS,
        "--cluster=marin",
        "job",
        "run",
        "--job-name",
        job_name,
        "--no-wait",
        "--enable-extra-resources",
        "--priority",
        PRIORITY,
        "--zone",
        args.zone,
        "--tpu",
        args.tpu,
        "--extra",
        "tpu",
        "--cpu",
        "16",
        "--memory",
        "96GB",
        "--disk",
        "64GB",
        "--max-retries",
        "1",
        "--",
        "bash",
        "-lc",
        command_text,
    ]
    subprocess.run(command, cwd=MARIN, check=True)
    print(f"/zack/{job_name}\noutput={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
