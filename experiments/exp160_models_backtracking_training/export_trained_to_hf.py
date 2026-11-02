# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the trained backtracking checkpoint to HF safetensors (#160).

The training run writes a levanter/orbax train state and no HF export, so the
eval workers (vLLM) need this conversion. Same route exp89 and exp169 used:
levanter's ``export_lm_to_hf`` on **CPU**, with the training-time model config
and the tokenizer co-located in the output (the repo's "the tokenizer always
travels with the model" rule).

Two things are specific to this run and easy to get wrong:

* **The tokenizer is the 3849-token superset**, published next to the corpus
  rather than in an HF model repo, so it is fetched from the bucket to a local
  directory first. Handing levanter the *contacts-v1* tokenizer instead would
  build a 2845-row embedding table and load the checkpoint's 3849-row one into
  it — a strict-shape error if you are lucky, and a silent truncation of
  ``<retract>`` if you are not.
* **``--step`` is asserted, not inferred.** ``latest_checkpoint_path`` resolves
  whatever is newest under the directory; on a preemptible pool that can be a
  rolling mid-training checkpoint rather than the finished run. Exporting the
  wrong step and reporting it as the trained model is the failure this guards.

The output is fp32 — feed it to ``prepare_eval_model.py`` for the bf16 +
transformers-4.x form the workers load.

levanter and jax are heavy and platform-coupled, so this runs out of a marin
checkout's venv rather than this experiment's::

    /home/bizon/git/marin-freshiris/.venv/bin/python export_trained_to_hf.py \\
        --checkpoint-dir gs://…/runs/<run>/checkpoints --step 2059 \\
        --output-dir /home/bizon/exp160_eval/exp160_fp32
"""

from __future__ import annotations

import argparse
from pathlib import Path

# The 50:50 mix's tokenizer, published beside the corpus (#160 launch config).
DEFAULT_TOKENIZER = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp160_backtracking_training/"
    "corpus/tokenizer"
)
VOCAB_SIZE = 3849
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def fetch_tokenizer(remote: str, local: Path) -> str:
    """Copy a bucket tokenizer directory somewhere ``load_tokenizer`` can read it."""
    if "://" not in remote:
        return remote
    import fsspec

    local.mkdir(parents=True, exist_ok=True)
    fs, _ = fsspec.core.url_to_fs(remote)
    for name in TOKENIZER_FILES:
        fs.get(f"{remote.rstrip('/')}/{name}", str(local / name))
    print(f"[export] tokenizer {remote} -> {local}")
    return str(local)


def build_model_config():
    """Must match ``train_backtracking.MODEL_CONFIG`` exactly.

    Restated here rather than imported because that module pulls in the marin
    training stack; the geometry is exp75/exp117/exp120's Qwen3 1.47B. The
    **Llama3 rope** is the field that matters: it carries no parameters, so a
    mismatch exports cleanly and produces a differently-positioned model.
    """
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.models.qwen import Qwen3Config

    return Qwen3Config(
        max_seq_len=8192,
        hidden_dim=2048,
        intermediate_dim=8192,
        num_heads=32,
        num_kv_heads=8,
        num_layers=24,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                    help="directory containing step-<N>/ (local or gs://)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    from levanter.checkpoint import latest_checkpoint_path
    from levanter.main import export_lm_to_hf
    from levanter.main.export_lm_to_hf import ConvertLmConfig
    from levanter.trainer import TrainerConfig

    resolved = latest_checkpoint_path(args.checkpoint_dir)
    if not resolved.rstrip("/").endswith(f"step-{args.step}"):
        raise SystemExit(
            f"--step {args.step} requested but {args.checkpoint_dir} resolves to {resolved}"
        )

    tokenizer = fetch_tokenizer(args.tokenizer, Path("/tmp/exp160_export_tokenizer"))
    config = ConvertLmConfig(
        trainer=TrainerConfig(),  # unused on the use_cpu=True path (local_cpu_mesh)
        checkpoint_path=args.checkpoint_dir,
        output_dir=args.output_dir,
        model=build_model_config(),
        tokenizer=tokenizer,
        override_vocab_size=args.vocab_size,
        save_tokenizer=True,  # hard rule: the tokenizer travels with the model
        use_cpu=True,
    )
    print(f"[export] {resolved} -> {config.output_dir}  vocab={args.vocab_size}")
    export_lm_to_hf.main(config)
    print("[export] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
