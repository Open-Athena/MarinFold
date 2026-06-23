# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the #61/#75 best contacts-v1 1.5B checkpoint (eval loss 2.7566) to HF.

The model is eric-czech's tuned sweep winner from issue #75
(``prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084``; epochs=8, lr=1e-3, wd=0.2;
final ``eval/contacts-v1-val/loss`` = 2.756602 at step 35679). Its levanter
checkpoint lives at::

    gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/

There is no HF export yet (the run's ``hf_save_path`` is set but ``hf/`` is
empty), so we convert it here. We call levanter's ``export_lm_to_hf`` directly
(the same entrypoint marin's ``convert_checkpoint_to_hf_step`` shells into) on
**CPU**, so this runs locally in the marin ``.venv`` with no TPU/iris.

Model config + tokenizer are copied verbatim from
``marin:experiments/protein/exp75_sweep.py`` (the exact training recipe):
Qwen3 1.47B (exp44 dims, Llama3 rope), contacts-v1 tokenizer (2845 vocab).

Run (from the marin checkout so the venv resolves levanter/marin)::

    cd /home/bizon/git/marin
    uv run --no-sync python \
        /path/to/experiments/exp89_.../export_contacts_v1_best_to_hf.py \
        --output-dir /home/bizon/exp89_export/hf_step35679
"""
from __future__ import annotations

import argparse

# --- exp75 training recipe (verbatim from marin exp75_sweep.py) --------------
CHECKPOINT_DIR = (
    "gs://marin-us-east5/checkpoints/"
    "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints"
)
# exp75 trained from this exact tokenizer revision. This levanter's
# ``load_tokenizer`` doesn't parse the ``repo@rev`` syntax (HF repo-id
# validation rejects the ``@``), so we snapshot the pinned revision to a local
# dir and hand that path to the exporter instead.
TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
TOKENIZER_REVISION = "5d68a24a899f"
VOCAB_SIZE = 2845


def resolve_tokenizer(repo: str, revision: str) -> str:
    """Download the tokenizer at the pinned revision; return the local path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo, revision=revision)
    print(f"[export] tokenizer {repo}@{revision} -> {path}")
    return path


def build_model_config():
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.models.qwen import Qwen3Config

    # exp75 MODEL_CONFIG (exp49 Qwen3 1.47B; exp44 dims + Llama3 rope).
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
    ap.add_argument("--output-dir", required=True, help="HF output dir (local or gs://)")
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    ap.add_argument("--tokenizer-repo", default=TOKENIZER_REPO)
    ap.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Construct the config and exit before loading the checkpoint.",
    )
    args = ap.parse_args()

    from levanter.main.export_lm_to_hf import ConvertLmConfig
    from levanter.main import export_lm_to_hf
    from levanter.trainer import TrainerConfig

    tokenizer_path = resolve_tokenizer(args.tokenizer_repo, args.tokenizer_revision)

    cfg = ConvertLmConfig(
        trainer=TrainerConfig(),  # unused on the use_cpu=True path (local_cpu_mesh)
        checkpoint_path=args.checkpoint_dir,
        output_dir=args.output_dir,
        model=build_model_config(),
        tokenizer=tokenizer_path,
        override_vocab_size=args.vocab_size,
        save_tokenizer=True,  # hard rule: tokenizer travels with the model
        use_cpu=True,
    )
    print(f"[export] checkpoint={args.checkpoint_dir}")
    print(f"[export] output={args.output_dir}  vocab={args.vocab_size}")
    if args.smoke:
        print("[export] SMOKE: config constructed OK; exiting before checkpoint load.")
        return 0

    export_lm_to_hf.main(cfg)
    print("[export] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
