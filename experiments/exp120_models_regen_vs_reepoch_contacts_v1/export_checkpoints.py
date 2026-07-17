# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export exp120 fine-tuned checkpoints to HF safetensors (CPU) for the downstream
contact-prediction eval, and (optionally) publish them to the open-athena bucket.

We convert each requested levanter ``step-N`` checkpoint with the exp75 Qwen3
config + contacts-v1 tokenizer co-located (hard rule: tokenizer travels with the
model), exactly as exp89 exported Eric's base checkpoint — so the exported models
drop straight into exp89's ``score_eval_set.py`` / ``compute_metrics.py`` harness.
``override_vocab_size=2845`` matches exp89's canonical HF format (slicing off any
Qwen vocab-padding dummies), so downstream scoring is directly comparable.

Runs locally on CPU in a marin/levanter venv (no TPU)::

    cd /home/bizon/git/marin
    uv run --no-sync python <exp120>/export_checkpoints.py \\
        --checkpoint gs://.../exp120_.../checkpoints/<run>/checkpoints/step-<N> \\
        --output-dir gs://.../exp120_.../hf/<run>/step-<N>

Repeat per (arm, LR, epoch) checkpoint you want scored. The checkpoint path is
the run's ``checkpoints/step-N`` dir under the exp120 MARIN_PREFIX. Add
``--bucket-dest hf://buckets/open-athena/MarinFold/checkpoints/<run>/step-<N>``
to publish (needs an open-athena-scoped HF token + an `hf` with `buckets`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess

TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
TOKENIZER_REVISION = "5d68a24a899f"
VOCAB_SIZE = 2845


def resolve_tokenizer(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo, revision=revision)
    print(f"[export] tokenizer {repo}@{revision} -> {path}")
    return path


def build_model_config():
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


def _find_hf() -> str:
    cands = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "hf")
        if os.path.exists(p) and ".venv" not in p and "/venv/" not in p and p not in cands:
            cands.append(p)
    w = shutil.which("hf")
    if w and w not in cands:
        cands.append(w)
    for hf in cands:
        try:
            if subprocess.run([hf, "buckets", "--help"], capture_output=True).returncode == 0:
                return hf
        except OSError:
            continue
    raise RuntimeError("no `hf` with `buckets` support on PATH")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="levanter step-N checkpoint dir")
    ap.add_argument("--output-dir", required=True, help="HF output (local or gs://)")
    ap.add_argument("--tokenizer-repo", default=TOKENIZER_REPO)
    ap.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--bucket-dest", default=None, help="hf://buckets/... to publish")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from levanter.main.export_lm_to_hf import ConvertLmConfig
    from levanter.main import export_lm_to_hf
    from levanter.trainer import TrainerConfig

    tokenizer_path = resolve_tokenizer(args.tokenizer_repo, args.tokenizer_revision)
    cfg = ConvertLmConfig(
        trainer=TrainerConfig(),  # unused on the use_cpu=True path
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model=build_model_config(),
        tokenizer=tokenizer_path,
        override_vocab_size=args.vocab_size,
        save_tokenizer=True,
        use_cpu=True,
    )
    print(f"[export] {args.checkpoint} -> {args.output_dir} (vocab {args.vocab_size})")
    if args.smoke:
        print("[export] SMOKE: config built OK; exiting before checkpoint load.")
        return 0

    export_lm_to_hf.main(cfg)
    print("[export] DONE")

    if args.bucket_dest:
        # `hf buckets cp` (hf 1.5) has no --recursive, so publish per-file. Only
        # supported for a local output_dir (list its files); gs:// dests are
        # already durable and don't need re-publishing.
        if args.output_dir.startswith("gs://") or args.output_dir.startswith("hf://"):
            raise SystemExit("--bucket-dest requires a LOCAL --output-dir to publish from")
        hf = _find_hf()
        dest = args.bucket_dest.rstrip("/")
        print(f"[export] publishing {args.output_dir} -> {dest}")
        for name in sorted(os.listdir(args.output_dir)):
            src = os.path.join(args.output_dir, name)
            if os.path.isfile(src):
                subprocess.run([hf, "buckets", "cp", src, f"{dest}/{name}"], check=True)
        print("[export] published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
