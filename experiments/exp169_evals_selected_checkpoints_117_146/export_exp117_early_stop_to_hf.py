# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the #117 early-stop checkpoint (step 33450) to HF safetensors.

Issue #169 lists three checkpoints; two ship a published ``hf/step-N/`` export
and this one does not — only the levanter/orbax train state exists. So we
convert it the same way exp89 converted the #75 winner: levanter's
``export_lm_to_hf`` on **CPU**, with the training-time model config and the
contacts-v1 tokenizer co-located in the output (the repo's "always save the
tokenizer with the model" rule).

The model config is the exp117 sweep's, read off marin branch ``eac/plm-exp117``
(``experiments/protein/exp117_sweep.py``): the same Qwen3 1.47B geometry as
exp75 — the #117 sweep varied only lr / wd / batch size / epochs.

Levanter and jax are heavy and platform-coupled, so this runs out of a marin
checkout's venv rather than this experiment's::

    /home/bizon/git/marin/.venv/bin/python export_exp117_early_stop_to_hf.py \\
        --checkpoint-dir /home/bizon/exp169_eval/dl_exp117_lev33450/<run>/checkpoints \\
        --step 33450 --output-dir /home/bizon/exp169_eval/hf_exp117_step33450

``--checkpoint-dir`` is the directory *containing* ``step-<N>/`` (levanter
resolves the step itself), local or ``gs://``.
"""

import argparse

# Verbatim from marin `eac/plm-exp117` (exp117_sweep.py MODEL_CONFIG), which is
# exp49's Qwen3 1.47B: exp44 dims + Llama3 rope. Identical to exp75's — the #117
# sweep tuned only the optimizer/schedule, never the architecture.
TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
TOKENIZER_REVISION = "5d68a24a899f"
VOCAB_SIZE = 2845


def resolve_tokenizer(repo: str, revision: str) -> str:
    """Download the tokenizer at the pinned revision; return the local path.

    levanter's ``load_tokenizer`` rejects the ``repo@rev`` syntax (HF repo-id
    validation trips on the ``@``), so the revision is snapshotted and the
    resulting path handed to the exporter instead.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                    help="directory containing step-<N>/ (local or gs://)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--output-dir", required=True, help="HF output dir (local or gs://)")
    ap.add_argument("--tokenizer-repo", default=TOKENIZER_REPO)
    ap.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    from levanter.checkpoint import latest_checkpoint_path
    from levanter.main import export_lm_to_hf
    from levanter.main.export_lm_to_hf import ConvertLmConfig
    from levanter.trainer import TrainerConfig

    # levanter resolves the concrete step itself. Assert it picked the step we
    # were asked for rather than trusting whatever happens to be newest under
    # the directory — silently exporting the wrong step is the failure mode
    # this whole experiment exists to avoid.
    resolved = latest_checkpoint_path(args.checkpoint_dir)
    if not resolved.rstrip("/").endswith(f"step-{args.step}"):
        raise SystemExit(
            f"--step {args.step} requested but {args.checkpoint_dir} resolves to {resolved}"
        )

    config = ConvertLmConfig(
        trainer=TrainerConfig(),  # unused on the use_cpu=True path (local_cpu_mesh)
        checkpoint_path=args.checkpoint_dir,
        output_dir=args.output_dir,
        model=build_model_config(),
        tokenizer=resolve_tokenizer(args.tokenizer_repo, args.tokenizer_revision),
        override_vocab_size=args.vocab_size,
        save_tokenizer=True,  # hard rule: tokenizer travels with the model
        use_cpu=True,
    )
    print(f"[export] {resolved} -> {config.output_dir}  vocab={args.vocab_size}")
    export_lm_to_hf.main(config)
    print("[export] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
