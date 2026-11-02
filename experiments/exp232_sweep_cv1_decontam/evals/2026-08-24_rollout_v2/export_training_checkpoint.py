# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the selected exp232 Levanter checkpoint to eval-local HF format.

The command submitted here runs entirely in CoreWeave. It reads the source
checkpoint from CoreWeave S3 and writes the Hugging Face export to the dated
evaluation prefix in the same object store; neither model path touches this VM.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import textwrap
from pathlib import Path

SOURCE_CHECKPOINT = (
    "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
    "checkpoints/protein/"
    "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-"
    "lr005-us-east1/2026.08.21.1/checkpoints/step-363000"
)
OUTPUT_CHECKPOINT = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/"
    "models/exp232-decontam-train-m2-p06-step363000/hf/step-363000"
)
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
TARGET_CLUSTER = "cw-us-east-02a"
MARIN_PREFIX = "s3://marin-us-east-02a/marin"
MARIN_REPO = Path("/home/exedev/repos/marin-br/main")
DEFAULT_IRIS = MARIN_REPO / ".venv/bin/iris"


def export_payload() -> str:
    """Return the direct Levanter CPU-export program run in CoreWeave."""

    return textwrap.dedent(
        f"""
        from levanter.layers.attention import AttentionBackend
        from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
        from levanter.main import export_lm_to_hf
        from levanter.main.export_lm_to_hf import ConvertLmConfig
        from levanter.models.qwen import Qwen3Config
        from levanter.trainer import TrainerConfig

        model = Qwen3Config(
            max_seq_len=8192,
            hidden_dim=2048,
            intermediate_dim=8192,
            num_heads=32,
            num_kv_heads=8,
            num_layers=24,
            rope=Llama3RotaryEmbeddingsConfig(),
            use_qk_norm=True,
            attn_backend=AttentionBackend.JAX_FLASH,
        )
        config = ConvertLmConfig(
            trainer=TrainerConfig(require_accelerator=False),
            checkpoint_path={SOURCE_CHECKPOINT!r},
            output_dir={OUTPUT_CHECKPOINT!r},
            checkpoint_subpath="model",
            max_shard_size=5_000_000_000,
            model=model,
            save_tokenizer=True,
            tokenizer={TOKENIZER!r},
            override_vocab_size=2845,
            use_cpu=True,
        )
        export_lm_to_hf.main(config)
        """
    ).strip()


def main() -> None:
    """Submit the CPU export through the production Marin Iris controller."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [
        str(args.iris_bin),
        "--cluster=marin",
        "job",
        "run",
        "--target-cluster",
        TARGET_CLUSTER,
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--user",
        "eczech",
        "--job-name",
        "exp232-export-train-step363000-v2-01-r1",
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "64GB",
        "--max-retries",
        "3",
        "--timeout",
        "10800",
        "--no-wait",
        "--sync-package",
        "marin-levanter",
        "-e",
        "MARIN_PREFIX",
        MARIN_PREFIX,
        "--",
        "python",
        "-c",
        export_payload(),
    ]
    print(f"source: {SOURCE_CHECKPOINT}")
    print(f"destination: {OUTPUT_CHECKPOINT}")
    print(f"target: {TARGET_CLUSTER} via marin as eczech")
    if args.dry_run:
        print("dry run: validated export command; inline payload omitted")
        return
    subprocess.run(command, cwd=MARIN_REPO, check=True)


if __name__ == "__main__":
    main()
