# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one cataloged Levanter checkpoint to a local HF directory."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main import export_lm_to_hf
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig

from checkpoint_specs import CHECKPOINTS, CheckpointSpec

VOCAB_SIZE = 2845


def build_model_config(spec: CheckpointSpec) -> Qwen3Config:
    """Recreate the exact Qwen3 geometry for the selected training run."""

    geometry = spec.geometry
    optional = {"head_dim": geometry.head_dim} if geometry.head_dim is not None else {}
    return Qwen3Config(
        max_seq_len=8192,
        hidden_dim=geometry.hidden_dim,
        intermediate_dim=geometry.intermediate_dim,
        num_layers=geometry.num_layers,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        rope=Llama3RotaryEmbeddingsConfig(),
        use_qk_norm=True,
        **optional,
    )


def run(checkpoint: str, output_dir: Path) -> int:
    """Restore one model from GCS and emit an HF export with its tokenizer."""

    spec = CHECKPOINTS[checkpoint]
    tokenizer = snapshot_download(repo_id=spec.tokenizer_repo, max_workers=1)
    config = ConvertLmConfig(
        trainer=TrainerConfig(),
        checkpoint_path=spec.checkpoint_uri,
        output_dir=str(output_dir),
        model=build_model_config(spec),
        tokenizer=tokenizer,
        override_vocab_size=VOCAB_SIZE,
        save_tokenizer=True,
        use_cpu=True,
    )
    print(f"[export] {spec.checkpoint_uri} -> {output_dir}", flush=True)
    export_lm_to_hf.main(config)
    print("[export] complete", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the selected checkpoint and destination."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.checkpoint, args.output_dir))
