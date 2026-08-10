# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export one exp199 Levanter checkpoint to a local HF directory."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main import export_lm_to_hf
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig

from checkpoint_specs import CHECKPOINTS

VOCAB_SIZE = 2845


def build_model_config() -> Qwen3Config:
    """Recreate the shared model geometry used by every exp199 trial."""

    return Qwen3Config(
        max_seq_len=8192,
        hidden_dim=2048,
        intermediate_dim=8192,
        num_heads=32,
        num_kv_heads=8,
        num_layers=24,
        rope=Llama3RotaryEmbeddingsConfig(),
        use_qk_norm=True,
    )


def run(checkpoint: str, output_dir: Path) -> int:
    """Restore one model from GCS and emit an HF export with its tokenizer."""

    spec = CHECKPOINTS[checkpoint]
    if spec.checkpoint_uri is None or spec.tokenizer_repo is None:
        raise ValueError(f"{spec.key} is not a native Levanter checkpoint")
    tokenizer = snapshot_download(repo_id=spec.tokenizer_repo, max_workers=1)
    config = ConvertLmConfig(
        trainer=TrainerConfig(),
        checkpoint_path=spec.checkpoint_uri,
        output_dir=str(output_dir),
        model=build_model_config(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.checkpoint, args.output_dir))
