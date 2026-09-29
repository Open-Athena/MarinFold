"""Export one native exp299 checkpoint with its exact V2 tokenizer."""

import argparse
from pathlib import Path

import jmp
from levanter.main.export_lm_to_hf import ConvertLmConfig, main as export_main
from levanter.trainer import TrainerConfig

from dispatch_delta_stream_full import MODEL_CONFIG, _mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    args = parser.parse_args()
    export_main(
        ConvertLmConfig(
            checkpoint_path=args.checkpoint,
            output_dir=args.out,
            tokenizer=str(args.tokenizer),
            model=MODEL_CONFIG,
            trainer=TrainerConfig(
                id="exp299-delta-export",
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                mesh=_mesh(1),
            ),
            use_cpu=True,
        )
    )


if __name__ == "__main__":
    main()
