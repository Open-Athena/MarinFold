"""Export one native exp299 checkpoint with its exact V2 tokenizer."""

import argparse
from pathlib import Path

import jmp
from dispatch_delta_stream_full import MODEL_CONFIG, _mesh
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.main.export_lm_to_hf import main as export_main
from levanter.trainer import TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--parameter-dtype",
        choices=("f32", "bfloat16"),
        default="f32",
        help="Parameter dtype for the exported weights.",
    )
    args = parser.parse_args()
    export_main(
        ConvertLmConfig(
            checkpoint_path=args.checkpoint,
            output_dir=args.out,
            tokenizer=str(args.tokenizer),
            model=MODEL_CONFIG,
            trainer=TrainerConfig(
                id="exp299-delta-export",
                mp=jmp.get_policy(f"p={args.parameter_dtype},c=bfloat16"),
                mesh=_mesh(1),
            ),
            use_cpu=True,
        )
    )


if __name__ == "__main__":
    main()
