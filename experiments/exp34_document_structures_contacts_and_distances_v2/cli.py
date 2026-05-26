# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v2 driver — generate / tokenizer.

Usage::

    python cli.py generate  --input cifs/ --num-docs 100 --out docs.parquet
    python cli.py tokenizer --save-local ./tok/
    python cli.py tokenizer --push open-athena/contacts-and-distances-v2-tokenizer

Scope is intentionally narrower than v1's cli: this experiment only
implements training-data *generation*. Inference and evaluation will
land in follow-up experiments (one to run generation at scale, then
one to train on the result, then one to eval the trained model).
"""

import argparse
import sys
from pathlib import Path

from marinfold import (
    build_tokenizer,
    write_docs,
)

import generate
from vocab import CONTEXT_LENGTH, NAME, all_domain_tokens


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = generate.GenerationConfig(
        contact_cutoff_angstrom=args.contact_cutoff_angstrom,
        contact_f_range=tuple(args.contact_f_range),
        contact_rank_mean=args.contact_rank_mean,
        distance_rank_mean=args.distance_rank_mean,
        rank_std=args.rank_std,
        residue_plddt_min=args.residue_plddt_min,
        think_initial_prob=args.think_initial_prob,
        think_initial_geom_p=args.think_initial_geom_p,
        think_additional_count_range=tuple(args.think_additional_count_range),
        think_run_length_geom_p=args.think_run_length_geom_p,
    )
    docs = generate.generate_documents(
        input_path=args.input,
        num_docs=args.num_docs,
        context_length=args.context_length,
        config=cfg,
    )
    write_docs(args.out, docs, structure_name=NAME)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


def cmd_tokenizer(args: argparse.Namespace) -> None:
    tokenizer = build_tokenizer(all_domain_tokens())
    print(f"[{NAME}] built tokenizer with {len(tokenizer)} tokens", file=sys.stderr)
    did_anything = False
    if args.save_local is not None:
        out_dir = Path(args.save_local)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[{NAME}] saved tokenizer to {out_dir}", file=sys.stderr)
        did_anything = True
    if args.push is not None:
        tokenizer.push_to_hub(args.push, private=args.private)
        print(
            f"[{NAME}] pushed tokenizer to https://huggingface.co/{args.push}",
            file=sys.stderr,
        )
        did_anything = True
    if not did_anything:
        sample = " ".join(all_domain_tokens()[:8])
        encoded = tokenizer.encode(sample, add_special_tokens=False)
        print(f"sample: {sample!r}")
        print(f"  ids:  {encoded}")
        print(f"  back: {tokenizer.decode(encoded)!r}")
        print("Use --save-local DIR or --push REPO to persist the tokenizer.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=f"{NAME} driver — generate / tokenizer.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- generate ----------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate training documents.")
    p_gen.add_argument("--input", type=Path, required=True,
                       help="PDB / mmCIF (.gz) file or directory of them.")
    p_gen.add_argument("--num-docs", type=int, default=None,
                       help="Cap on docs produced (default: one per input).")
    p_gen.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                       help="Token budget per document.")
    p_gen.add_argument("--out", type=Path, required=True,
                       help="Output path (.parquet or .jsonl).")
    cfg_defaults = generate.GenerationConfig()
    # ---- v1-inherited knobs ----
    p_gen.add_argument("--contact-cutoff-angstrom", type=float,
                       default=cfg_defaults.contact_cutoff_angstrom)
    p_gen.add_argument("--residue-plddt-min", type=float,
                       default=cfg_defaults.residue_plddt_min)
    p_gen.add_argument("--contact-f-range", type=float, nargs=2,
                       metavar=("LOW", "HIGH"),
                       default=list(cfg_defaults.contact_f_range))
    p_gen.add_argument("--contact-rank-mean", type=float,
                       default=cfg_defaults.contact_rank_mean)
    p_gen.add_argument("--distance-rank-mean", type=float,
                       default=cfg_defaults.distance_rank_mean)
    p_gen.add_argument("--rank-std", type=float, default=cfg_defaults.rank_std)
    # ---- v2 think-token knobs ----
    p_gen.add_argument("--think-initial-prob", type=float,
                       default=cfg_defaults.think_initial_prob,
                       help="P(any think tokens right after <begin_statements>). "
                            "Default 0.75 per issue #34.")
    p_gen.add_argument("--think-initial-geom-p", type=float,
                       default=cfg_defaults.think_initial_geom_p,
                       help="Geometric p for the initial run length (support >= 1). "
                            "Default 0.13.")
    p_gen.add_argument("--think-additional-count-range", type=float, nargs=2,
                       metavar=("LOW", "HIGH"),
                       default=list(cfg_defaults.think_additional_count_range),
                       help="Uniform range for k2; n_additional_runs = max(int(k2), 0). "
                            "Default (-4, 4).")
    p_gen.add_argument("--think-run-length-geom-p", type=float,
                       default=cfg_defaults.think_run_length_geom_p,
                       help="Geometric p for the length of each additional run. "
                            "Default 0.25.")
    p_gen.set_defaults(func=cmd_generate)

    # ---- tokenizer ---------------------------------------------------------
    p_tok = sub.add_parser("tokenizer",
                           help="Build / save / push the v2 tokenizer.")
    p_tok.add_argument("--save-local", type=Path, default=None,
                       help="Save via tokenizer.save_pretrained() to this dir.")
    p_tok.add_argument("--push", type=str, default=None,
                       help="Push to this HF Hub repo "
                            "(e.g. open-athena/<name>-tokenizer).")
    p_tok.add_argument("--private", action="store_true",
                       help="If --push, create the repo private.")
    p_tok.set_defaults(func=cmd_tokenizer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
