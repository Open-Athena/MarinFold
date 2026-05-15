# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 driver — generate / infer / evaluate / tokenizer.

Usage::

    python cli.py generate  --input cifs/  --num-docs 100 --out docs.parquet
    python cli.py infer     --model M     --input seqs/ --out preds.parquet
    python cli.py evaluate  --model M     --input pdbs/ --out metrics.json
    python cli.py tokenizer --save-local ./tok/
    python cli.py tokenizer --push open-athena/contacts-and-distances-v1-tokenizer

Each subcommand wires argparse → the relevant library module
(:mod:`generate`, :mod:`inference`) or directly into
:mod:`marinfold_document_structures` for the tokenizer path. All
file-IO conventions (parquet vs. json by suffix, per-example
sidecars) live in :mod:`marinfold_document_structures.writers`.
"""

import argparse
import sys
from pathlib import Path

from marinfold_document_structures import (
    build_tokenizer,
    write_docs,
    write_eval,
    write_predictions,
)

import generate
import inference
from vocab import CONTEXT_LENGTH, NAME, all_domain_tokens


def _seed_n_values(s: str) -> tuple[int, ...]:
    """Parse `--seed-n-values 0,5,20,50` into a tuple of ints."""
    out: list[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        n = int(token)
        if n < 0:
            raise argparse.ArgumentTypeError(f"seed-n value must be >= 0; got {n}")
        out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("--seed-n-values must include at least one value")
    return tuple(out)


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = generate.GenerationConfig(
        contact_cutoff_angstrom=args.contact_cutoff_angstrom,
        contact_f_range=tuple(args.contact_f_range),
        contact_rank_mean=args.contact_rank_mean,
        distance_rank_mean=args.distance_rank_mean,
        rank_std=args.rank_std,
        residue_plddt_min=args.residue_plddt_min,
    )
    docs = generate.generate_documents(
        input_path=args.input,
        num_docs=args.num_docs,
        context_length=args.context_length,
        config=cfg,
    )
    write_docs(args.out, docs, structure_name=NAME)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


def cmd_infer(args: argparse.Namespace) -> None:
    cfg = inference.InferenceConfig(
        model=args.model,
        input_path=args.input,
        seed_n_values=args.seed_n_values,
        query_atom=args.query_atom,
        top_k_logprobs=args.top_k_logprobs,
        batch_size=args.batch_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_pairs_per_structure=args.max_pairs_per_structure,
        keep_bin_probs=args.keep_bin_probs,
    )
    records = inference.predict(cfg)
    write_predictions(args.out, records, structure_name=NAME)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = inference.InferenceConfig(
        model=args.model,
        input_path=args.input,
        seed_n_values=args.seed_n_values,
        query_atom=args.query_atom,
        top_k_logprobs=args.top_k_logprobs,
        batch_size=args.batch_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_pairs_per_structure=args.max_pairs_per_structure,
        distance_cap_angstrom=args.distance_cap_angstrom,
    )
    result = inference.evaluate(cfg)
    write_eval(args.out, result, structure_name=NAME)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)
    for k, v in result.metrics.items():
        print(f"  {k} = {v}")


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


# --------------------------------------------------------------------------
# Argparse wiring
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python cli.py",
        description=f"{NAME} driver — generate / infer / evaluate / tokenizer.",
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
    # Algorithm knobs (defaults reproduce contactdoc).
    cfg_defaults = generate.GenerationConfig()
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
    p_gen.set_defaults(func=cmd_generate)

    # ---- shared inference args (used by infer + evaluate) ------------------
    def _add_inference_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True,
                       help="HuggingFace model path or local dir. Tokenizer "
                            "must be co-located.")
        p.add_argument("--input", type=Path, required=True,
                       help="Structure file (PDB / mmCIF / .gz) or directory. "
                            "For evaluate, the input IS the ground truth — "
                            "its coordinates are compared against predictions.")
        p.add_argument("--query-atom", default="CA",
                       help="Atom queried on both i and j (default CA).")
        p.add_argument("--seed-n-values", type=_seed_n_values, default=(0,),
                       help="Comma-separated seeded-contact counts "
                            "(e.g. '0,5,20,50'). Default '0' = zero-shot.")
        p.add_argument("--top-k-logprobs", type=int, default=128)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
        p.add_argument("--max-pairs-per-structure", type=int, default=None,
                       help="Cap pairs per structure (useful for smoke tests). "
                            "In evaluate mode this caps *evaluatable* pairs "
                            "(pairs with finite GT below --distance-cap-angstrom); "
                            "in infer mode it caps all queried pairs.")
        p.add_argument("--out", type=Path, required=True,
                       help="Output path.")

    # ---- infer -------------------------------------------------------------
    p_inf = sub.add_parser("infer", help="Run a trained model (no ground truth).")
    _add_inference_common(p_inf)
    p_inf.add_argument("--keep-bin-probs", action="store_true",
                       help="Include the full 64-bin distribution per pair "
                            "in the output records.")
    p_inf.set_defaults(func=cmd_infer)

    # ---- evaluate ----------------------------------------------------------
    p_eval = sub.add_parser("evaluate",
                            help="Run a trained model + score against ground truth.")
    _add_inference_common(p_eval)
    p_eval.add_argument("--distance-cap-angstrom", type=float, default=32.0,
                        help="GT distances above this are masked from MAE. "
                             "Anything above lands in the saturated bin.")
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- tokenizer ---------------------------------------------------------
    p_tok = sub.add_parser("tokenizer",
                           help="Build / save / push the v1 tokenizer.")
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
