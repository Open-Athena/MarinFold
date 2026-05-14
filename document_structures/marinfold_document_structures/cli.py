# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold-document-structure`` — local CLI for poking at a document-structure impl.

Three subcommands:

* ``generate IMPL_PATH INPUT_PATH --num-docs N --context-length N --out OUT.parquet``
* ``evaluate IMPL_PATH MODEL_PATH GROUND_TRUTH_PATH --out OUT.json``
* ``tokenizer IMPL_PATH [--save-local DIR] [--push REPO]``

All three load the implementation by file path (``importlib`` spec
from file location) so experiment dirs don't need to be ``pip
install``ed. The implementation module must expose ``get_structure()``
or a top-level ``STRUCTURE`` attribute (see ``interface.py``).

This CLI is for **local** smoke-testing only. Production data-gen and
evals go through ``data/`` and ``evals/`` respectively.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from marinfold_document_structures.interface import (
    DocumentStructure,
    EvalResult,
    build_tokenizer,
    load_structure,
)


def _cmd_generate(args: argparse.Namespace) -> int:
    structure = load_structure(args.impl)
    print(
        f"[marinfold-document-structure] loaded {structure.name!r} from {args.impl}",
        file=sys.stderr, flush=True,
    )
    ctx_len = args.context_length if args.context_length is not None else structure.context_length

    docs = structure.generate_documents(
        structure.iter_inputs(args.input),
        context_length=ctx_len,
        num_docs=args.num_docs,
    )

    out = Path(args.out)
    if out.suffix == ".parquet":
        _write_parquet(out, docs, structure_name=structure.name, context_length=ctx_len)
    elif out.suffix in (".jsonl", ".json"):
        _write_jsonl(out, docs, structure_name=structure.name, context_length=ctx_len)
    else:
        raise SystemExit(f"--out must end in .parquet or .jsonl; got {out}")

    print(f"[marinfold-document-structure] wrote {out}", file=sys.stderr)
    return 0


def _write_parquet(out: Path, docs, *, structure_name: str, context_length: int) -> None:
    # Defer pyarrow import so `--help` works without it installed.
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {"document": doc, "structure": structure_name, "context_length": context_length}
        for doc in docs
    ]
    if not rows:
        raise SystemExit(
            "Implementation produced 0 documents — check the input path and --num-docs."
        )
    table = pa.Table.from_pylist(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out), compression="zstd")


def _write_jsonl(out: Path, docs, *, structure_name: str, context_length: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for doc in docs:
            f.write(json.dumps({
                "document": doc,
                "structure": structure_name,
                "context_length": context_length,
            }) + "\n")
            n += 1
    if n == 0:
        raise SystemExit(
            "Implementation produced 0 documents — check the input path and --num-docs."
        )


def _cmd_evaluate(args: argparse.Namespace) -> int:
    structure = load_structure(args.impl)
    print(
        f"[marinfold-document-structure] loaded {structure.name!r} from {args.impl}",
        file=sys.stderr, flush=True,
    )

    result: EvalResult = structure.evaluate(
        model_path=args.model,
        ground_truth_records=structure.iter_ground_truth(args.ground_truth),
    )

    summary: dict[str, Any] = {
        "structure": structure.name,
        "model": args.model,
        "metrics": result.metrics,
        "extras": result.extras,
        "n_examples": len(result.per_example),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
    elif out.suffix == ".parquet":
        # Summary scalars as a single row; per-example records as a sibling.
        import pyarrow as pa
        import pyarrow.parquet as pq

        flat = {"structure": structure.name, "model": args.model, "n_examples": summary["n_examples"]}
        for k, v in result.metrics.items():
            flat[f"metric_{k}"] = v
        pq.write_table(pa.Table.from_pylist([flat]), str(out), compression="zstd")
        if result.per_example:
            per_path = out.with_name(out.stem + "_per_example.parquet")
            pq.write_table(
                pa.Table.from_pylist(result.per_example),
                str(per_path),
                compression="zstd",
            )
            print(f"[marinfold-document-structure] wrote per-example to {per_path}", file=sys.stderr)
    else:
        raise SystemExit(f"--out must end in .json or .parquet; got {out}")

    print(f"[marinfold-document-structure] wrote {out}", file=sys.stderr)
    for k, v in result.metrics.items():
        print(f"  {k} = {v}")
    return 0


def _cmd_tokenizer(args: argparse.Namespace) -> int:
    structure = load_structure(args.impl)
    print(
        f"[marinfold-document-structure] loaded {structure.name!r} from {args.impl}",
        file=sys.stderr, flush=True,
    )
    tokenizer = build_tokenizer(structure)
    n = len(tokenizer)
    print(f"[marinfold-document-structure] built tokenizer with {n} tokens", file=sys.stderr)

    did_anything = False
    if args.save_local is not None:
        out_dir = Path(args.save_local)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[marinfold-document-structure] saved to {out_dir}", file=sys.stderr)
        did_anything = True
    if args.push is not None:
        tokenizer.push_to_hub(args.push, private=args.private)
        print(
            f"[marinfold-document-structure] pushed to https://huggingface.co/{args.push}",
            file=sys.stderr,
        )
        did_anything = True
    if not did_anything:
        # Spot-check round-trip and print a sample so the user sees something useful.
        sample_tokens = structure.tokens()[:8]
        sample = " ".join(sample_tokens)
        encoded = tokenizer.encode(sample, add_special_tokens=False)
        print(f"sample: {sample!r}")
        print(f"  ids:  {encoded}")
        print(f"  back: {tokenizer.decode(encoded)!r}")
        print("Use --save-local DIR or --push REPO to persist the tokenizer.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marinfold-document-structure",
        description="Local CLI for poking at a MarinFold document-structure implementation.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser(
        "generate",
        help="Generate training documents from input data using an impl.",
    )
    gen.add_argument("impl", type=Path, help="Path to the impl .py file.")
    gen.add_argument("input", type=Path, help="Path to input data (impl decides format).")
    gen.add_argument("--num-docs", type=int, default=None, help="Cap on docs to generate.")
    gen.add_argument(
        "--context-length", type=int, default=None,
        help="Max token length per doc. Defaults to the impl's structure.context_length.",
    )
    gen.add_argument("--out", type=Path, required=True, help="Output parquet or jsonl.")
    gen.set_defaults(func=_cmd_generate)

    ev = sub.add_parser(
        "evaluate",
        help="Run an impl's evaluate() on a model + ground-truth corpus.",
    )
    ev.add_argument("impl", type=Path, help="Path to the impl .py file.")
    ev.add_argument("model", type=str, help="Model handle (impl decides how to load it).")
    ev.add_argument("ground_truth", type=Path, help="Path to ground-truth data.")
    ev.add_argument("--out", type=Path, required=True, help="Output json or parquet.")
    ev.set_defaults(func=_cmd_evaluate)

    tk = sub.add_parser(
        "tokenizer",
        help="Build the tokenizer implied by an impl's tokens() and optionally save / push it.",
    )
    tk.add_argument("impl", type=Path, help="Path to the impl .py file.")
    tk.add_argument(
        "--save-local", type=Path, default=None,
        help="Save the constructed tokenizer to this directory (via tokenizer.save_pretrained).",
    )
    tk.add_argument(
        "--push", type=str, default=None,
        help="Push to this HF Hub repo (e.g. open-athena/contacts-and-distances-v1-tokenizer).",
    )
    tk.add_argument(
        "--private", action="store_true",
        help="If --push, create the repo as private.",
    )
    tk.set_defaults(func=_cmd_tokenizer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
