# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold-document-structure`` — local CLI for poking at a document-structure impl.

Two subcommands:

* ``generate IMPL_PATH INPUT_PATH --num-docs N --context-length N --out OUT.parquet``
* ``evaluate IMPL_PATH MODEL_PATH GROUND_TRUTH_PATH --out OUT.json``

Both load the implementation by file path (``importlib`` spec from
file location) so experiment dirs don't need to be ``pip install``ed.
The implementation module must expose ``get_structure()`` or a
top-level ``STRUCTURE`` attribute (see ``interface.py``).

This CLI is for **local** smoke-testing only. Production data-gen and
evals go through ``data/`` and ``evals/`` respectively.

The CLI currently delegates input-iteration to the implementation:
the impl receives a path and a kwarg-pass-through dict, and is
responsible for opening / parsing / iterating. That keeps this CLI
agnostic to input format (PDB, mmCIF, parquet, jsonl, AFDB tar, …).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from marinfold_document_structures.interface import (
    DocumentStructure,
    EvalResult,
    load_structure,
)


def _iter_input(structure: DocumentStructure, input_path: Path) -> Any:
    """Ask the implementation for an iterator over its input records.

    Implementations should expose ``iter_inputs(path) -> Iterator``
    for the generate path and ``iter_ground_truth(path) -> Iterator``
    for the evaluate path. The CLI delegates so it doesn't need to
    know about file formats.
    """
    fn = getattr(structure, "iter_inputs", None)
    if fn is None:
        raise AttributeError(
            f"{type(structure).__name__} does not define iter_inputs(path); "
            "the marinfold-document-structure CLI delegates input parsing to "
            "the implementation. Add iter_inputs(self, path) returning an "
            "iterator over input records."
        )
    return fn(input_path)


def _iter_ground_truth(structure: DocumentStructure, gt_path: Path) -> Any:
    fn = getattr(structure, "iter_ground_truth", None)
    if fn is None:
        raise AttributeError(
            f"{type(structure).__name__} does not define iter_ground_truth(path); "
            "the marinfold-document-structure CLI delegates ground-truth parsing "
            "to the implementation. Add iter_ground_truth(self, path) returning "
            "an iterator over ground-truth records."
        )
    return fn(gt_path)


def _cmd_generate(args: argparse.Namespace) -> int:
    structure = load_structure(args.impl)
    print(
        f"[marinfold-document-structure] loaded {structure.name!r} from {args.impl}",
        file=sys.stderr, flush=True,
    )
    ctx_len = args.context_length if args.context_length is not None else structure.context_length

    docs = structure.generate_documents(
        _iter_input(structure, args.input),
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
        ground_truth_records=_iter_ground_truth(structure, args.ground_truth),
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
        # Write the summary scalars as a single row, plus a sibling
        # parquet for per-example records if any.
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
