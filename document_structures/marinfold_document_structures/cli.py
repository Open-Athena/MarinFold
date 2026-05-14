# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""``marinfold-document-structure`` — local CLI for document-structure impls.

Four subcommands:

* ``generate IMPL_DIR --out OUT.parquet [impl-specific args...]``
* ``infer    IMPL_DIR --out OUT.parquet [impl-specific args...]``
* ``evaluate IMPL_DIR --out OUT.json    [impl-specific args...]``
* ``tokenizer IMPL_DIR [--save-local DIR] [--push REPO]``

``IMPL_DIR`` is the experiment directory containing ``generate.py``
+ ``inference.py``. The CLI loads the appropriate file by path
(``importlib.util.spec_from_file_location``), then asks the impl to
register its own args (``add_args(parser)`` or
``add_args(parser, subcommand=...)``) before final parsing.

Two-pass parsing: first pass extracts ``IMPL_DIR``, loads the impl;
the impl then populates the rest of the parser; second pass yields
the final ``args`` namespace.

The CLI owns only ``IMPL_DIR`` and ``--out``. Everything else (input
paths, model paths, batch sizes, atom choices, …) is the impl's
responsibility.

For local smoke-testing only. Production data-gen and evals go
through ``data/`` and ``evals/`` respectively.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from marinfold_document_structures.interface import (
    EvalResult,
    build_tokenizer,
    load_generator,
    load_inference,
)


# --------------------------------------------------------------------------
# Two-pass parsing helpers
# --------------------------------------------------------------------------


def _base_parser(prog: str) -> argparse.ArgumentParser:
    """First-pass parser — just ``impl_dir`` + a permissive ``--out``.

    ``add_help=False`` so that a ``--help`` in argv falls through to
    the second-pass parser (which has the impl's args).
    """
    p = argparse.ArgumentParser(prog=prog, add_help=False)
    p.add_argument("impl_dir", type=Path)
    p.add_argument("--out", type=Path, required=False)
    return p


def _full_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "impl_dir", type=Path,
        help="Experiment dir containing generate.py + inference.py.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output path.")
    return p


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------


def _write_docs(out: Path, docs, *, structure_name: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [{"document": d, "structure": structure_name} for d in docs]
        if not rows:
            raise SystemExit("generator produced 0 documents")
        pq.write_table(pa.Table.from_pylist(rows), str(out), compression="zstd")
    elif out.suffix in (".jsonl", ".json"):
        n = 0
        with open(out, "w") as f:
            for d in docs:
                f.write(json.dumps({"document": d, "structure": structure_name}) + "\n")
                n += 1
        if n == 0:
            raise SystemExit("generator produced 0 documents")
    else:
        raise SystemExit(f"--out must end in .parquet or .jsonl; got {out}")


def _write_predictions(out: Path, records, *, structure_name: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise SystemExit("inference produced 0 records")
    if out.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(records), str(out), compression="zstd")
    elif out.suffix in (".jsonl", ".json"):
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps({"structure": structure_name, **r}, default=str) + "\n")
    else:
        raise SystemExit(f"--out must end in .parquet or .jsonl; got {out}")


def _write_eval(out: Path, result: EvalResult, *, structure_name: str) -> None:
    summary: dict[str, Any] = {
        "structure": structure_name,
        "metrics": result.metrics,
        "extras": result.extras,
        "n_examples": len(result.per_example),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        if result.per_example:
            per_path = out.with_name(out.stem + "_per_example.jsonl")
            with open(per_path, "w") as f:
                for r in result.per_example:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"[marinfold-document-structure] wrote per-example to {per_path}", file=sys.stderr)
    elif out.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        flat = {"structure": structure_name, "n_examples": summary["n_examples"]}
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


# --------------------------------------------------------------------------
# Subcommand entry points
# --------------------------------------------------------------------------


def _cmd_generate(argv: list[str]) -> int:
    base = _base_parser("marinfold-document-structure generate")
    base_args, _ = base.parse_known_args(argv)
    generator = load_generator(base_args.impl_dir)
    print(
        f"[marinfold-document-structure] loaded generator {generator.name!r} "
        f"from {base_args.impl_dir}/generate.py",
        file=sys.stderr, flush=True,
    )

    full = _full_parser(
        "marinfold-document-structure generate",
        f"Generate {generator.name!r} training documents.",
    )
    generator.add_args(full)
    args = full.parse_args(argv)

    docs = generator.run(args)
    _write_docs(args.out, docs, structure_name=generator.name)
    print(f"[marinfold-document-structure] wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_infer(argv: list[str]) -> int:
    base = _base_parser("marinfold-document-structure infer")
    base_args, _ = base.parse_known_args(argv)
    inference = load_inference(base_args.impl_dir)
    print(
        f"[marinfold-document-structure] loaded inference {inference.name!r} "
        f"from {base_args.impl_dir}/inference.py",
        file=sys.stderr, flush=True,
    )

    full = _full_parser(
        "marinfold-document-structure infer",
        f"Run {inference.name!r} inference (no ground truth).",
    )
    inference.add_args(full, subcommand="infer")
    args = full.parse_args(argv)

    records = inference.predict(args)
    _write_predictions(args.out, records, structure_name=inference.name)
    print(f"[marinfold-document-structure] wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_evaluate(argv: list[str]) -> int:
    base = _base_parser("marinfold-document-structure evaluate")
    base_args, _ = base.parse_known_args(argv)
    inference = load_inference(base_args.impl_dir)
    print(
        f"[marinfold-document-structure] loaded inference {inference.name!r} "
        f"from {base_args.impl_dir}/inference.py",
        file=sys.stderr, flush=True,
    )

    full = _full_parser(
        "marinfold-document-structure evaluate",
        f"Evaluate {inference.name!r} against ground truth.",
    )
    inference.add_args(full, subcommand="evaluate")
    args = full.parse_args(argv)

    result = inference.evaluate(args)
    _write_eval(args.out, result, structure_name=inference.name)
    print(f"[marinfold-document-structure] wrote {args.out}", file=sys.stderr)
    for k, v in result.metrics.items():
        print(f"  {k} = {v}")
    return 0


def _cmd_tokenizer(argv: list[str]) -> int:
    """Build the tokenizer implied by the impl's vocab and optionally save / push."""
    p = argparse.ArgumentParser(
        prog="marinfold-document-structure tokenizer",
        description=(
            "Build the WordLevel tokenizer implied by the impl's tokens(). "
            "Prefers generate.py (the format spec); falls back to "
            "inference.py — both must agree on the vocab."
        ),
    )
    p.add_argument("impl_dir", type=Path)
    p.add_argument(
        "--save-local", type=Path, default=None,
        help="Save via tokenizer.save_pretrained() to this directory.",
    )
    p.add_argument(
        "--push", type=str, default=None,
        help="Push to this HF Hub repo (e.g. open-athena/<name>-tokenizer).",
    )
    p.add_argument("--private", action="store_true", help="If --push, create the repo private.")
    args = p.parse_args(argv)

    impl_dir = Path(args.impl_dir).resolve()
    if (impl_dir / "generate.py").is_file():
        component = load_generator(impl_dir)
    else:
        component = load_inference(impl_dir)
    print(
        f"[marinfold-document-structure] loaded {component.name!r} from {impl_dir}",
        file=sys.stderr, flush=True,
    )

    tokenizer = build_tokenizer(component)
    print(
        f"[marinfold-document-structure] built tokenizer with {len(tokenizer)} tokens",
        file=sys.stderr,
    )

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
        sample = " ".join(component.tokens()[:8])
        encoded = tokenizer.encode(sample, add_special_tokens=False)
        print(f"sample: {sample!r}")
        print(f"  ids:  {encoded}")
        print(f"  back: {tokenizer.decode(encoded)!r}")
        print("Use --save-local DIR or --push REPO to persist the tokenizer.")
    return 0


# --------------------------------------------------------------------------
# Top-level dispatcher
# --------------------------------------------------------------------------


_COMMANDS = {
    "generate": _cmd_generate,
    "infer": _cmd_infer,
    "evaluate": _cmd_evaluate,
    "tokenizer": _cmd_tokenizer,
}

_USAGE = """\
usage: marinfold-document-structure <command> <impl_dir> [args...]

Commands:
  generate   Generate training documents from input structures.
  infer      Run a trained model on inputs (no ground truth).
  evaluate   Run a trained model + compare against ground truth.
  tokenizer  Build / save / push the tokenizer implied by the impl's vocab.

Each subcommand takes <impl_dir> (the experiment dir containing
generate.py + inference.py). Run `<command> <impl_dir> --help` for
the impl's specific args.
"""


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE.rstrip())
        return 0
    cmd, *rest = argv
    func = _COMMANDS.get(cmd)
    if func is None:
        print(f"unknown command: {cmd!r}\n\n{_USAGE}", file=sys.stderr)
        return 2
    return func(rest)


if __name__ == "__main__":
    sys.exit(main())
