# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Local driver for bio2token-v1 document generation.

This is the slow, single-process path — no Zephyr, runs anywhere torch
installs. It parses each input structure, tokenizes it with the pure-PyTorch
bio2token encoder, and writes one document row per structure to parquet/jsonl.
The Zephyr/Iris at-scale runner (tomorrow) will reuse ``generate.py`` as its
per-row worker; this CLI is for local development and smoke checks.

Examples::

    uv run python cli.py generate --input tests/data/1QYS.cif --out /tmp/docs.jsonl
    uv run python cli.py generate --input /path/to/cifs/ --out /tmp/docs.parquet
    uv run python cli.py vocab            # print vocabulary size
"""

import argparse
import json
import sys
import time
from pathlib import Path

import generate as gen
from vocab import all_tokens

_STRUCTURE_EXTS = (".cif", ".cif.gz", ".mmcif", ".mmcif.gz",
                   ".pdb", ".pdb.gz", ".ent", ".ent.gz")


def _iter_structure_paths(path: Path):
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name.lower().endswith(_STRUCTURE_EXTS):
            yield p


def _write_rows(out: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("generator produced 0 documents")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), str(out), compression="zstd")
    elif out.suffix in (".jsonl", ".json"):
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        raise SystemExit(f"--out must end in .parquet or .jsonl; got {out}")


def cmd_generate(args: argparse.Namespace) -> None:
    paths = list(_iter_structure_paths(Path(args.input)))
    if args.num_docs is not None:
        paths = paths[: args.num_docs]
    rows, skipped = [], 0
    for i, p in enumerate(paths, 1):
        t = time.perf_counter()
        row = gen.generate_document(
            str(p), device=args.device, max_context=args.max_context)
        if row is None:
            skipped += 1
            print(f"[{i}/{len(paths)}] {p.name}: skipped (> max-context)", file=sys.stderr)
            continue
        rows.append(row)
        print(f"[{i}/{len(paths)}] {p.name}: {row['num_atoms']} atoms, "
              f"{row['num_tokens']} tokens, {1000*(time.perf_counter()-t):.0f} ms",
              file=sys.stderr)
    _write_rows(Path(args.out), rows)
    print(f"[{gen.NAME}] wrote {len(rows)} docs to {args.out} "
          f"({skipped} skipped)", file=sys.stderr)


def cmd_vocab(args: argparse.Namespace) -> None:
    toks = all_tokens()
    print(f"bio2token-v1 vocabulary: {len(toks)} tokens")
    print("first 8:", toks[:8])
    print("last 4 :", toks[-4:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python cli.py",
                                     description="bio2token-v1 document generation (local).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="Generate documents from structure file(s).")
    p.add_argument("--input", required=True, help="Structure file or directory.")
    p.add_argument("--out", required=True, help="Output path (.parquet or .jsonl).")
    p.add_argument("--device", default="cpu", help="torch device (cpu/mps/cuda/xla).")
    p.add_argument("--num-docs", type=int, default=None, help="Cap number of structures.")
    p.add_argument("--max-context", type=int, default=None,
                   help="Skip structures whose document exceeds this token count.")
    p.set_defaults(func=cmd_generate)

    p_vocab = sub.add_parser("vocab", help="Print the vocabulary.")
    p_vocab.set_defaults(func=cmd_vocab)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
