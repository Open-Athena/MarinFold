# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Output writers shared by document-structure ``cli.py`` drivers.

The three subcommands every impl typically exposes —
``generate`` / ``infer`` / ``evaluate`` — produce the same three
shapes of output, so the writers live here rather than being
re-implemented per impl. Each writer accepts the impl's
``structure_name`` so the column shows up on every row in the
output (and survives whichever serialization format the user picks
via the file suffix).

All three writers dispatch on ``out.suffix``:

- ``.parquet`` (preferred for downstream tooling)
- ``.jsonl`` / ``.json`` (handy for spot-checking)

Anything else is a hard error — callers should add new formats here
rather than working around in their CLI.
"""

import json
import sys
from pathlib import Path
from typing import Any, Iterable

from marinfold_document_structures.core import EvalResult


def write_docs(out: Path, docs: Iterable[str], *, structure_name: str) -> None:
    """Write generated training documents to parquet or jsonl.

    Each output row is ``{"document": <string>, "structure": <name>}``.
    Errors loudly when the generator produced no documents (a silent
    empty file would mask a broken generator).
    """
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


def write_predictions(
    out: Path, records: Iterable[dict], *, structure_name: str
) -> None:
    """Write per-input inference records to parquet or jsonl.

    Each record is the impl's prediction dict; the writer prepends a
    ``structure`` column so the rows are unambiguous when concatenated
    with output from other impls.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise SystemExit("inference produced 0 records")
    rows = [{"structure": structure_name, **r} for r in records]
    if out.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), str(out), compression="zstd")
    elif out.suffix in (".jsonl", ".json"):
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
    else:
        raise SystemExit(f"--out must end in .parquet or .jsonl; got {out}")


def write_eval(out: Path, result: EvalResult, *, structure_name: str) -> None:
    """Write an :class:`EvalResult` to parquet or json.

    ``.json``: a single object with ``metrics`` / ``extras`` /
    ``n_examples``; if ``per_example`` is non-empty, it's written to a
    sibling ``_per_example.jsonl`` so the main file stays small.

    ``.parquet``: a single-row table with ``metric_<name>`` columns,
    plus every scalar / list entry from ``result.extras`` flattened
    into its own column. Nested dicts (which parquet can't infer a
    schema for) get JSON-stringified into an ``extras_json`` column so
    callers can still recover them.
    """
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
            print(f"[marinfold] wrote per-example to {per_path}", file=sys.stderr)
    elif out.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        flat: dict[str, Any] = {
            "structure": structure_name,
            "n_examples": summary["n_examples"],
        }
        for k, v in result.metrics.items():
            flat[f"metric_{k}"] = v
        nested: dict[str, Any] = {}
        for k, v in result.extras.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
            elif isinstance(v, list) and all(
                isinstance(x, (str, int, float, bool)) or x is None for x in v
            ):
                flat[k] = v
            else:
                nested[k] = v
        if nested:
            flat["extras_json"] = json.dumps(nested, default=str)
        pq.write_table(pa.Table.from_pylist([flat]), str(out), compression="zstd")
        if result.per_example:
            per_path = out.with_name(out.stem + "_per_example.parquet")
            pq.write_table(
                pa.Table.from_pylist(result.per_example),
                str(per_path),
                compression="zstd",
            )
            print(f"[marinfold] wrote per-example to {per_path}", file=sys.stderr)
    else:
        raise SystemExit(f"--out must end in .json or .parquet; got {out}")
