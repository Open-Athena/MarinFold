# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot: zero the ``global_plddt`` column in already-generated shards.

``curate_and_generate.py`` now writes 0.0 for this column (see the comment
there for why), but the corpora were generated before that change and
regenerating them costs 2.7 hours to alter one float per row. This rewrites
the column in place instead, and asserts that nothing else moved.

Usage::

    uv run python zero_global_plddt.py --root /data/exp222_pdb_curation
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def rewrite(shard: Path) -> tuple[int, float, float]:
    """Zero ``global_plddt`` in one shard; return (rows, min_before, max_before)."""
    table = pq.read_table(shard)
    column = table.column("global_plddt")
    before = (column[0].as_py(), column[-1].as_py()) if table.num_rows else (0.0, 0.0)

    index = table.schema.get_field_index("global_plddt")
    zeroed = table.set_column(
        index,
        table.schema.field(index),
        pa.chunked_array([pa.array([0.0] * table.num_rows, type=pa.float64())]),
    )

    # Everything except that one column must be untouched.
    assert zeroed.schema == table.schema, "schema changed"
    assert zeroed.num_rows == table.num_rows, "row count changed"
    for name in table.schema.names:
        if name == "global_plddt":
            continue
        assert zeroed.column(name).equals(table.column(name)), f"{name} changed"

    pq.write_table(zeroed, shard, compression="zstd")
    return table.num_rows, before[0], before[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/exp222_pdb_curation"))
    args = parser.parse_args(argv)

    total = 0
    for subset in ("monomers", "multimers", "deduped"):
        directory = args.root / "docs" / subset
        if not directory.is_dir():
            continue
        shards = sorted(directory.glob("*.parquet"))
        rows = 0
        samples = []
        for shard in shards:
            n, first, last = rewrite(shard)
            rows += n
            if len(samples) < 2:
                samples.append(round(first, 2))
        print(f"{subset}: zeroed {rows} rows across {len(shards)} shards "
              f"(was e.g. {samples})")
        total += rows

    # Read back through the dataset API to confirm.
    import pyarrow.dataset as ds
    for subset in ("monomers", "multimers", "deduped"):
        directory = args.root / "docs" / subset
        if not directory.is_dir():
            continue
        column = ds.dataset(directory, format="parquet").to_table(
            columns=["global_plddt"]
        ).column("global_plddt")
        assert all(v == 0.0 for v in column.to_pylist()), f"{subset} not fully zeroed"
        print(f"{subset}: verified all {len(column)} values are 0.0")

    print(f"done: {total} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
