# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A — selection: afdb-24M manifest -> contacts-v1 round manifest.

This is the *new* logic for exp53. It is cheap and structure-free: it
reads only small columns of the afdb-24M manifest (never ``cif_content``),
groups rows by structural cluster, ranks each cluster's members by pLDDT,
assigns *rounds*, and drops clusters too small to fill the early rounds.
**No pyconfind / no structure parsing happens here** — that is Stage B
(``cli.py``), which fetches + generates a document for each row this stage
selects.

Rounds (mirrors ``timodonnell/protein-docs`` ``*-5x`` subsets, with two
issue-#53 changes):

* For each ``struct_cluster_id`` we keep up to ``num_rounds`` (=5) members,
  ordered by ``global_plddt`` descending (deterministic ``entry_id`` tie
  break). ``round = 0`` is the highest-pLDDT representative, ``round = 1``
  the next, and so on.
* **Change 1 — drop tiny clusters.** Clusters with fewer than
  ``min_cluster_size`` (=3) usable members are discarded entirely. This
  makes round-0/1/2 all the same (largest) size rather than round-0 being
  inflated by singletons. "Usable" = passes the ``seq_len`` pre-filter,
  since a member contacts-v1 can't serialize is not a usable example.
* **Change 2 — reverse order.** The manifest is written as *single-round*
  shards numbered in **descending** round order per split (round-4 shards
  first ... round-0 shards last). Stage B maps each manifest shard to one
  output shard, so the published corpus is physically ordered highest-round
  first / round-0 last — i.e. the highest-pLDDT data is trained on last.
  Every row also carries an explicit ``round`` column.

The ``seq_len`` pre-filter (``[min_seq_len, max_seq_len]`` = ``[2, 2000]``)
matches contacts-v1's serializable range (``>=2`` residues, ``<=2000``
position indices) so we don't select members that Stage B would only drop.

(The module is named ``selection`` — not ``select`` — to avoid shadowing
Python's stdlib ``select``, which ``gcsfs``/``asyncio`` import.)

Run it::

    # local smoke over cached afdb-24M shards
    uv run python selection.py \
        --input /path/to/afdb-24M/snapshots/<sha> \
        --out /tmp/exp53_manifest

    # full scan straight from HuggingFace (afdb-24M is public)
    uv run python selection.py \
        --input "hf://datasets/timodonnell/afdb-24M/**/*.parquet" \
        --out gs://.../selection_manifest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

# Small columns copied from the afdb-24M manifest onto every selected row.
# Explicitly excludes the bulky ``cif_content`` (Stage B fetches structures
# from ``gcs_uri`` instead). ``struct_cluster_id`` is the dedup key;
# ``global_plddt`` is the round-ranking key; ``gcs_uri`` is Stage B's input.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "entry_id",
    "gcs_uri",
    "struct_cluster_id",
    "seq_cluster_id",
    "global_plddt",
    "seq_len",
    "split",
    "uniprot_accession",
    "tax_id",
    "organism_name",
)

DEFAULT_MIN_SEQ_LEN = 2
DEFAULT_MAX_SEQ_LEN = 2000
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_NUM_ROUNDS = 5
DEFAULT_SHARD_SIZE = 2000


def _as_parquet_glob(input_path: str) -> str:
    """Normalize ``--input`` into something ``read_parquet`` can expand.

    A literal ``.parquet`` file/glob is used as-is; a directory becomes
    ``<dir>/**/*.parquet`` (afdb-24M nests shards under ``shard_XXXXXX-YYYYYY/``).
    Remote URIs (``hf://``, ``gs://``) are assumed to already be a glob.
    """
    if input_path.endswith(".parquet") or "*" in input_path:
        return input_path
    if "://" in input_path:
        return input_path.rstrip("/") + "/**/*.parquet"
    p = Path(input_path)
    if p.is_dir():
        return str(p / "**" / "*.parquet")
    return input_path


def _sql_literal(text: str) -> str:
    """Single-quote a string for inline use in SQL (CLI-trusted input)."""
    return "'" + text.replace("'", "''") + "'"


def _connect(input_literal: str) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection wired to read any remote URI in ``input_literal``.

    We register the matching **fsspec** filesystem (``hf://`` via
    ``huggingface_hub``, ``gs://`` via ``gcsfs``) rather than relying on
    duckdb's bundled ``httpfs`` extension — the extension's auto-download
    is brittle across duckdb builds, and fsspec reuses our existing HF /
    GCS auth.
    """
    con = duckdb.connect()
    protocols = [p for p in ("hf", "gs", "gcs", "s3") if f"{p}://" in input_literal]
    if protocols:
        import fsspec

        for proto in protocols:
            con.register_filesystem(fsspec.filesystem(proto))
    return con


def build_filtered_table(
    con: duckdb.DuckDBPyConnection,
    input_glob: str,
    *,
    min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    table: str = "filtered",
) -> None:
    """Materialize *all* seq_len-passing rows, with round + cluster_size.

    **This is the only scan of the (possibly remote) input.** Everything
    downstream — the selection manifest and the summary stats — is derived
    from this local temp table, so we never re-read afdb-24M over the
    network. Applies the ``seq_len`` pre-filter and computes, per cluster,
    each member's pLDDT-descending rank (``round``, ``entry_id`` tie break)
    and the cluster's usable-member ``cluster_size``.
    """
    cols = ", ".join(MANIFEST_COLUMNS)
    src = _sql_literal(input_glob)
    con.execute(
        f"""
        CREATE TEMP TABLE {table} AS
        WITH src AS (
            SELECT {cols}
            FROM read_parquet({src})
            WHERE seq_len BETWEEN {int(min_seq_len)} AND {int(max_seq_len)}
              AND struct_cluster_id IS NOT NULL
        )
        SELECT *,
            CAST(row_number() OVER (
                PARTITION BY struct_cluster_id
                ORDER BY global_plddt DESC, entry_id ASC
            ) - 1 AS INTEGER) AS round,
            count(*) OVER (PARTITION BY struct_cluster_id) AS cluster_size
        FROM src
        """
    )


def build_selection_table(
    con: duckdb.DuckDBPyConnection,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    source: str = "filtered",
    table: str = "selection",
) -> None:
    """Derive the kept (entry, round) rows from the ``filtered`` table.

    Keeps clusters with ``>= min_cluster_size`` usable members and the top
    ``num_rounds`` members per cluster. Local-only (no input rescan).
    """
    cols = ", ".join(MANIFEST_COLUMNS)
    con.execute(
        f"""
        CREATE TEMP TABLE {table} AS
        SELECT {cols}, round
        FROM {source}
        WHERE cluster_size >= {int(min_cluster_size)}
          AND round < {int(num_rounds)}
        """
    )


def compute_stats(
    con: duckdb.DuckDBPyConnection,
    *,
    min_cluster_size: int,
    source: str = "filtered",
    table: str = "selection",
) -> dict[str, Any]:
    """Counts for the README / sizing Stage B — all from the local tables."""
    clusters_post, clusters_kept, rows_post = con.execute(
        f"""
        SELECT
            count(DISTINCT struct_cluster_id) AS clusters_post_seqlen,
            count(DISTINCT struct_cluster_id)
                FILTER (WHERE cluster_size >= {int(min_cluster_size)}) AS clusters_kept,
            count(*) AS rows_post_seqlen
        FROM {source}
        """
    ).fetchone()

    per = con.execute(
        f"SELECT split, round, count(*) AS n FROM {table} GROUP BY split, round ORDER BY split, round"
    ).fetchall()
    total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    per_split = con.execute(
        f"SELECT split, count(*) AS n, count(DISTINCT struct_cluster_id) AS clusters "
        f"FROM {table} GROUP BY split ORDER BY split"
    ).fetchall()
    return {
        "clusters_post_seqlen": int(clusters_post),
        "clusters_kept": int(clusters_kept or 0),
        "clusters_dropped": int(clusters_post) - int(clusters_kept or 0),
        "rows_post_seqlen": int(rows_post or 0),
        "selected_docs_total": int(total),
        "per_split": [
            {"split": s, "docs": int(n), "clusters": int(c)} for s, n, c in per_split
        ],
        "per_split_round": [
            {"split": s, "round": int(r), "docs": int(n)} for s, r, n in per
        ],
    }


def write_manifest(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path,
    *,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    shard_size: int = DEFAULT_SHARD_SIZE,
    table: str = "selection",
) -> list[dict[str, Any]]:
    """Write the selection manifest as single-round, round-descending shards.

    Layout: ``<out_dir>/<split>/shard_<NNNNN>.parquet``. Within each split,
    shards are filled round-4 first ... round-0 last, and no shard mixes two
    rounds — so a consumer reading shards in filename order sees rounds in
    descending order (the issue's "train highest-quality data last"). The
    ``round`` column is authoritative regardless of physical order.

    Returns one record per written shard (split, round, index, rows).
    """
    splits = [r[0] for r in con.execute(
        f"SELECT DISTINCT split FROM {table} ORDER BY split"
    ).fetchall()]
    written: list[dict[str, Any]] = []
    for split in splits:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        gidx = 0
        for rnd in range(num_rounds - 1, -1, -1):  # high -> low
            tbl = con.execute(
                f"SELECT * FROM {table} WHERE split = ? AND round = ? "
                f"ORDER BY struct_cluster_id",
                [split, rnd],
            ).to_arrow_table()
            for off in range(0, tbl.num_rows, shard_size):
                chunk = tbl.slice(off, shard_size)
                path = split_dir / f"shard_{gidx:05d}.parquet"
                pq.write_table(chunk, path)
                written.append(
                    {"split": split, "round": rnd, "shard": gidx, "rows": chunk.num_rows}
                )
                gidx += 1
    return written


def run(
    input_path: str,
    out_dir: Path,
    *,
    min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    """End-to-end Stage A: select, write the manifest, return + save stats."""
    input_glob = _as_parquet_glob(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = _connect(input_glob)
    build_filtered_table(
        con, input_glob, min_seq_len=min_seq_len, max_seq_len=max_seq_len,
    )
    build_selection_table(
        con, min_cluster_size=min_cluster_size, num_rounds=num_rounds,
    )
    shards = write_manifest(
        con, out_dir, num_rounds=num_rounds, shard_size=shard_size
    )
    stats = compute_stats(con, min_cluster_size=min_cluster_size)
    stats["config"] = {
        "input": input_glob,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "min_cluster_size": min_cluster_size,
        "num_rounds": num_rounds,
        "shard_size": shard_size,
    }
    stats["num_shards"] = len(shards)
    (out_dir / "_selection_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _print_stats(stats: dict[str, Any]) -> None:
    c = stats
    print(
        f"[select] clusters: kept={c['clusters_kept']:,} "
        f"dropped(<{c['config']['min_cluster_size']})={c['clusters_dropped']:,} "
        f"of {c['clusters_post_seqlen']:,} post-seqlen",
        file=sys.stderr,
    )
    print(f"[select] selected docs: {c['selected_docs_total']:,} in {c['num_shards']:,} shards",
          file=sys.stderr)
    for row in c["per_split_round"]:
        print(f"[select]   {row['split']:>5}  round {row['round']}: {row['docs']:,}",
              file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python selection.py",
        description="Stage A: afdb-24M -> contacts-v1 round selection manifest.",
    )
    p.add_argument("--input", required=True,
                   help="afdb-24M parquet glob/dir (local) or hf://datasets/"
                        "timodonnell/afdb-24M/**/*.parquet.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir for the manifest (<out>/<split>/shard_*.parquet).")
    p.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                   help="Drop structural clusters with fewer usable members "
                        f"(default {DEFAULT_MIN_CLUSTER_SIZE}).")
    p.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS,
                   help=f"Max examples kept per cluster (default {DEFAULT_NUM_ROUNDS}).")
    p.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                   help=f"Rows per manifest shard (default {DEFAULT_SHARD_SIZE}).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = run(
        args.input, args.out,
        min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len,
        min_cluster_size=args.min_cluster_size, num_rounds=args.num_rounds,
        shard_size=args.shard_size,
    )
    _print_stats(stats)
    print(f"[select] wrote manifest + _selection_stats.json to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
