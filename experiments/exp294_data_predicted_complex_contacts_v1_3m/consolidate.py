# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage F — consolidate both arms into the publishable corpus.

Extraction writes one small parquet per input unit (per tar for AFCDB, per
batch for PINDER) because that is what makes preemption survivable. That shape
is wrong for a corpus: ~17,000 files of a few hundred KB each are slow to read
and awkward to publish. This reads them, reconciles the accounting, applies the
PINDER eval drop list, and writes evenly-sized shards.

It also emits the two manifests the issue asks for, kept separate:

* **natural** — every document, in the mixture the sources actually have.
* **balanced** — an inverse-square-root interface-cluster weight with a
  per-cluster cap, so no small set of clusters dominates sampling.

The weights are columns on a manifest, not a filter: the corpus keeps all its
redundancy and training decides what to do with it.

    uv run python consolidate.py --afcdb gs://.../corpus \\
        --pinder gs://.../pinder/corpus --pinder-droplist ... --out gs://.../release
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

#: Rows per output shard. These documents run ~6 KB compressed, so 20k keeps a
#: shard near 120 MB -- comfortable for both HF and a streaming reader, and in
#: line with the other published corpora.
DEFAULT_SHARD_ROWS = 20_000
#: PINDER publishes its own train/val/test split. Its val and test systems are a
#: community benchmark, so shipping them inside a *training* corpus would
#: contaminate anyone who evaluates on PINDER. They are excluded here even
#: though they survive the eval2 drop list, which only knows about our own eval.
PINDER_EXCLUDED_SPLITS = ("val", "test")
#: No single interface cluster may own more than this share of balanced
#: sampling probability. The issue asks for 0.1%.
DEFAULT_CLUSTER_CAP = 0.001


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _connect(threads: int, temp_dir: str | None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads TO {int(threads)}")
    con.execute("SET preserve_insertion_order = false")
    if temp_dir:
        # The intermediate tables hold the full document text several times
        # over; without this duckdb spills onto whatever partition it started
        # in, which is not necessarily the one with room.
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = {_sql_literal(temp_dir)}")
    return con


def build(
    afcdb: str,
    pinder: str,
    out_dir: str,
    *,
    pinder_droplist: str | None = None,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    cluster_cap: float = DEFAULT_CLUSTER_CAP,
    threads: int = 16,
    temp_dir: str | None = None,
) -> dict[str, Any]:
    """Merge, reconcile, shard, and write both sampling manifests."""
    con = _connect(threads, temp_dir)
    out = str(out_dir).rstrip("/")
    if "://" in out:
        raise ValueError(
            f"--out must be a local directory (got {out!r}); the shard flatten "
            "step renames files, and publishing is a separate upload")

    # --- AFCDB arm -------------------------------------------------------
    con.execute(
        f"""
        CREATE OR REPLACE VIEW afcdb_docs AS
        SELECT source_model_key AS document_id, 'afcdb' AS source_arm,
               complex_type, document, sha1, seq_len, num_tokens, num_chains,
               chain_ids, chain_lengths, contacts_emitted, contacts_emitted_inter_chain,
               truncated, confidence_tier,
               accession_a AS partner_a, accession_b AS partner_b,
               quality_ratio, ipsae_score, pdockq2_score,
               NULL::VARCHAR AS pdb_id, NULL::DOUBLE AS resolution,
               NULL::VARCHAR AS interface_cluster_id,
               NULL::BOOLEAN AS pair_also_in_afcdb
        FROM read_parquet({_sql_literal(afcdb + "/documents/*.parquet")})
        """
    )
    # --- PINDER arm, minus eval homologs and PINDER's own benchmark ------
    splits = ", ".join(_sql_literal(s) for s in PINDER_EXCLUDED_SPLITS)
    predicates = [f"p.split NOT IN ({splits})"]
    drop_join = ""
    if pinder_droplist:
        con.execute(
            f"CREATE OR REPLACE VIEW pinder_drop AS "
            f"SELECT system_id FROM read_parquet({_sql_literal(pinder_droplist)})"
        )
        drop_join = "LEFT JOIN pinder_drop d USING (system_id)"
        predicates.append("d.system_id IS NULL")
    drop_where = "WHERE " + " AND ".join(predicates)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW pinder_docs AS
        SELECT p.system_id AS document_id, 'pinder' AS source_arm,
               'heterodimer' AS complex_type, p.document, p.sha1, p.seq_len,
               p.num_tokens, p.num_chains, p.chain_ids, p.chain_lengths, p.contacts_emitted,
               p.contacts_emitted_inter_chain, p.truncated,
               'experimental'::VARCHAR AS confidence_tier,
               p.uniprot_R AS partner_a, p.uniprot_L AS partner_b,
               NULL::DOUBLE AS quality_ratio, NULL::DOUBLE AS ipsae_score,
               NULL::DOUBLE AS pdockq2_score,
               p.pdb_id, p.resolution, p.cluster_id AS interface_cluster_id,
               p.pair_also_in_afcdb
        FROM read_parquet({_sql_literal(pinder + "/documents/*.parquet")}) p
        {drop_join} {drop_where}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW pinder_all AS
        SELECT system_id, split FROM read_parquet(
            {_sql_literal(pinder + "/documents/*.parquet")})
        """
    )
    intake = {
        "afcdb_documents_in": con.execute(
            "SELECT count(*) FROM afcdb_docs").fetchone()[0],
        "pinder_documents_in": con.execute(
            "SELECT count(*) FROM pinder_all").fetchone()[0],
        "pinder_dropped_benchmark_split": con.execute(
            f"SELECT count(*) FROM pinder_all WHERE split IN ({splits})"
        ).fetchone()[0],
        "pinder_dropped_eval_homolog": con.execute(
            f"SELECT count(*) FROM pinder_all WHERE split NOT IN ({splits}) AND "
            "system_id IN (SELECT system_id FROM pinder_drop)"
        ).fetchone()[0] if pinder_droplist else 0,
        "pinder_documents_kept": con.execute(
            "SELECT count(*) FROM pinder_docs").fetchone()[0],
    }
    expected = (intake["pinder_documents_in"]
                - intake["pinder_dropped_benchmark_split"]
                - intake["pinder_dropped_eval_homolog"])
    if intake["pinder_documents_kept"] != expected:
        raise RuntimeError(
            f"PINDER accounting does not close: kept "
            f"{intake['pinder_documents_kept']:,} but "
            f"{intake['pinder_documents_in']:,} in minus drops is {expected:,}")
    con.execute(
        """
        CREATE OR REPLACE TABLE corpus AS
        SELECT * FROM afcdb_docs UNION ALL BY NAME SELECT * FROM pinder_docs
        """
    )

    # Cluster key: PINDER carries a real interface cluster; AFCDB has none, so
    # its sequence pair stands in. Labelled so the two are never conflated.
    con.execute(
        """
        CREATE OR REPLACE TABLE weighted AS
        WITH keyed AS (
            SELECT *, coalesce(
                'pinder:' || interface_cluster_id,
                'afcdb_pair:' || least(partner_a, partner_b) || '|' ||
                    greatest(partner_a, partner_b)) AS cluster_key
            FROM corpus
        ), sized AS (
            SELECT *, count(*) OVER (PARTITION BY cluster_key) AS cluster_size
            FROM keyed
        )
        SELECT *, 1.0 / sqrt(cluster_size) AS raw_weight FROM sized
        """
    )
    total_raw = con.execute("SELECT sum(raw_weight) FROM weighted").fetchone()[0]
    con.execute(
        f"""
        CREATE OR REPLACE TABLE balanced AS
        WITH per_cluster AS (
            SELECT cluster_key, sum(raw_weight) / {float(total_raw)} AS cluster_share
            FROM weighted GROUP BY cluster_key
        )
        SELECT w.*,
            w.raw_weight * least(1.0,
                {float(cluster_cap)} / nullif(c.cluster_share, 0)) AS sampling_weight
        FROM weighted w JOIN per_cluster c USING (cluster_key)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE sharded AS
        SELECT *, CAST(floor((row_number() OVER (ORDER BY hash(document_id),
                    document_id) - 1) / {int(shard_rows)}) AS INTEGER) AS shard
        FROM balanced
        """
    )
    n_shards = int(con.execute("SELECT max(shard) + 1 FROM sharded").fetchone()[0])
    stage = Path(out) / "_staged"
    stage.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (SELECT * EXCLUDE (raw_weight) FROM sharded)
        TO {_sql_literal(stage)}
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (shard),
         OVERWRITE_OR_IGNORE, FILENAME_PATTERN 'part_{{i}}')
        """
    )
    # PARTITION_BY is the only way to get duckdb to write many evenly sized
    # files, and it insists on `shard=N/` directories. The published corpora all
    # use one flat file per shard, so rename into that.
    train = Path(out) / "train"
    train.mkdir(parents=True, exist_ok=True)
    for index in range(n_shards):
        parts = sorted((stage / f"shard={index}").glob("*.parquet"))
        if len(parts) != 1:
            raise RuntimeError(
                f"shard {index}: expected one part file, found {len(parts)}")
        parts[0].rename(train / f"shard-{index:05d}-of-{n_shards:05d}.parquet")
    shutil.rmtree(stage)
    for name, order in (
        ("natural", "hash(document_id), document_id"),
        ("balanced", "sampling_weight DESC, document_id"),
    ):
        con.execute(
            f"""
            COPY (
                SELECT document_id, source_arm, complex_type, confidence_tier,
                       cluster_key, cluster_size, sampling_weight, num_tokens,
                       seq_len, contacts_emitted_inter_chain
                FROM sharded ORDER BY {order}
            ) TO {_sql_literal(out + f"/manifest_{name}.parquet")}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )

    by_arm = con.execute(
        """
        SELECT source_arm, complex_type, count(*), sum(num_tokens),
               count(DISTINCT cluster_key)
        FROM sharded GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()
    # Rank explicitly: an ORDER BY inside a CTE is not preserved when the CTE is
    # re-read under a LIMIT, which silently returns arbitrary clusters.
    top = con.execute(
        """
        WITH c AS (SELECT cluster_key, sum(sampling_weight) AS w FROM sharded
                   GROUP BY 1),
             ranked AS (SELECT w, row_number() OVER (ORDER BY w DESC) AS rank,
                               sum(w) OVER () AS total FROM c)
        SELECT sum(CASE WHEN rank <= 1 THEN w END) / max(total),
               sum(CASE WHEN rank <= 10 THEN w END) / max(total),
               sum(CASE WHEN rank <= 100 THEN w END) / max(total)
        FROM ranked
        """
    ).fetchone()
    totals = con.execute(
        "SELECT count(*), sum(num_tokens), count(DISTINCT cluster_key), "
        "count(DISTINCT sha1) FROM sharded"
    ).fetchone()
    stats = {
        "out_dir": out,
        **{k: int(v) for k, v in intake.items()},
        "pinder_excluded_splits": list(PINDER_EXCLUDED_SPLITS),
        "shards": n_shards,
        "shard_rows": shard_rows,
        "documents": int(totals[0]),
        "tokens": int(totals[1]),
        "clusters": int(totals[2]),
        "distinct_sha1": int(totals[3]),
        "cluster_cap": cluster_cap,
        "balanced_share_top_1_cluster": round(float(top[0]), 6),
        "balanced_share_top_10_clusters": round(float(top[1]), 6),
        "balanced_share_top_100_clusters": round(float(top[2]), 6),
        "by_arm": [
            {"source_arm": a, "complex_type": c, "documents": int(n),
             "tokens": int(t), "clusters": int(k)}
            for a, c, n, t, k in by_arm
        ],
    }
    con.execute(
        f"COPY (SELECT {_sql_literal(json.dumps(stats))} AS j) "
        f"TO {_sql_literal(out + '/corpus_stats.json')} (FORMAT CSV, HEADER false, QUOTE '')"
    )
    print(json.dumps(stats, indent=2))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afcdb", required=True)
    parser.add_argument("--pinder", required=True)
    parser.add_argument("--pinder-droplist", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--cluster-cap", type=float, default=DEFAULT_CLUSTER_CAP)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--temp-dir", default=None,
                        help="Where duckdb spills; needs room for a few copies of the corpus.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build(args.afcdb, args.pinder, args.out, pinder_droplist=args.pinder_droplist,
          shard_rows=args.shard_rows, cluster_cap=args.cluster_cap, threads=args.threads,
          temp_dir=args.temp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
