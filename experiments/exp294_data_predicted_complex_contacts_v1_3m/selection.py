# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a deterministic, decontaminated AFCDB complex training manifest.

The production selector requires one annotation row per UniProt accession::

    accession, sequence_sha256, sequence_length, eval_decontam_reason

``eval_decontam_reason`` is null for an eligible sequence and names the exp225
sequence-decontamination rule for an excluded sequence. Requiring this join
prevents accession-pair deduplication from being mislabeled as sequence-pair
deduplication and ensures every subunit is checked against the evaluation set.

Tier A contains every hard-eligible upstream-quality-pass model. Tier B first
fills the requested heterodimer floor and then the total-document quota, ranked
by distance above a configurable relaxed quality floor. Threshold selection is
explicitly pre-calibration; the 50k structure pilot must set the production
floor before the full run.

The defaults below are the 2026-09-16 metadata census's measured frontier, not
aspirations. Where the issue's stated target and the source's actual yield
disagree, the default follows the yield and says so.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_TARGET_DOCS = 3_000_000

#: The issue asked for 500,000 heterodimers "if the calibrated source yield
#: supports it". The 2026-09-16 census says it does not: only 922,359
#: hard-eligible heterodimers have a nonzero ipSAE at all and 353,774 reach
#: ipSAE >= 0.1, so a 500k floor would be filled with models that have
#: essentially no predicted interface. This is the measured yield at the 0.30
#: quality floor instead. Raising it back is a Stage-E question (PINDER,
#: Predictomes, PPIRef), not an AFCDB threshold question.
DEFAULT_MIN_HETERODIMERS = 200_000

#: Also set by the census: a 0.5 floor yields 2,946,613 candidates *before*
#: sequence-pair dedup and eval decontamination, so it cannot reach 3M. 0.30
#: yields 3,455,113, which leaves margin for the decontamination loss (#225
#: removed 1.8-4.0% on comparable corpora). Still pre-calibration: the 50k
#: structure pilot sets the production value.
DEFAULT_MIN_RELAXED_QUALITY_RATIO = 0.3
DEFAULT_MAX_TOTAL_RESIDUES = 1998
DEFAULT_MAX_BACKBONE_CLASHES = 10


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _prepare_tables(
    con: duckdb.DuckDBPyConnection,
    input_glob: str,
    annotations_path: Path,
    *,
    max_total_residues: int,
    max_backbone_clashes: int,
) -> None:
    """Join sequence annotations and assign hard-filter/dedup status."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE normalized AS
        SELECT * FROM read_parquet({_sql_literal(input_glob)}, union_by_name=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE annotations AS
        SELECT
            CAST(accession AS VARCHAR) AS accession,
            CAST(sequence_sha256 AS VARCHAR) AS sequence_sha256,
            CAST(sequence_length AS INTEGER) AS sequence_length,
            CAST(eval_decontam_reason AS VARCHAR) AS eval_decontam_reason
        FROM read_parquet({_sql_literal(annotations_path)})
        """
    )
    duplicate_annotations = con.execute(
        "SELECT count(*) FROM (SELECT accession FROM annotations GROUP BY accession HAVING count(*) != 1)"
    ).fetchone()[0]
    if duplicate_annotations:
        raise ValueError(
            f"annotation table has {duplicate_annotations} non-unique accessions"
        )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE joined AS
        WITH annotated AS (
            SELECT
                n.*,
                a.sequence_sha256 AS sequence_sha256_a,
                b.sequence_sha256 AS sequence_sha256_b,
                a.sequence_length AS sequence_length_a,
                b.sequence_length AS sequence_length_b,
                a.eval_decontam_reason AS eval_decontam_reason_a,
                b.eval_decontam_reason AS eval_decontam_reason_b,
                a.sequence_length + b.sequence_length AS total_residues,
                least(a.sequence_sha256, b.sequence_sha256) || '|' ||
                    greatest(a.sequence_sha256, b.sequence_sha256) AS sequence_pair_id
            FROM normalized n
            LEFT JOIN annotations a ON n.accession_a = a.accession
            LEFT JOIN annotations b ON n.accession_b = b.accession
        ), classified AS (
            SELECT *,
                CASE
                    WHEN accession_a IS NULL OR accession_b IS NULL
                        THEN 'missing_accession'
                    WHEN sequence_sha256_a IS NULL OR sequence_sha256_b IS NULL
                        THEN 'missing_sequence_annotation'
                    WHEN eval_decontam_reason_a IS NOT NULL OR eval_decontam_reason_b IS NOT NULL
                        THEN 'eval_sequence_homolog'
                    WHEN total_residues < 4 OR total_residues > {int(max_total_residues)}
                        THEN 'does_not_fit_contacts_ring'
                    WHEN coalesce(num_interactions, 0) < 1
                        THEN 'no_reported_interaction'
                    WHEN coalesce(clashes_backbone, 1000000) > {int(max_backbone_clashes)}
                        THEN 'too_many_backbone_clashes'
                    WHEN quality_ratio IS NULL
                        THEN 'missing_quality_score'
                    ELSE NULL
                END AS hard_rejection
            FROM annotated
        )
        SELECT *,
            CASE WHEN hard_rejection IS NULL THEN row_number() OVER (
                PARTITION BY sequence_pair_id
                ORDER BY (hard_rejection IS NULL) DESC,
                         source_quality_pass DESC, quality_ratio DESC,
                         ipsae_score DESC, pdockq2_score DESC, iptm DESC,
                         num_interactions DESC, clashes_backbone ASC, model_id ASC
            ) END AS sequence_pair_rank
        FROM classified
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE unique_candidates AS
        SELECT * FROM joined
        WHERE hard_rejection IS NULL AND sequence_pair_rank = 1
        """
    )


def _selection_order() -> str:
    return (
        "quality_ratio DESC, ipsae_score DESC, pdockq2_score DESC, "
        "iptm DESC, num_interactions DESC, clashes_backbone ASC, model_id ASC"
    )


def _choose_selection(
    con: duckdb.DuckDBPyConnection,
    *,
    target_docs: int,
    min_heterodimers: int,
    min_relaxed_quality_ratio: float,
) -> dict[str, int]:
    """Build ``selected`` with all Tier A plus quota-filling Tier B rows."""
    con.execute(
        """
        CREATE OR REPLACE TABLE tier_a AS
        SELECT *, 'A'::VARCHAR AS confidence_tier, 'source_quality_pass'::VARCHAR AS selection_reason
        FROM unique_candidates
        WHERE source_quality_pass
        """
    )
    tier_a_docs, tier_a_heterodimers = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE complex_type = 'heterodimer') FROM tier_a"
    ).fetchone()
    heterodimer_needed = max(0, min_heterodimers - int(tier_a_heterodimers))

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tier_b_heterodimer AS
        SELECT *, 'B'::VARCHAR AS confidence_tier,
               'heterodimer_floor'::VARCHAR AS selection_reason
        FROM unique_candidates
        WHERE NOT source_quality_pass
          AND complex_type = 'heterodimer'
          AND quality_ratio >= {float(min_relaxed_quality_ratio)}
        ORDER BY {_selection_order()}
        LIMIT {int(heterodimer_needed)}
        """
    )
    tier_b_heterodimers = int(
        con.execute("SELECT count(*) FROM tier_b_heterodimer").fetchone()[0]
    )
    remaining_needed = max(0, target_docs - int(tier_a_docs) - tier_b_heterodimers)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE tier_b_remainder AS
        SELECT u.*, 'B'::VARCHAR AS confidence_tier,
               'total_document_quota'::VARCHAR AS selection_reason
        FROM unique_candidates u
        LEFT JOIN tier_b_heterodimer h USING (model_id)
        WHERE NOT u.source_quality_pass
          AND u.quality_ratio >= {float(min_relaxed_quality_ratio)}
          AND h.model_id IS NULL
        ORDER BY {_selection_order()}
        LIMIT {int(remaining_needed)}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE selected AS
        SELECT * FROM tier_a
        UNION ALL BY NAME SELECT * FROM tier_b_heterodimer
        UNION ALL BY NAME SELECT * FROM tier_b_remainder
        """
    )
    selected_docs, selected_heterodimers = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE complex_type = 'heterodimer') FROM selected"
    ).fetchone()
    return {
        "tier_a_docs": int(tier_a_docs),
        "tier_a_heterodimers": int(tier_a_heterodimers),
        "tier_b_heterodimer_floor_docs": tier_b_heterodimers,
        "tier_b_quota_docs": int(
            con.execute("SELECT count(*) FROM tier_b_remainder").fetchone()[0]
        ),
        "selected_docs": int(selected_docs),
        "selected_heterodimers": int(selected_heterodimers),
    }


def _write_outputs(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "selected.parquet"
    ledger_path = out_dir / "selection_ledger.parquet"
    con.execute(
        f"""
        COPY (
            SELECT
                model_id, complex_type, accession_a, accession_b,
                sequence_sha256_a, sequence_sha256_b,
                sequence_length_a, sequence_length_b, total_residues,
                sequence_pair_id, source_quality_pass, confidence_tier,
                selection_reason, quality_ratio, ipsae_score, pdockq2_score,
                iptm, pdockq, lis_ab, lis_ba, num_interactions,
                clashes_backbone, clashes_heavy_atom, tax_id_a, tax_id_b,
                gene_a, gene_b, local_tar_name, source_tar_uri
            FROM selected
            ORDER BY source_tar_uri, model_id
        ) TO {_sql_literal(manifest_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                j.model_id,
                CASE
                    WHEN j.hard_rejection IS NOT NULL THEN 'rejected'
                    WHEN j.sequence_pair_rank > 1 THEN 'rejected'
                    WHEN s.model_id IS NOT NULL THEN 'selected'
                    ELSE 'not_selected'
                END AS status,
                CASE
                    WHEN j.hard_rejection IS NOT NULL THEN j.hard_rejection
                    WHEN j.sequence_pair_rank > 1 THEN 'duplicate_sequence_pair'
                    WHEN s.model_id IS NOT NULL THEN 'tier_' || lower(s.confidence_tier)
                    WHEN j.quality_ratio < {float(policy["min_relaxed_quality_ratio"])}
                        THEN 'below_relaxed_quality_floor'
                    ELSE 'quota_filled_by_higher_ranked_models'
                END AS reason
            FROM joined j
            LEFT JOIN selected s USING (model_id)
            ORDER BY j.model_id
        ) TO {_sql_literal(ledger_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    status_rows = con.execute(
        f"""
        SELECT status, reason, count(*)
        FROM read_parquet({_sql_literal(ledger_path)})
        GROUP BY status, reason
        ORDER BY status, reason
        """
    ).fetchall()
    concentration = con.execute(
        """
        SELECT
            count(*) AS selected_docs,
            count(DISTINCT sequence_pair_id) AS unique_sequence_pairs,
            max(pair_docs) AS max_documents_per_sequence_pair
        FROM (
            SELECT *, count(*) OVER (PARTITION BY sequence_pair_id) AS pair_docs
            FROM selected
        )
        """
    ).fetchone()
    stats = {
        "policy": policy,
        "target_met": int(concentration[0]) >= int(policy["target_docs"]),
        "heterodimer_floor_met": (
            int(policy["selected_heterodimers"]) >= int(policy["min_heterodimers"])
        ),
        "selected_docs": int(concentration[0]),
        "unique_sequence_pairs": int(concentration[1]),
        "max_documents_per_sequence_pair": int(concentration[2] or 0),
        "ledger": [
            {"status": status, "reason": reason, "rows": int(rows)}
            for status, reason, rows in status_rows
        ],
    }
    (out_dir / "selection.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def run_selection(
    input_glob: str,
    annotations_path: Path,
    out_dir: Path,
    *,
    target_docs: int = DEFAULT_TARGET_DOCS,
    min_heterodimers: int = DEFAULT_MIN_HETERODIMERS,
    min_relaxed_quality_ratio: float = DEFAULT_MIN_RELAXED_QUALITY_RATIO,
    max_total_residues: int = DEFAULT_MAX_TOTAL_RESIDUES,
    max_backbone_clashes: int = DEFAULT_MAX_BACKBONE_CLASHES,
    database: Path | None = None,
) -> dict[str, Any]:
    """Create selection manifest, complete ledger, policy, and statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(database or out_dir / "selection.duckdb")
    con = duckdb.connect(db_path)
    _prepare_tables(
        con,
        input_glob,
        annotations_path,
        max_total_residues=max_total_residues,
        max_backbone_clashes=max_backbone_clashes,
    )
    selection_counts = _choose_selection(
        con,
        target_docs=target_docs,
        min_heterodimers=min_heterodimers,
        min_relaxed_quality_ratio=min_relaxed_quality_ratio,
    )
    policy: dict[str, Any] = {
        "version": "pilot-v1-census",
        "target_docs": target_docs,
        "min_heterodimers": min_heterodimers,
        "min_relaxed_quality_ratio": min_relaxed_quality_ratio,
        "max_total_residues": max_total_residues,
        "min_reported_interactions": 1,
        "max_backbone_clashes": max_backbone_clashes,
        "sequence_annotation_path": str(annotations_path.resolve()),
        "ranking": _selection_order(),
        **selection_counts,
    }
    stats = _write_outputs(con, out_dir, policy)
    if not stats["target_met"]:
        raise RuntimeError(
            f"quality floor yielded {stats['selected_docs']:,} documents, below "
            f"the target of {target_docs:,}; inspect outputs before lowering the floor"
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Normalized Parquet file or glob."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-docs", type=int, default=DEFAULT_TARGET_DOCS)
    parser.add_argument(
        "--min-heterodimers", type=int, default=DEFAULT_MIN_HETERODIMERS
    )
    parser.add_argument(
        "--min-relaxed-quality-ratio",
        type=float,
        default=DEFAULT_MIN_RELAXED_QUALITY_RATIO,
    )
    parser.add_argument(
        "--max-total-residues", type=int, default=DEFAULT_MAX_TOTAL_RESIDUES
    )
    parser.add_argument(
        "--max-backbone-clashes", type=int, default=DEFAULT_MAX_BACKBONE_CLASHES
    )
    parser.add_argument("--database", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = run_selection(
        args.input,
        args.annotations,
        args.out,
        target_docs=args.target_docs,
        min_heterodimers=args.min_heterodimers,
        min_relaxed_quality_ratio=args.min_relaxed_quality_ratio,
        max_total_residues=args.max_total_residues,
        max_backbone_clashes=args.max_backbone_clashes,
        database=args.database,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
