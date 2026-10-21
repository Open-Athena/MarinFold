# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize and census the NVIDIA/AFDB complex metadata release.

Stage A deliberately operates only on the two metadata CSVs. It writes both a
typed, lossless Parquet copy of each source table and a small common schema used
by selection. Coordinates are not downloaded or parsed here.

Download each CSV exactly once from ``AFCDB_BASE_URL`` into storage local to the
machine running this script, then run::

    uv run python metadata.py normalize \
        --homodimer-csv /data/afcdb/homodimer_metadata.csv \
        --heterodimer-csv /data/afcdb/heterodimer_metadata.csv \
        --out /data/exp294/metadata

    uv run python metadata.py census \
        --input '/data/exp294/metadata/normalized_*.parquet' \
        --out /data/exp294/census

The source CSVs are large enough that accepting HTTPS inputs here would cause
accidental repeated multi-gigabyte transfers during development. Local paths
are therefore required. The normalizer is fail-loud: malformed rows abort the
run instead of being discarded by an ``ignore_errors`` option.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

AFCDB_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda"
HOMODIMER_URL = f"{AFCDB_BASE_URL}/homodimer_metadata.csv"
HETERODIMER_URL = f"{AFCDB_BASE_URL}/heterodimer_metadata.csv"
SOURCE_README_URL = f"{AFCDB_BASE_URL}/README.txt"

STRING = "VARCHAR"
FLOAT = "DOUBLE"
INTEGER = "BIGINT"
BOOLEAN = "BOOLEAN"

HOMODIMER_COLUMNS: dict[str, str] = {
    "modelEntityId": STRING,
    "uniprotAccession": STRING,
    "gene": STRING,
    "taxId": INTEGER,
    "organismScientificName": STRING,
    "ipTM": FLOAT,
    "ipSAE_AB": FLOAT,
    "ipSAE_BA": FLOAT,
    "pDockQ": FLOAT,
    "pDockQ2_AB": FLOAT,
    "pDockQ2_BA": FLOAT,
    "LIS_AB": FLOAT,
    "LIS_BA": FLOAT,
    "ipTM_d0chn_AB": FLOAT,
    "ipTM_d0chn_BA": FLOAT,
    "n0chn": FLOAT,
    "n0res_AB": FLOAT,
    "n0dom_AB": FLOAT,
    "d0res_AB": FLOAT,
    "d0chn_AB": FLOAT,
    "d0dom_AB": FLOAT,
    "nres1_AB": FLOAT,
    "dist1_AB": FLOAT,
    "nres2_AB": FLOAT,
    "dist2_AB": FLOAT,
    "n0res_BA": FLOAT,
    "n0dom_BA": FLOAT,
    "d0res_BA": FLOAT,
    "d0chn_BA": FLOAT,
    "d0dom_BA": FLOAT,
    "nres1_BA": FLOAT,
    "dist1_BA": FLOAT,
    "nres2_BA": FLOAT,
    "dist2_BA": FLOAT,
    "ipSAE_PAE_cutoff": FLOAT,
    "ipSAE_dist_cutoff": FLOAT,
    "numberOfInteractions": FLOAT,
    "N_clash_backbone": FLOAT,
    "N_clash_heavyAtom": FLOAT,
    "local_tar_name": STRING,
}

HETERODIMER_COLUMNS: dict[str, str] = {
    "modelEntityId": STRING,
    "passes_quality_threshold": BOOLEAN,
    "max_ipSAE": FLOAT,
    "max_pDockQ2_AB": FLOAT,
    "quality_ipsae_threshold": FLOAT,
    "quality_pdockq2_threshold": FLOAT,
    "uniprot_ac_1": STRING,
    "uniprot_ac_2": STRING,
    "tax_id_1": INTEGER,
    "tax_id_2": INTEGER,
    "gene_name_1": STRING,
    "gene_name_2": STRING,
    "LIS_AB": FLOAT,
    "LIS_BA": FLOAT,
    "N_clash_backbone": FLOAT,
    "N_clash_heavyAtom": FLOAT,
    "numberOfInteractions": FLOAT,
    "d0chn": FLOAT,
    "d0dom_AB": FLOAT,
    "d0dom_BA": FLOAT,
    "d0res_AB": FLOAT,
    "d0res_BA": FLOAT,
    "max_dist_cutoff": FLOAT,
    "max_dist1_AB": FLOAT,
    "max_dist1_BA": FLOAT,
    "max_dist2_AB": FLOAT,
    "max_dist2_BA": FLOAT,
    "ipSAE_AB": FLOAT,
    "ipSAE_BA": FLOAT,
    "score_ipsae_d0chn_AB": FLOAT,
    "score_ipsae_d0chn_BA": FLOAT,
    "max_ipSAE_d0dom_AB": FLOAT,
    "max_ipSAE_d0dom_BA": FLOAT,
    "ipTM": FLOAT,
    "ipTM_d0chn_AB": FLOAT,
    "ipTM_d0chn_BA": FLOAT,
    "n0chn": FLOAT,
    "n0dom_AB": FLOAT,
    "n0dom_BA": FLOAT,
    "n0res_AB": FLOAT,
    "n0res_BA": FLOAT,
    "nres1_AB": FLOAT,
    "nres1_BA": FLOAT,
    "nres2_AB": FLOAT,
    "nres2_BA": FLOAT,
    "pDockQ": FLOAT,
    "pDockQ2_AB": FLOAT,
    "pDockQ2_BA": FLOAT,
    "ipSAE_PAE_cutoff": FLOAT,
    "local_tar_name": STRING,
}

QUALITY_IPSAE_THRESHOLD = 0.6
QUALITY_PDOCKQ2_THRESHOLD = 0.23
DEFAULT_MAX_TOTAL_RESIDUES = 1998
SCORE_RATIO_THRESHOLDS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


@dataclass(frozen=True)
class SourceSpec:
    """Schema and normalized projection for one upstream metadata table."""

    name: str
    columns: dict[str, str]
    source_url: str
    normalized_sql: str


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _columns_struct(columns: dict[str, str]) -> str:
    fields = ", ".join(
        f"{_sql_literal(name)}: {_sql_literal(dtype)}"
        for name, dtype in columns.items()
    )
    return "{" + fields + "}"


def _tar_expression(directory: str) -> str:
    return (
        f"'{AFCDB_BASE_URL}/{directory}/' || "
        "CASE WHEN ends_with(local_tar_name, '.tar') "
        "THEN local_tar_name ELSE local_tar_name || '.tar' END"
    )


HOMODIMER_NORMALIZED_SQL = f"""
SELECT
    modelEntityId AS model_id,
    'homodimer' AS complex_type,
    uniprotAccession AS accession_a,
    uniprotAccession AS accession_b,
    taxId AS tax_id_a,
    taxId AS tax_id_b,
    gene AS gene_a,
    gene AS gene_b,
    organismScientificName AS organism_name_a,
    organismScientificName AS organism_name_b,
    CASE WHEN ipSAE_AB IS NOT NULL AND ipSAE_BA IS NOT NULL
        THEN least(ipSAE_AB, ipSAE_BA) END AS ipsae_score,
    CASE WHEN pDockQ2_AB IS NOT NULL AND pDockQ2_BA IS NOT NULL
        THEN greatest(pDockQ2_AB, pDockQ2_BA) END AS pdockq2_score,
    coalesce(
        ipSAE_AB IS NOT NULL AND ipSAE_BA IS NOT NULL
        AND pDockQ2_AB IS NOT NULL AND pDockQ2_BA IS NOT NULL
        AND least(ipSAE_AB, ipSAE_BA) >= {QUALITY_IPSAE_THRESHOLD}
        AND greatest(pDockQ2_AB, pDockQ2_BA) >= {QUALITY_PDOCKQ2_THRESHOLD},
        false
    ) AS source_quality_pass,
    {QUALITY_IPSAE_THRESHOLD}::DOUBLE AS source_ipsae_threshold,
    {QUALITY_PDOCKQ2_THRESHOLD}::DOUBLE AS source_pdockq2_threshold,
    least(
        CASE WHEN ipSAE_AB IS NOT NULL AND ipSAE_BA IS NOT NULL
            THEN least(ipSAE_AB, ipSAE_BA) / {QUALITY_IPSAE_THRESHOLD} END,
        CASE WHEN pDockQ2_AB IS NOT NULL AND pDockQ2_BA IS NOT NULL
            THEN greatest(pDockQ2_AB, pDockQ2_BA) / {QUALITY_PDOCKQ2_THRESHOLD} END
    ) AS quality_ratio,
    ipTM AS iptm,
    ipSAE_AB AS ipsae_ab,
    ipSAE_BA AS ipsae_ba,
    pDockQ AS pdockq,
    pDockQ2_AB AS pdockq2_ab,
    pDockQ2_BA AS pdockq2_ba,
    LIS_AB AS lis_ab,
    LIS_BA AS lis_ba,
    CAST(round(n0chn) AS INTEGER) AS source_total_residues,
    CAST(round(numberOfInteractions) AS INTEGER) AS num_interactions,
    CAST(round(N_clash_backbone) AS INTEGER) AS clashes_backbone,
    CAST(round(N_clash_heavyAtom) AS INTEGER) AS clashes_heavy_atom,
    local_tar_name,
    {_tar_expression("homodimers")} AS source_tar_uri,
    uniprotAccession || '|' || uniprotAccession AS accession_pair_id
FROM read_parquet({{raw_path}})
"""

HETERODIMER_NORMALIZED_SQL = f"""
SELECT
    modelEntityId AS model_id,
    'heterodimer' AS complex_type,
    uniprot_ac_1 AS accession_a,
    uniprot_ac_2 AS accession_b,
    tax_id_1 AS tax_id_a,
    tax_id_2 AS tax_id_b,
    gene_name_1 AS gene_a,
    gene_name_2 AS gene_b,
    NULL::VARCHAR AS organism_name_a,
    NULL::VARCHAR AS organism_name_b,
    max_ipSAE AS ipsae_score,
    max_pDockQ2_AB AS pdockq2_score,
    coalesce(passes_quality_threshold, false) AS source_quality_pass,
    quality_ipsae_threshold AS source_ipsae_threshold,
    quality_pdockq2_threshold AS source_pdockq2_threshold,
    least(
        coalesce(max_ipSAE, 0.0) / nullif(quality_ipsae_threshold, 0.0),
        coalesce(max_pDockQ2_AB, 0.0) / nullif(quality_pdockq2_threshold, 0.0)
    ) AS quality_ratio,
    ipTM AS iptm,
    ipSAE_AB AS ipsae_ab,
    ipSAE_BA AS ipsae_ba,
    pDockQ AS pdockq,
    pDockQ2_AB AS pdockq2_ab,
    pDockQ2_BA AS pdockq2_ba,
    LIS_AB AS lis_ab,
    LIS_BA AS lis_ba,
    CAST(round(n0chn) AS INTEGER) AS source_total_residues,
    CAST(round(numberOfInteractions) AS INTEGER) AS num_interactions,
    CAST(round(N_clash_backbone) AS INTEGER) AS clashes_backbone,
    CAST(round(N_clash_heavyAtom) AS INTEGER) AS clashes_heavy_atom,
    local_tar_name,
    {_tar_expression("heterodimers")} AS source_tar_uri,
    least(uniprot_ac_1, uniprot_ac_2) || '|' || greatest(uniprot_ac_1, uniprot_ac_2)
        AS accession_pair_id
FROM read_parquet({{raw_path}})
"""

SOURCES = {
    "homodimer": SourceSpec(
        name="homodimer",
        columns=HOMODIMER_COLUMNS,
        source_url=HOMODIMER_URL,
        normalized_sql=HOMODIMER_NORMALIZED_SQL,
    ),
    "heterodimer": SourceSpec(
        name="heterodimer",
        columns=HETERODIMER_COLUMNS,
        source_url=HETERODIMER_URL,
        normalized_sql=HETERODIMER_NORMALIZED_SQL,
    ),
}


def _validate_local_csv(path: Path) -> None:
    if "://" in str(path):
        raise ValueError(
            "normalize requires a local CSV path; mirror the upstream metadata once "
            "instead of repeatedly streaming a multi-gigabyte URL"
        )
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_source(
    con: duckdb.DuckDBPyConnection,
    source: SourceSpec,
    csv_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Write lossless typed and normalized Parquet for one source CSV."""
    _validate_local_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_{source.name}.parquet"
    normalized_path = out_dir / f"normalized_{source.name}.parquet"

    read_csv = (
        f"read_csv({_sql_literal(csv_path)}, header=true, "
        f"columns={_columns_struct(source.columns)}, strict_mode=true, "
        "ignore_errors=false, parallel=true)"
    )
    con.execute(
        f"COPY (SELECT * FROM {read_csv}) TO {_sql_literal(raw_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    normalized = source.normalized_sql.format(raw_path=_sql_literal(raw_path))
    con.execute(
        f"COPY ({normalized}) TO {_sql_literal(normalized_path)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    rows, quality_pass = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE source_quality_pass) "
        f"FROM read_parquet({_sql_literal(normalized_path)})"
    ).fetchone()
    return {
        "source": source.name,
        "source_url": source.source_url,
        "csv_path": str(csv_path.resolve()),
        "csv_bytes": csv_path.stat().st_size,
        "raw_parquet": str(raw_path.resolve()),
        "normalized_parquet": str(normalized_path.resolve()),
        "rows": int(rows),
        "source_quality_pass": int(quality_pass),
    }


def normalize_all(
    homodimer_csv: Path, heterodimer_csv: Path, out_dir: Path
) -> dict[str, Any]:
    """Normalize both release tables and write source provenance."""
    con = duckdb.connect()
    summaries = [
        normalize_source(con, SOURCES["homodimer"], homodimer_csv, out_dir),
        normalize_source(con, SOURCES["heterodimer"], heterodimer_csv, out_dir),
    ]
    result = {
        "source_readme": SOURCE_README_URL,
        "quality_gate_note": (
            "The heterodimer pass flag is upstream-authored. The homodimer pass flag "
            "is the exp145-style proxy: min directional ipSAE >= 0.6 and max "
            "directional pDockQ2 >= 0.23. The AFCDB paper's pLDDT condition cannot "
            "be reproduced from the current metadata CSV alone."
        ),
        "sources": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalization.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_census(input_glob: str, out_dir: Path) -> dict[str, Any]:
    """Report confidence and hard-filter yield without reading coordinates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    src = f"read_parquet({_sql_literal(input_glob)}, union_by_name=true)"
    per_type = con.execute(
        f"""
        SELECT
            complex_type,
            count(*) AS rows,
            count(*) FILTER (WHERE source_quality_pass) AS source_quality_pass,
            count(*) FILTER (
                WHERE source_total_residues BETWEEN 4 AND {DEFAULT_MAX_TOTAL_RESIDUES}
            ) AS context_eligible,
            count(*) FILTER (WHERE coalesce(num_interactions, 0) >= 1) AS has_interaction,
            count(*) FILTER (WHERE coalesce(clashes_backbone, 1000000) <= 10) AS clash_eligible,
            count(*) FILTER (
                WHERE source_total_residues BETWEEN 4 AND {DEFAULT_MAX_TOTAL_RESIDUES}
                  AND coalesce(num_interactions, 0) >= 1
                  AND coalesce(clashes_backbone, 1000000) <= 10
            ) AS hard_eligible
        FROM {src}
        GROUP BY complex_type
        ORDER BY complex_type
        """
    ).fetchall()

    ratio_rows: list[dict[str, Any]] = []
    for ratio in SCORE_RATIO_THRESHOLDS:
        for complex_type, rows in con.execute(
            f"""
            SELECT complex_type, count(*)
            FROM {src}
            WHERE quality_ratio >= ?
              AND source_total_residues BETWEEN 4 AND {DEFAULT_MAX_TOTAL_RESIDUES}
              AND coalesce(num_interactions, 0) >= 1
              AND coalesce(clashes_backbone, 1000000) <= 10
            GROUP BY complex_type
            ORDER BY complex_type
            """,
            [ratio],
        ).fetchall():
            ratio_rows.append(
                {
                    "quality_ratio_min": ratio,
                    "complex_type": complex_type,
                    "rows": int(rows),
                }
            )

    values_sql = ",".join(
        f"({row['quality_ratio_min']}, {_sql_literal(row['complex_type'])}, {row['rows']})"
        for row in ratio_rows
    )
    con.execute(
        f"""
        COPY (
            SELECT * FROM (VALUES
                {values_sql}
            ) AS t(quality_ratio_min, complex_type, rows)
            ORDER BY quality_ratio_min DESC, complex_type
        ) TO {_sql_literal(out_dir / "yield_by_score.csv")} (HEADER, DELIMITER ',')
        """
    )
    result = {
        "hard_filter": {
            "min_total_residues": 4,
            "max_total_residues": DEFAULT_MAX_TOTAL_RESIDUES,
            "min_interactions": 1,
            "max_backbone_clashes": 10,
        },
        "by_complex_type": [
            {
                "complex_type": row[0],
                "rows": int(row[1]),
                "source_quality_pass": int(row[2]),
                "context_eligible": int(row[3]),
                "has_interaction": int(row[4]),
                "clash_eligible": int(row[5]),
                "hard_eligible": int(row[6]),
            }
            for row in per_type
        ],
        "yield_by_score": ratio_rows,
    }
    (out_dir / "census.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    normalize = sub.add_parser(
        "normalize", help="Convert the two source CSVs to Parquet."
    )
    normalize.add_argument("--homodimer-csv", type=Path, required=True)
    normalize.add_argument("--heterodimer-csv", type=Path, required=True)
    normalize.add_argument("--out", type=Path, required=True)

    census = sub.add_parser("census", help="Compute metadata-only yield curves.")
    census.add_argument(
        "--input", required=True, help="Normalized Parquet file or glob."
    )
    census.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "normalize":
        result = normalize_all(args.homodimer_csv, args.heterodimer_csv, args.out)
    else:
        result = run_census(args.input, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
