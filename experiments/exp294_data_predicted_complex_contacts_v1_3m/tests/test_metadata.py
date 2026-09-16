# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from metadata import (
    HETERODIMER_COLUMNS,
    HOMODIMER_COLUMNS,
    normalize_all,
    run_census,
)


def _write_csv(path: Path, columns: dict[str, str], rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_normalize_and_census_preserve_source_and_derive_scores(tmp_path: Path) -> None:
    homo_csv = tmp_path / "homo.csv"
    hetero_csv = tmp_path / "hetero.csv"
    _write_csv(
        homo_csv,
        HOMODIMER_COLUMNS,
        [
            {
                "modelEntityId": "AF-HOMO",
                "uniprotAccession": "P1",
                "gene": "g1",
                "taxId": 1,
                "organismScientificName": "Species one",
                "ipTM": 0.8,
                "ipSAE_AB": 0.72,
                "ipSAE_BA": 0.68,
                "pDockQ": 0.5,
                "pDockQ2_AB": 0.24,
                "pDockQ2_BA": 0.30,
                "LIS_AB": 0.7,
                "LIS_BA": 0.7,
                "n0chn": 512,
                "numberOfInteractions": 40,
                "N_clash_backbone": 0,
                "N_clash_heavyAtom": 2,
                "local_tar_name": "chunk_0001",
            }
        ],
    )
    _write_csv(
        hetero_csv,
        HETERODIMER_COLUMNS,
        [
            {
                "modelEntityId": "AF-HETERO",
                "passes_quality_threshold": "true",
                "max_ipSAE": 0.65,
                "max_pDockQ2_AB": 0.31,
                "quality_ipsae_threshold": 0.6,
                "quality_pdockq2_threshold": 0.23,
                "uniprot_ac_1": "P3",
                "uniprot_ac_2": "P2",
                "tax_id_1": 2,
                "tax_id_2": 2,
                "gene_name_1": "g3",
                "gene_name_2": "g2",
                "ipTM": 0.75,
                "ipSAE_AB": 0.62,
                "ipSAE_BA": 0.55,
                "pDockQ": 0.4,
                "pDockQ2_AB": 0.31,
                "pDockQ2_BA": 0.25,
                "LIS_AB": 0.6,
                "LIS_BA": 0.5,
                "n0chn": 600,
                "numberOfInteractions": 25,
                "N_clash_backbone": 1,
                "N_clash_heavyAtom": 4,
                "local_tar_name": "shard_0_batch_0.tar",
            }
        ],
    )

    normalized = tmp_path / "normalized"
    summary = normalize_all(homo_csv, hetero_csv, normalized)
    assert [source["rows"] for source in summary["sources"]] == [1, 1]
    assert (normalized / "raw_homodimer.parquet").exists()

    rows = duckdb.sql(
        f"SELECT * FROM read_parquet('{normalized}/normalized_*.parquet', union_by_name=true) "
        "ORDER BY model_id"
    ).fetchall()
    columns = [
        item[0]
        for item in duckdb.sql(
            f"DESCRIBE SELECT * FROM read_parquet('{normalized}/normalized_*.parquet', union_by_name=true)"
        ).fetchall()
    ]
    by_id = {
        dict(zip(columns, row))["model_id"]: dict(zip(columns, row)) for row in rows
    }
    assert by_id["AF-HOMO"]["ipsae_score"] == 0.68
    assert by_id["AF-HOMO"]["source_quality_pass"] is True
    assert by_id["AF-HOMO"]["source_tar_uri"].endswith("/homodimers/chunk_0001.tar")
    assert by_id["AF-HETERO"]["accession_pair_id"] == "P2|P3"
    assert by_id["AF-HETERO"]["source_tar_uri"].endswith(
        "/heterodimers/shard_0_batch_0.tar"
    )

    census = run_census(str(normalized / "normalized_*.parquet"), tmp_path / "census")
    assert sum(row["rows"] for row in census["by_complex_type"]) == 2
    assert all(row["hard_eligible"] == 1 for row in census["by_complex_type"])
    assert (tmp_path / "census" / "yield_by_score.csv").exists()
