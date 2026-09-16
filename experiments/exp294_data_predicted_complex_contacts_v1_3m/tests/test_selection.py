# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from selection import run_selection


def _model(
    model_id: str,
    complex_type: str,
    accession_a: str,
    accession_b: str,
    *,
    source_quality_pass: bool = False,
    quality_ratio: float = 0.8,
    interactions: int = 10,
    clashes: int = 0,
) -> dict:
    return {
        "model_id": model_id,
        "complex_type": complex_type,
        "accession_a": accession_a,
        "accession_b": accession_b,
        "source_quality_pass": source_quality_pass,
        "quality_ratio": quality_ratio,
        "ipsae_score": quality_ratio * 0.6,
        "pdockq2_score": quality_ratio * 0.23,
        "iptm": quality_ratio,
        "pdockq": quality_ratio / 2,
        "lis_ab": quality_ratio,
        "lis_ba": quality_ratio,
        "source_total_residues": 200,
        "num_interactions": interactions,
        "clashes_backbone": clashes,
        "clashes_heavy_atom": clashes,
        "tax_id_a": 1,
        "tax_id_b": 1,
        "gene_a": accession_a,
        "gene_b": accession_b,
        "local_tar_name": "chunk.tar",
        "source_tar_uri": "https://example.test/chunk.tar",
    }


def _annotations(accessions: set[str]) -> list[dict]:
    rows = []
    for index, accession in enumerate(sorted(accessions)):
        rows.append(
            {
                "accession": accession,
                "sequence_sha256": f"hash-{index}",
                "sequence_length": 100,
                "eval_decontam_reason": "identity" if accession == "CONTAM" else None,
            }
        )
    for row in rows:
        if row["accession"] == "LONG":
            row["sequence_length"] = 1900
    return rows


def test_selection_is_tiered_decontaminated_deduplicated_and_audited(
    tmp_path: Path,
) -> None:
    models = [
        _model("A-HOMO", "homodimer", "H1", "H1", source_quality_pass=True),
        _model("A-HET", "heterodimer", "A1", "A2", source_quality_pass=True),
        _model("B-HET", "heterodimer", "B1", "B2", quality_ratio=0.9),
        _model("B-HOMO", "homodimer", "H2", "H2", quality_ratio=0.8),
        _model("LOW", "heterodimer", "L1", "L2", quality_ratio=0.2),
        _model("DUP-LOWER", "heterodimer", "A2", "A1", quality_ratio=0.7),
        _model("CONTAM", "heterodimer", "CONTAM", "C2", quality_ratio=0.95),
        _model("TOO-LONG", "heterodimer", "LONG", "L3", quality_ratio=0.95),
        _model("NO-INTERFACE", "heterodimer", "N1", "N2", interactions=0),
        _model("CLASH", "heterodimer", "C3", "C4", clashes=11),
    ]
    input_path = tmp_path / "normalized.parquet"
    annotations_path = tmp_path / "annotations.parquet"
    pq.write_table(pa.Table.from_pylist(models), input_path)
    accessions = {m[key] for m in models for key in ("accession_a", "accession_b")}
    pq.write_table(pa.Table.from_pylist(_annotations(accessions)), annotations_path)

    out = tmp_path / "selection"
    stats = run_selection(
        str(input_path),
        annotations_path,
        out,
        target_docs=4,
        min_heterodimers=2,
        min_relaxed_quality_ratio=0.5,
        database=tmp_path / "selection.duckdb",
    )

    selected = pq.read_table(out / "selected.parquet").to_pylist()
    assert {row["model_id"] for row in selected} == {
        "A-HOMO",
        "A-HET",
        "B-HET",
        "B-HOMO",
    }
    assert {
        row["confidence_tier"] for row in selected if row["model_id"].startswith("A-")
    } == {"A"}
    assert {
        row["confidence_tier"] for row in selected if row["model_id"].startswith("B-")
    } == {"B"}
    assert stats["target_met"] is True
    assert stats["selected_docs"] == 4
    assert stats["unique_sequence_pairs"] == 4
    assert stats["max_documents_per_sequence_pair"] == 1

    ledger = {
        row["model_id"]: row
        for row in pq.read_table(out / "selection_ledger.parquet").to_pylist()
    }
    assert ledger["DUP-LOWER"]["reason"] == "duplicate_sequence_pair"
    assert ledger["CONTAM"]["reason"] == "eval_sequence_homolog"
    assert ledger["TOO-LONG"]["reason"] == "does_not_fit_contacts_ring"
    assert ledger["NO-INTERFACE"]["reason"] == "no_reported_interaction"
    assert ledger["CLASH"]["reason"] == "too_many_backbone_clashes"
    assert ledger["LOW"]["reason"] == "below_relaxed_quality_floor"


def test_selection_fails_loud_after_writing_frontier_when_quota_is_unmet(
    tmp_path: Path,
) -> None:
    model = _model("ONLY", "homodimer", "H1", "H1", source_quality_pass=True)
    input_path = tmp_path / "normalized.parquet"
    annotations_path = tmp_path / "annotations.parquet"
    pq.write_table(pa.Table.from_pylist([model]), input_path)
    pq.write_table(pa.Table.from_pylist(_annotations({"H1"})), annotations_path)

    try:
        run_selection(
            str(input_path),
            annotations_path,
            tmp_path / "out",
            target_docs=2,
            min_heterodimers=0,
            database=tmp_path / "selection.duckdb",
        )
    except RuntimeError as error:
        assert "below the target" in str(error)
    else:
        raise AssertionError("an unmet production quota must fail loud")
    assert (tmp_path / "out" / "selection.json").exists()
    assert (tmp_path / "out" / "selection_ledger.parquet").exists()
