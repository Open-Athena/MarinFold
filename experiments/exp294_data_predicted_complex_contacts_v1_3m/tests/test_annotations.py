# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from annotations import accession_from_header, build_annotations, iter_fasta


def test_accession_from_common_headers() -> None:
    assert accession_from_header("sp|P12345|PROTEIN description") == "P12345"
    assert accession_from_header("tr|A0A123|ENTRY") == "A0A123"
    assert accession_from_header("AF-Q9XYZ1-F1 model") == "Q9XYZ1"
    assert accession_from_header("Q8TEST arbitrary") == "Q8TEST"


def test_accession_from_alphafold_bulk_sequences_header() -> None:
    """The AFDB bulk ``sequences.fasta`` prefixes the model id with ``AFDB:``."""
    header = (
        "AFDB:AF-A0A919MGV6-F1 Glycosyltransferase RgtA/B/C/D-like domain-containing "
        "protein UA=A0A919MGV6 UI=A0A919MGV6_9ACTN OS=Actinoplanes nipponensis "
        "OX=135950 GN=Ani05nite_25430"
    )
    assert accession_from_header(header) == "A0A919MGV6"


def test_iter_fasta_reads_gzip(tmp_path: Path) -> None:
    path = tmp_path / "sequences.fasta.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(">AFDB:AF-P1-F1 desc UA=P1\nACDE\nFGHI\n>sp|P2|TWO\nKLMN\n")
    assert list(iter_fasta(path)) == [
        ("P1", "ACDEFGHI", "AFDB:AF-P1-F1 desc UA=P1"),
        ("P2", "KLMN", "sp|P2|TWO"),
    ]


def test_build_annotations_hashes_sequences_and_joins_drop_list(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"accession_a": "P1", "accession_b": "P2"},
                {"accession_a": "P2", "accession_b": "P3"},
            ]
        ),
        normalized,
    )
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text(">sp|P1|ONE\nACDE\n>P2 two\nFGHI\n>AF-P3-F1\nKLMN\n")
    drop_list = tmp_path / "drop.parquet"
    pq.write_table(
        pa.table(
            {
                "accession": pa.array(["P2"], type=pa.string()),
                "eval_decontam_reason": pa.array(["identity>=0.3"], type=pa.string()),
            }
        ),
        drop_list,
    )

    output = tmp_path / "annotations.parquet"
    result = build_annotations(
        str(normalized), [fasta], drop_list, output, batch_size=2
    )
    rows = {row["accession"]: row for row in pq.read_table(output).to_pylist()}
    assert result["accessions"] == 3
    assert result["eval_excluded_accessions"] == 1
    assert rows["P1"]["sequence_length"] == 4
    assert len(rows["P1"]["sequence_sha256"]) == 64
    assert rows["P2"]["eval_decontam_reason"] == "identity>=0.3"
    assert rows["P3"]["eval_decontam_reason"] is None


def test_build_annotations_fails_when_an_afcdb_accession_is_missing(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"accession_a": "P1", "accession_b": "MISSING"}]),
        normalized,
    )
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text(">P1\nACDE\n")
    drop_list = tmp_path / "drop.parquet"
    pq.write_table(
        pa.table(
            {
                "accession": pa.array([], type=pa.string()),
                "eval_decontam_reason": pa.array([], type=pa.string()),
            }
        ),
        drop_list,
    )

    with pytest.raises(ValueError, match="missing 1 AFCDB accessions"):
        build_annotations(str(normalized), [fasta], drop_list, tmp_path / "out.parquet")
