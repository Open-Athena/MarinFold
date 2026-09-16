# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the sequence/decontamination annotation table required by selection.

The normalized AFCDB metadata has UniProt accessions but not sequences. This
stage joins those accessions to one or more local FASTA files, computes exact
sequence hashes and lengths, and attaches a completed eval-homology drop list.
It fails if an AFCDB accession is missing or resolves to conflicting sequences.

The intended sequence source is the AlphaFold DB bulk ``sequences.fasta``, whose
headers look like ``>AFDB:AF-A0A919MGV6-F1 <description> UA=A0A919MGV6 ...``.
UniProt ``sp|``/``tr|`` headers and bare accessions are also accepted. Inputs may
be gzipped.

The drop-list input is Parquet with columns ``accession`` and
``eval_decontam_reason``. It contains only excluded accessions; absence from the
table means the accession passed the versioned search that produced the list.
The search provenance belongs next to that input and is copied into the output
provenance JSON by path.
"""

import argparse
import gzip
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ANNOTATION_SCHEMA = pa.schema(
    [
        ("accession", pa.string()),
        ("sequence_sha256", pa.string()),
        ("sequence_length", pa.int32()),
        ("eval_decontam_reason", pa.string()),
    ]
)
_SEQUENCE_SCHEMA = pa.schema(
    [
        ("accession", pa.string()),
        ("sequence_sha256", pa.string()),
        ("sequence_length", pa.int32()),
        ("source_fasta", pa.string()),
        ("source_header", pa.string()),
    ]
)
_AFDB_MODEL = re.compile(r"^(?:AFDB:)?AF-(?P<accession>.+)-F\d+$")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def accession_from_header(header: str) -> str:
    """Extract a UniProt accession from common UniProt/AFDB FASTA headers."""
    token = header.strip().split(maxsplit=1)[0]
    if not token:
        raise ValueError("empty FASTA header")
    fields = token.split("|")
    if len(fields) >= 3 and fields[0] in {"sp", "tr"}:
        return fields[1]
    match = _AFDB_MODEL.fullmatch(token)
    if match:
        return match.group("accession")
    return token


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield ``(accession, sequence, header)`` from one plain or gzipped FASTA."""
    header: str | None = None
    parts: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if header is not None:
                    yield accession_from_header(header), "".join(parts), header
                header = stripped[1:]
                parts = []
                continue
            if header is None:
                raise ValueError(f"{path}:{line_number}: sequence before first header")
            parts.append(stripped)
    if header is not None:
        yield accession_from_header(header), "".join(parts), header


def _sequence_record(
    path: Path, accession: str, sequence: str, header: str
) -> dict[str, Any]:
    normalized = sequence.upper()
    invalid = sorted(set(normalized) - _AMINO_ACIDS)
    if not normalized:
        raise ValueError(f"{path}: {accession} has an empty sequence")
    if invalid:
        raise ValueError(f"{path}: {accession} has invalid residues {invalid}")
    return {
        "accession": accession,
        "sequence_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "sequence_length": len(normalized),
        "source_fasta": str(path.resolve()),
        "source_header": header,
    }


def fasta_to_parquet(
    fasta_paths: list[Path], output: Path, batch_size: int = 100_000
) -> int:
    """Stream FASTA records to Parquet without retaining the sequence set in RAM."""
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, _SEQUENCE_SCHEMA, compression="zstd")
    batch: list[dict[str, Any]] = []
    rows = 0
    try:
        for path in fasta_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            for accession, sequence, header in iter_fasta(path):
                batch.append(_sequence_record(path, accession, sequence, header))
                if len(batch) >= batch_size:
                    writer.write_table(
                        pa.Table.from_pylist(batch, schema=_SEQUENCE_SCHEMA)
                    )
                    rows += len(batch)
                    batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=_SEQUENCE_SCHEMA))
            rows += len(batch)
    finally:
        writer.close()
    return rows


def build_annotations(
    normalized_glob: str,
    fasta_paths: list[Path],
    decontam_drop_list: Path,
    output: Path,
    *,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    """Join complete sequences and a versioned exclusion list to AFCDB accessions."""
    output.parent.mkdir(parents=True, exist_ok=True)
    sequences_path = output.with_suffix(".sequences.parquet")
    fasta_rows = fasta_to_parquet(fasta_paths, sequences_path, batch_size=batch_size)
    if not decontam_drop_list.is_file():
        raise FileNotFoundError(decontam_drop_list)

    con = duckdb.connect()
    conflicts = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT accession
            FROM read_parquet({_sql_literal(sequences_path)})
            GROUP BY accession
            HAVING count(DISTINCT sequence_sha256) > 1
        )
        """
    ).fetchone()[0]
    if conflicts:
        raise ValueError(
            f"FASTA inputs contain {conflicts} accessions with conflicting sequences"
        )

    con.execute(
        f"""
        CREATE TEMP TABLE needed AS
        SELECT DISTINCT accession
        FROM (
            SELECT accession_a AS accession
            FROM read_parquet({_sql_literal(normalized_glob)}, union_by_name=true)
            UNION ALL
            SELECT accession_b AS accession
            FROM read_parquet({_sql_literal(normalized_glob)}, union_by_name=true)
        )
        WHERE accession IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE sequence_index AS
        SELECT
            accession,
            any_value(sequence_sha256) AS sequence_sha256,
            any_value(sequence_length) AS sequence_length
        FROM read_parquet({_sql_literal(sequences_path)})
        GROUP BY accession
        """
    )
    missing = int(
        con.execute(
            "SELECT count(*) FROM needed n LEFT JOIN sequence_index s USING (accession) "
            "WHERE s.accession IS NULL"
        ).fetchone()[0]
    )
    if missing:
        raise ValueError(f"FASTA inputs are missing {missing} AFCDB accessions")

    con.execute(
        f"""
        COPY (
            SELECT
                n.accession,
                s.sequence_sha256,
                s.sequence_length,
                d.eval_decontam_reason
            FROM needed n
            JOIN sequence_index s USING (accession)
            LEFT JOIN read_parquet({_sql_literal(decontam_drop_list)}) d USING (accession)
            ORDER BY n.accession
        ) TO {_sql_literal(output)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    total, excluded = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE eval_decontam_reason IS NOT NULL) "
        f"FROM read_parquet({_sql_literal(output)})"
    ).fetchone()
    result = {
        "normalized_glob": normalized_glob,
        "fasta_paths": [str(path.resolve()) for path in fasta_paths],
        "fasta_rows": fasta_rows,
        "decontam_drop_list": str(decontam_drop_list.resolve()),
        "annotations": str(output.resolve()),
        "accessions": int(total),
        "eval_excluded_accessions": int(excluded),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized", required=True, help="Normalized Parquet file or glob."
    )
    parser.add_argument("--fasta", type=Path, action="append", required=True)
    parser.add_argument("--decontam-drop-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_annotations(
        args.normalized,
        args.fasta,
        args.decontam_drop_list,
        args.out,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
