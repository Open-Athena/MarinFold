"""Uniform document sampling from all four exp277 source corpora in CoreWeave.

Run on a CoreWeave CPU worker near the source S3 bucket. This reads every
Parquet footer, then reads only the files containing selected document rows.
The output is a small JSON snapshot; no corpus is copied to the workstation.
"""

import json
import os
import random
import re
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import s3fs


BUCKET = "marin-us-east-02a"
PREFIX = f"{BUCKET}/MarinFold"
OUTPUT = f"{PREFIX}/training_explorer/2026-09-22/latest.json"
SEED = 20260922
SAMPLE_SIZE = 100
POPULATION = 232_090_905
SOURCES = (
    (
        "native AFDB",
        f"{PREFIX}/exp232_sweep_cv1_decontam/data/afdb/*.parquet",
        3_963_003,
    ),
    (
        "native ESM-Atlas",
        f"{PREFIX}/exp232_sweep_cv1_decontam/data/esm/*.parquet",
        65_553_178,
    ),
    ("MPNN AFDB", f"{PREFIX}/exp266/documents/*.parquet", 31_702_680),
    ("MPNN ESM-Atlas", f"{PREFIX}/exp266/esm_documents/*.parquet", 130_872_044),
)
RESIDUE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")
AA = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)


def sequence_from_row(row: dict) -> str:
    """Invert the contacts-v1 resampled sequence section."""
    sequence = ["X"] * int(row["seq_len"])
    section = row["document"].split("<begin_statements>", 1)[0]
    start = int(row["n_term_index"])
    for position, three in RESIDUE.findall(section):
        index = (int(position) - start) % 2000
        if index < len(sequence):
            sequence[index] = AA.get(three, "X")
    if sequence.count("X") > row["document"].count("<UNK>"):
        raise ValueError(f"Unfilled sequence positions for {row['entry_id']}")
    return "".join(sequence)


def footer(fs: s3fs.S3FileSystem, path: str) -> tuple[str, int]:
    """Read only the small footer of one co-located source file."""
    with fs.open(path, "rb") as stream:
        return path, pq.read_metadata(stream).num_rows


def get_rows(fs: s3fs.S3FileSystem, path: str, indexes: set[int]) -> dict[int, dict]:
    """Stream projected columns until the last selected row in one file."""
    found: dict[int, dict] = {}
    with fs.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        columns = [
            name
            for name in (
                "document",
                "entry_id",
                "seq_len",
                "n_term_index",
                "design_index",
            )
            if name in parquet.schema_arrow.names
        ]
        ordinal = 0
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                if ordinal in indexes:
                    found[ordinal] = row
                ordinal += 1
            if ordinal > max(indexes):
                break
    if set(found) != indexes:
        raise ValueError(f"Missing selected rows in {path}: {indexes - set(found)}")
    return found


def main() -> None:
    """Write an exactly uniform, auditable latest-corpus sample."""
    config = json.loads(os.environ["FSSPEC_S3"])
    fs = s3fs.S3FileSystem(**config)
    files: list[tuple[str, str, int]] = []
    for name, pattern, expected in SOURCES:
        paths = sorted(fs.glob(pattern))
        if not paths:
            raise ValueError(f"No files for {name}: {pattern}")
        with ThreadPoolExecutor(max_workers=32) as pool:
            counts = list(pool.map(lambda path: footer(fs, path), paths))
        total = sum(count for _, count in counts)
        print(f"{name}: {len(paths)} files, {total:,} rows", flush=True)
        if total != expected:
            raise ValueError(f"{name}: expected {expected:,}, got {total:,}")
        files.extend((name, path, count) for path, count in counts)
    cumulative = []
    running = 0
    for _, _, count in files:
        running += count
        cumulative.append(running)
    if running != POPULATION:
        raise ValueError(f"Population mismatch: {running:,}")

    rng = random.Random(SEED)
    ordinals = sorted(rng.sample(range(POPULATION), SAMPLE_SIZE))
    by_file: dict[int, set[int]] = {}
    for ordinal in ordinals:
        file_index = bisect_right(cumulative, ordinal)
        start = cumulative[file_index - 1] if file_index else 0
        by_file.setdefault(file_index, set()).add(ordinal - start)

    proteins = []
    for file_index, wanted in by_file.items():
        name, path, _ = files[file_index]
        rows = get_rows(fs, path, wanted)
        start = cumulative[file_index - 1] if file_index else 0
        for local_index, row in sorted(rows.items()):
            entry = row["entry_id"]
            sequence = sequence_from_row(row)
            proteins.append(
                {
                    "id": f"latest:{start + local_index}",
                    "label": f"{entry}#{row['design_index']}"
                    if row.get("design_index") is not None
                    else entry,
                    "source": name,
                    "sequence": sequence,
                    "length": len(sequence),
                    "entryId": entry,
                    "designIndex": row.get("design_index"),
                    "corpusOrdinal": start + local_index,
                    "sourceFile": path,
                    "sourceRow": local_index,
                    "structureUrl": None,
                    "structureFormat": "mmcif",
                    "neighbors": [],
                }
            )
        print(f"Sampled {len(proteins)}/{SAMPLE_SIZE}", flush=True)
    proteins.sort(key=lambda p: p["corpusOrdinal"])
    output = {
        "title": "Latest training set",
        "description": "100 uniform document draws from the exact four-corpus exp277 full-epoch mixture.",
        "population": POPULATION,
        "seed": SEED,
        "sampling": "random.sample over row counts from every source Parquet footer",
        "sourceCounts": {name: count for name, _, count in SOURCES},
        "provenance": "exp277/config.py · source Parquet footer census",
        "neighborCorpus": "latest",
        "neighborsComplete": False,
        "proteins": proteins,
    }
    with fs.open(OUTPUT, "wb") as stream:
        stream.write(json.dumps(output, separators=(",", ":")).encode())
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
