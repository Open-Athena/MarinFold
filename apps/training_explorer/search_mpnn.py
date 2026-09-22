"""Search every exp277 MPNN design in co-located CoreWeave S3 shards."""

import argparse
import csv
import gzip
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import s3fs

from sample_latest import PREFIX, sequence_from_row


OUTPUT = f"{PREFIX}/training_explorer/2026-09-22"
QUERY = f"{OUTPUT}/latest_eval_queries.fasta"
UNMASKED_QUERY = f"{OUTPUT}/low_complexity_query.fasta"
EXPECTED = 31_702_680 + 130_872_044
MMSEQS_URL = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits,tlen"


def mmseqs_binary(work: Path) -> Path:
    """Install a pinned-in-run MMseqs binary on the local worker scratch."""
    archive = work / "mmseqs.tar.gz"
    urllib.request.urlretrieve(MMSEQS_URL, archive)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(work)], check=True)
    binary = work / "mmseqs/bin/mmseqs"
    if not binary.exists():
        raise FileNotFoundError(binary)
    return binary


def run(binary: Path, *arguments: str) -> None:
    """Execute a local MMseqs2 step and propagate failures."""
    subprocess.run([str(binary), *map(str, arguments)], check=True)


def main() -> None:
    """Decode one disjoint document partition, search it, and publish hits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--unmasked", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    fs = s3fs.S3FileSystem(**json.loads(os.environ["FSSPEC_S3"]))
    work = Path("/tmp/training-explorer")
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "targets.fasta"
    sources = (
        ("mpnn-afdb", f"{PREFIX}/exp266/documents/*.parquet"),
        ("mpnn-esm", f"{PREFIX}/exp266/esm_documents/*.parquet"),
    )
    count = 0
    source_counts: dict[str, int] = {}
    with fasta.open("w") as out:
        for arm, pattern in sources:
            files = sorted(fs.glob(pattern))
            if len(files) not in (199, 3338):
                raise ValueError(f"Unexpected {arm} file count: {len(files)}")
            arm_count = 0
            for file_index in range(args.shard, len(files), args.shards):
                path = files[file_index]
                with fs.open(path, "rb") as stream:
                    parquet = pq.ParquetFile(stream)
                    columns = [
                        "document",
                        "entry_id",
                        "seq_len",
                        "n_term_index",
                        "design_index",
                    ]
                    ordinal = 0
                    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                        for row in batch.to_pylist():
                            sequence = sequence_from_row(row)
                            target_id = f"{arm}|{file_index}_{ordinal}_{row['entry_id']}#{row['design_index']}"
                            out.write(f">{target_id}\n{sequence}\n")
                            ordinal += 1
                    arm_count += ordinal
                    count += ordinal
                print(
                    f"{arm} file {file_index + 1}/{len(files)}: {ordinal:,} rows; total {count:,}",
                    flush=True,
                )
            source_counts[arm] = arm_count
    if not count:
        raise ValueError("No target sequences")

    query_path = UNMASKED_QUERY if args.unmasked else QUERY
    output_path = f"{OUTPUT}/low_complexity" if args.unmasked else OUTPUT
    with (
        fs.open(query_path, "rb") as source,
        (work / "queries.fasta").open("wb") as dest,
    ):
        dest.write(source.read())
    binary = mmseqs_binary(work)
    run(binary, "createdb", work / "queries.fasta", work / "queryDB")
    run(binary, "createdb", fasta, work / "targetDB")
    search_args = [
        "search",
        work / "queryDB",
        work / "targetDB",
        work / "alnDB",
        work / "tmp",
        "-s",
        "7.5",
        "--max-seqs",
        "5000" if args.unmasked else "500",
        "-e",
        "1000000" if args.unmasked else "10",
        "--threads",
        "8",
        "--split-memory-limit",
        "48G",
    ]
    if args.unmasked:
        search_args.extend(
            ("--mask", "0", "--comp-bias-corr", "0", "--min-ungapped-score", "0")
        )
    run(binary, *search_args)
    run(
        binary,
        "convertalis",
        work / "queryDB",
        work / "targetDB",
        work / "alnDB",
        work / "hits.tsv",
        "--format-output",
        FORMAT,
        "--threads",
        "8",
    )
    with (
        (work / "hits.tsv").open(newline="") as handle,
        gzip.open(work / "hits.tsv.gz", "wt") as compressed,
    ):
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != len(FORMAT.split(",")):
                raise ValueError(f"Malformed alignment row: {row[:2]}")
            compressed.write("\t".join(row) + "\n")
    hit_path = f"{output_path}/hits-{args.shard:02d}-of-{args.shards:02d}.tsv.gz"
    manifest_path = f"{output_path}/manifest-{args.shard:02d}-of-{args.shards:02d}.json"
    with fs.open(hit_path, "wb") as remote, (work / "hits.tsv.gz").open("rb") as local:
        remote.write(local.read())
    manifest = {
        "shard": args.shard,
        "shards": args.shards,
        "count": count,
        "sourceCounts": source_counts,
        "query": query_path,
        "format": FORMAT,
        "sensitivity": 7.5,
        "maxSeqs": 5000 if args.unmasked else 500,
        "evalueLimit": 1_000_000 if args.unmasked else 10,
        "unmasked": args.unmasked,
    }
    with fs.open(manifest_path, "wb") as remote:
        remote.write(json.dumps(manifest).encode())
    print(
        f"Published {count:,} sequences to {hit_path}; expected combined total {EXPECTED:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
