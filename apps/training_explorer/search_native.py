"""Search latest/eval queries against exp277's decontaminated native corpora.

The exp213 MMseqs target is the complete pre-filter AFDB/ESM union. Its 1.37M
exp225 removed rows are filtered by exact arm/shard/row identity after a deep
search, so the retained hits belong to the native exp277 training sources.
"""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
WORK = Path("/data/training_explorer_native_search")
MMSEQS = Path("/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs")
TARGET = Path("/data/exp213_overlap/targetDB")
DROP = Path("/data/exp225_decontam/droplist_final.parquet")
OUTPUT = HERE / "data/native_hits.json"
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits,tlen"


def run(*arguments: str) -> None:
    """Run MMseqs2 and propagate any indexing or search failure."""
    subprocess.run([str(MMSEQS), *map(str, arguments)], check=True)


def main() -> None:
    """Search, remove decontaminated documents, and retain ranked candidates."""
    latest = json.loads((HERE / "data/latest.json").read_text())
    evaluation = json.loads((HERE / "data/eval.json").read_text())
    proteins = latest["proteins"] + evaluation["proteins"]
    WORK.mkdir(parents=True, exist_ok=True)
    fasta = WORK / "queries.fasta"
    fasta.write_text("".join(f">{p['id']}\n{p['sequence']}\n" for p in proteins))
    query_db = WORK / "queryDB"
    aln_db = WORK / "alnDB"
    m8 = WORK / "hits.m8"
    if not m8.exists():
        run("createdb", fasta, query_db)
        run(
            "search",
            query_db,
            TARGET,
            aln_db,
            WORK / "tmp",
            "-s",
            "7.5",
            "--max-seqs",
            "3000",
            "-e",
            "10",
            "--threads",
            "48",
            "--split-memory-limit",
            "64G",
        )
        run(
            "convertalis",
            query_db,
            TARGET,
            aln_db,
            m8,
            "--format-output",
            FORMAT,
            "--threads",
            "48",
        )
    table = pq.read_table(DROP, columns=["arm", "shard", "row"])
    removed = {
        f"{arm}|{int(shard):05d}_{int(row)}"
        for arm, shard, row in zip(
            table.column("arm").to_pylist(),
            table.column("shard").to_pylist(),
            table.column("row").to_pylist(),
            strict=True,
        )
    }
    hits: dict[str, list[dict]] = defaultdict(list)
    with m8.open(newline="") as handle:
        for values in csv.reader(handle, delimiter="\t"):
            query, target, identity, _, qcov, tcov, evalue, bits, tlen = values
            arm, local = target.split("|", 1)
            shard, row, entry = local.split("_", 2)
            if f"{arm}|{shard}_{row}" in removed:
                continue
            hits[query].append(
                {
                    "id": target,
                    "label": entry,
                    "source": "native AFDB" if arm == "afdb" else "native ESM-Atlas",
                    "length": int(tlen),
                    "identity": float(identity),
                    "queryCoverage": float(qcov),
                    "targetCoverage": float(tcov),
                    "evalue": float(evalue),
                    "bitscore": float(bits),
                }
            )
    output = {}
    for protein in proteins:
        ranked = [
            h
            for h in hits[protein["id"]]
            if h["label"] != protein.get("entryId")
            or h["source"] != protein.get("source")
        ]
        ranked.sort(key=lambda h: (-h["bitscore"], h["evalue"], h["id"]))
        output[protein["id"]] = ranked[:50]
    OUTPUT.write_text(json.dumps(output, separators=(",", ":")))
    print(f"Retained native candidates for {len(proteins)} queries", flush=True)


if __name__ == "__main__":
    main()
