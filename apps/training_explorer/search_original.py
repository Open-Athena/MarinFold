"""Search sampled proteins against the complete 70.9M exp199 MMseqs database."""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/original.json"
WORK = Path("/data/training_explorer_original_search")
MMSEQS = Path("/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs")
TARGET = Path("/data/exp213_overlap/targetDB")
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits,tlen"


def run(*arguments: str) -> None:
    """Run MMseqs2 with the pinned exp213 target database."""
    subprocess.run([str(MMSEQS), *map(str, arguments)], check=True)


def main() -> None:
    """Attach the top 10 non-self neighbors to every original sample."""
    data = json.loads(DATA.read_text())
    proteins = data["proteins"]
    WORK.mkdir(parents=True, exist_ok=True)
    query_fasta = WORK / "queries.fasta"
    query_fasta.write_text("".join(f">{p['id']}\n{p['sequence']}\n" for p in proteins))
    query_db = WORK / "queryDB"
    alignment_db = WORK / "alnDB"
    m8 = WORK / "hits.m8"
    if not m8.exists():
        run("createdb", query_fasta, query_db)
        run(
            "search",
            query_db,
            TARGET,
            alignment_db,
            WORK / "tmp",
            "-s",
            "7.5",
            "--max-seqs",
            "1000",
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
            alignment_db,
            m8,
            "--format-output",
            FORMAT,
            "--threads",
            "48",
        )
    hits: dict[str, list[dict]] = defaultdict(list)
    with m8.open(newline="") as handle:
        for values in csv.reader(handle, delimiter="\t"):
            query, target, identity, _, query_cov, target_cov, evalue, bits, tlen = (
                values
            )
            hits[query].append(
                {
                    "id": target,
                    "label": target.split("|", 1)[1].split("_", 2)[2],
                    "source": "AFDB" if target.startswith("afdb|") else "ESM-Atlas",
                    "length": int(tlen),
                    "identity": float(identity),
                    "queryCoverage": float(query_cov),
                    "targetCoverage": float(target_cov),
                    "evalue": float(evalue),
                    "bitscore": float(bits),
                }
            )
    for protein in proteins:
        candidates = [h for h in hits[protein["id"]] if h["id"] != protein["targetId"]]
        candidates.sort(key=lambda h: (-h["bitscore"], h["evalue"], h["id"]))
        protein["neighbors"] = candidates[:10]
        protein["reportedAtE10"] = bool(protein["neighbors"])
    data["neighborsComplete"] = True
    data["search"] = {
        "tool": "MMseqs2",
        "target": str(TARGET),
        "sensitivity": 7.5,
        "maxSeqs": 1000,
        "evalueLimit": 10,
        "rank": "local alignment bit score; self document excluded",
    }
    DATA.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Attached neighbors for {len(proteins)} proteins", flush=True)


if __name__ == "__main__":
    main()
