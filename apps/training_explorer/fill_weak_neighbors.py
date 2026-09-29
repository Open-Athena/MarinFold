"""Fill fewer-than-10 neighbor lists with the best weak native alignments.

An E<=10 MMseqs search can report no relative for a designed protein. A
second, deliberately permissive search supplies the requested ten ranked
training sequences without pretending those weak matches are homologs.
"""

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK_ROOT = Path("/data/training_explorer_weak_neighbors")
MMSEQS = Path("/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs")
TARGET = Path("/data/exp213_overlap/targetDB")
DROP = Path("/data/exp225_decontam/droplist_final.parquet")
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits,tlen"
ORIGINAL_CORPUS = 70_889_604
LATEST_CORPUS = 232_090_905


def run(log: Path, *arguments: str) -> None:
    """Run one MMseqs2 phase with a durable log and fail on errors."""
    with log.open("w") as handle:
        subprocess.run(
            [str(MMSEQS), *map(str, arguments)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    """Require ten distinct non-self targets for every requested protein."""
    snapshots = {
        name: json.loads((DATA / f"{name}.json").read_text())
        for name in ("latest", "original", "eval")
    }
    incomplete = {
        protein["id"]: (name, protein)
        for name, snapshot in snapshots.items()
        for protein in snapshot["proteins"]
        if len(protein["neighbors"]) < 10
    }
    if not incomplete:
        print("Every protein already has ten reported neighbors", flush=True)
        return
    print(f"Searching weak matches for {len(incomplete)} proteins", flush=True)
    content = "".join(
        f">{key}\n{protein['sequence']}\n"
        for key, (_, protein) in sorted(incomplete.items())
    )
    work = WORK_ROOT / hashlib.sha256(content.encode()).hexdigest()[:12]
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "queries.fasta"
    fasta.write_text(content)
    m8 = work / "hits.m8"
    if not m8.exists():
        run(work / "createdb.log", "createdb", fasta, work / "queryDB")
        run(
            work / "search.log",
            "search",
            work / "queryDB",
            TARGET,
            work / "alnDB",
            work / "tmp",
            "-s",
            "7.5",
            "--max-seqs",
            "5000",
            "-e",
            "1000000",
            "--min-ungapped-score",
            "0",
            "--threads",
            "48",
            "--split-memory-limit",
            "64G",
        )
        run(
            work / "convert.log",
            "convertalis",
            work / "queryDB",
            TARGET,
            work / "alnDB",
            m8,
            "--format-output",
            FORMAT,
            "--threads",
            "48",
        )
    with m8.open(newline="") as handle:
        counts = Counter(row[0] for row in csv.reader(handle, delimiter="\t"))
    rescue = {
        key: protein
        for key, (_, protein) in incomplete.items()
        if len(protein["neighbors"]) + counts[key] < 10
    }
    files = [m8]
    if rescue:
        print(
            f"Searching {len(rescue)} low-complexity sequences without masking",
            flush=True,
        )
        rescue_work = work / "unmasked"
        rescue_work.mkdir(exist_ok=True)
        rescue_fasta = rescue_work / "queries.fasta"
        rescue_fasta.write_text(
            "".join(
                f">{key}\n{protein['sequence']}\n"
                for key, protein in sorted(rescue.items())
            )
        )
        rescue_m8 = rescue_work / "hits.m8"
        if not rescue_m8.exists():
            run(
                rescue_work / "createdb.log",
                "createdb",
                rescue_fasta,
                rescue_work / "queryDB",
            )
            run(
                rescue_work / "search.log",
                "search",
                rescue_work / "queryDB",
                TARGET,
                rescue_work / "alnDB",
                rescue_work / "tmp",
                "-s",
                "7.5",
                "--max-seqs",
                "5000",
                "-e",
                "1000000",
                "--min-ungapped-score",
                "0",
                "--mask",
                "0",
                "--comp-bias-corr",
                "0",
                "--threads",
                "48",
                "--split-memory-limit",
                "64G",
            )
            run(
                rescue_work / "convert.log",
                "convertalis",
                rescue_work / "queryDB",
                TARGET,
                rescue_work / "alnDB",
                rescue_m8,
                "--format-output",
                FORMAT,
                "--threads",
                "48",
            )
        files.append(rescue_m8)
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
    candidates: dict[str, list[dict]] = defaultdict(list)
    for hits_file in files:
        with hits_file.open(newline="") as handle:
            for values in csv.reader(handle, delimiter="\t"):
                query, target, identity, _, qcov, tcov, evalue, bits, tlen = values
                name, protein = incomplete[query]
                arm, local = target.split("|", 1)
                shard, row, entry = local.split("_", 2)
                if name != "original" and f"{arm}|{shard}_{row}" in removed:
                    continue
                if target == protein.get("targetId"):
                    continue
                if (
                    name == "latest"
                    and protein["source"].startswith("native")
                    and entry == protein["entryId"]
                ):
                    continue
                source = (
                    ("AFDB" if arm == "afdb" else "ESM-Atlas")
                    if name == "original"
                    else ("native AFDB" if arm == "afdb" else "native ESM-Atlas")
                )
                unmasked = hits_file != m8
                candidates[query].append(
                    {
                        "id": target,
                        "label": entry,
                        "source": source,
                        "length": int(tlen),
                        "identity": float(identity),
                        "queryCoverage": float(qcov),
                        "targetCoverage": float(tcov),
                        "evalue": float(evalue)
                        * (
                            LATEST_CORPUS / ORIGINAL_CORPUS if name != "original" else 1
                        ),
                        "bitscore": float(bits),
                        "weakFallback": float(evalue) > 10 or unmasked,
                        "lowComplexityRescue": unmasked,
                    }
                )
    for key, (name, protein) in incomplete.items():
        protein.setdefault("reportedAtE10", bool(protein["neighbors"]))
        if key in rescue:
            protein["lowComplexitySearch"] = True
        merged = {hit["id"]: hit for hit in protein["neighbors"]}
        for hit in candidates[key]:
            merged.setdefault(hit["id"], hit)
        ranked = sorted(
            merged.values(),
            key=lambda hit: (-hit["bitscore"], hit["evalue"], hit["id"]),
        )
        protein["neighbors"] = ranked[:10]
        if len(protein["neighbors"]) != 10:
            raise ValueError(f"Only {len(protein['neighbors'])} neighbors for {key}")
    for name, snapshot in snapshots.items():
        snapshot["search"]["weakFallback"] = {
            "queries": sum(
                key in incomplete for key in (p["id"] for p in snapshot["proteins"])
            ),
            "evalueLimit": 1_000_000,
            "minUngappedScore": 0,
            "lowComplexityRescueQueries": sum(
                key in rescue for key in (p["id"] for p in snapshot["proteins"])
            ),
            "source": "original 70.9M MMseqs database; exp225 removed rows excluded for latest/eval",
        }
        (DATA / f"{name}.json").write_text(json.dumps(snapshot, separators=(",", ":")))
    print(f"Filled all {len(incomplete)} deficient lists to ten neighbors", flush=True)


if __name__ == "__main__":
    main()
