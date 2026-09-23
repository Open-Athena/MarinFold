"""Turn the scale run's quality-pass candidates into a selected training corpus.

The bounded audit in scale_audit.py samples; this applies the same rules to all
of them. Stages run on one CPU node because Foldseek and MMseqs2 want every
structure on local disk, and each stage records a marker so a preempted node
resumes at the last finished stage instead of repeating hours of extraction.

Screening rules are the audit's, unchanged: a sequence is excluded when
(identity >= 0.30 and coverage of the shorter sequence >= 0.50) or E <= 1e-3
against the frozen legacy plus FoldBench reference, and a structure is excluded
when it matches an evaluation structure at TM >= 0.8 in both normalizations with
>= 0.8 coverage of both chains.

Clustering deviates deliberately. The audit built connected components from an
all-versus-all TM-align table, which is quadratic and, as its own report noted,
chains into very large transitive components. At three million structures that
is neither affordable nor well behaved, so this uses Foldseek's cascaded
clustering at the same TM and coverage thresholds. That is set-cover, not
connected components, so cluster membership is not identical to the audit's;
the cap of five per cluster is applied to these clusters and the command is
recorded alongside the output.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

TM_THRESHOLD = 0.8
COVERAGE_THRESHOLD = 0.8
CLUSTER_CAP = 5
SEARCH_FIELDS = "query,target,qtmscore,ttmscore,qcov,tcov,evalue"
SEQUENCE_FIELDS = "query,target,fident,qcov,tcov,qlen,tlen,evalue"
STAGES = ("extract", "seqscreen", "structscreen", "cluster", "assemble")


def sequence_excluded(row: dict) -> bool:
    """Apply the audit's frozen rule to one MMseqs2 hit row.

    Coverage is taken on the shorter sequence, and the E-value clause has no
    identity escape hatch: either condition alone excludes the candidate.
    """
    shorter = float(
        row["qcov"] if int(row["qlen"]) <= int(row["tlen"]) else row["tcov"]
    )
    return (float(row["fident"]) >= 0.3 and shorter >= 0.5) or float(
        row["evalue"]
    ) <= 1e-3


def structure_excluded(row: dict) -> bool:
    """True when a hit is a fine structural duplicate in both normalizations."""
    return (
        min(float(row["qtmscore"]), float(row["ttmscore"])) >= TM_THRESHOLD
        and min(float(row["qcov"]), float(row["tcov"])) >= COVERAGE_THRESHOLD
    )


def apply_cap(members: dict, index: dict, cap: int = CLUSTER_CAP) -> tuple[set, list]:
    """Keep the best `cap` members of each cluster, ranked as the audit ranks them."""
    selected, rows = set(), []
    for representative, group in sorted(members.items()):
        ranked = sorted(
            group,
            key=lambda stem: (-index[stem]["plddt"], index[stem]["scrmsd"], stem),
        )
        for rank, stem in enumerate(ranked):
            rows.append(
                {
                    "stem": stem,
                    "cluster": representative,
                    "rank": rank,
                    "selected": rank < cap,
                }
            )
            if rank < cap:
                selected.add(stem)
    return selected, rows


def run(command: list, log: Path) -> None:
    """Run one tool, keeping its output on disk rather than in the job log."""
    with log.open("ab") as handle:
        subprocess.run(
            [str(part) for part in command],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


STAGE_ARTIFACTS = {
    "extract": ("index.parquet", "candidates.fasta", "pdb"),
    "seqscreen": ("sequence-exclusions.txt",),
    "structscreen": ("structure-exclusions.txt",),
    "cluster": ("selected.txt",),
    "assemble": (),
}
# Everything here is copied to the object store when its stage finishes and
# pulled back when a replacement node finds it missing. `pdb/` is deliberately
# absent: 586 GB is not worth storing when extraction rebuilds it in 24 minutes,
# whereas losing the screens costs days.
PERSISTED = {
    "extract": ("index.parquet", "candidates.fasta"),
    "seqscreen": ("sequence-exclusions.txt",),
    "structscreen": ("structure-exclusions.txt",),
    "cluster": ("selected.txt", "selection.csv"),
    "assemble": (),
}


def persist_stage(fs, prefix: str, stage: str, work: Path) -> None:
    """Copy a finished stage's outputs off the node's ephemeral disk."""
    for name in PERSISTED[stage]:
        local = work / name
        if local.exists():
            fs.put_file(str(local), f"{prefix}/artifacts/{name}")


def restore_stage(fs, prefix: str, stage: str, work: Path) -> None:
    """Pull a finished stage's outputs back after the node was replaced."""
    for name in PERSISTED[stage]:
        local, remote = work / name, f"{prefix}/artifacts/{name}"
        if not local.exists() and fs.exists(remote):
            print(f"[stage] restoring {stage}/{name} from the object store", flush=True)
            local.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(remote, str(local))


def stage_done(fs, prefix: str, stage: str, work: Path) -> bool:
    """A stage is finished only if its marker and its local outputs both survive.

    Markers live in the object store but stage outputs live on the node's
    ephemeral disk. A replacement pod trusting the marker alone would skip
    extraction and then run the next stage against files that do not exist, so
    a marker without its artifacts means the stage has to run again.
    """
    if not fs.exists(f"{prefix}/stages/{stage}.json"):
        return False
    restore_stage(fs, prefix, stage, work)
    missing = [name for name in STAGE_ARTIFACTS[stage] if not (work / name).exists()]
    if missing:
        print(
            f"[stage] {stage} is marked complete but {missing} are missing from "
            f"{work}; re-running rather than trusting the marker",
            flush=True,
        )
        return False
    return True


def mark_stage(fs, prefix: str, stage: str, payload: dict) -> None:
    fs.pipe_file(
        f"{prefix}/stages/{stage}.json", json.dumps(payload, indent=2).encode()
    )
    print(f"[stage] {stage} complete: {payload}", flush=True)


def fetch_tools(fs, work: Path) -> tuple[Path, Path]:
    """Pull the exact Foldseek and MMseqs2 builds the audit used."""
    tools = work / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    base = "marin-us-east-02a/MarinFold/exp278-proteina/tools"
    for name in ("foldseek", "mmseqs"):
        target = tools / name
        if not target.exists():
            target.write_bytes(fs.cat(f"{base}/{name}"))
            target.chmod(0o755)
    print("tool versions:", fs.cat(f"{base}/versions.json").decode(), flush=True)
    return tools / "foldseek", tools / "mmseqs"


def fetch_reference(fs, work: Path) -> Path:
    """Pull the frozen sequence and evaluation-structure reference."""
    import tarfile

    reference = work / "reference"
    if not reference.exists():
        reference.mkdir(parents=True, exist_ok=True)
        data = fs.cat(
            "marin-us-east-02a/MarinFold/exp278-proteina/reference/frozen-reference.tar.gz"
        )
        tarfile.open(fileobj=io.BytesIO(data)).extractall(reference, filter="data")
    return reference


def extract(fs, base: str, work: Path, threads: int) -> dict:
    """Write every quality-pass structure to local disk with a compact index."""
    pdb_root = work / "pdb"
    for bucket in range(256):
        (pdb_root / f"{bucket:02x}").mkdir(parents=True, exist_ok=True)
    sources = sorted(
        path
        for path in fs.find(base + "/cases")
        if "/folded/candidates/" in path and path.endswith(".parquet")
    )
    print(f"extract: {len(sources):,} source files", flush=True)
    index_path = work / "index.parquet"
    schema = pa.schema(
        [
            ("stem", pa.string()),
            ("case", pa.string()),
            ("length", pa.int32()),
            ("condition", pa.string()),
            ("plddt", pa.float64()),
            ("scrmsd", pa.float64()),
            ("sequence", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(index_path, schema, compression="zstd")
    kept = scanned = 0
    lock = __import__("threading").Lock()

    def handle(source: str) -> list[dict]:
        table = pq.read_table(
            io.BytesIO(fs.cat(source)),
            columns=[
                "stem",
                "sequence",
                "pdb_content",
                "plddt",
                "scrmsd",
                "quality_pass",
            ],
        )
        rows = []
        case = source.split("/cases/", 1)[1].split("/", 1)[0]
        for row in table.to_pylist():
            if not row["quality_pass"]:
                continue
            stem = row["stem"]
            (pdb_root / stem[-2:] / f"{stem}.pdb").write_text(row["pdb_content"])
            rows.append(
                {
                    "stem": stem,
                    "case": case,
                    "length": len(row["sequence"]),
                    "condition": case.split("-", 2)[2],
                    "plddt": row["plddt"],
                    "scrmsd": row["scrmsd"],
                    "sequence": row["sequence"],
                }
            )
        return rows

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for batch in pool.map(handle, sources):
            with lock:
                scanned += 1
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    kept += len(batch)
                if scanned % 5000 == 0:
                    print(
                        f"  {scanned:,}/{len(sources):,} files, {kept:,} kept",
                        flush=True,
                    )
    writer.close()
    fasta = work / "candidates.fasta"
    with fasta.open("w") as handle:
        for batch in pq.ParquetFile(index_path).iter_batches(
            columns=["stem", "sequence"]
        ):
            for stem, sequence in zip(
                batch["stem"].to_pylist(), batch["sequence"].to_pylist()
            ):
                handle.write(f">{stem}\n{sequence}\n")
    return {"source_files": len(sources), "quality_pass": kept}


def seqscreen(mmseqs: Path, work: Path, reference: Path, threads: int) -> dict:
    """Apply the frozen identity/coverage rule to every candidate sequence."""
    hits = work / "sequence-hits.tsv"
    if not hits.exists():
        run(
            [
                mmseqs,
                "easy-search",
                work / "candidates.fasta",
                reference / "frozen-eval-sequences.fasta",
                hits,
                work / "sequence-tmp",
                "--format-output",
                SEQUENCE_FIELDS,
                "--alignment-mode",
                "3",
                "--max-seqs",
                "10000",
                "-e",
                "10000",
                "-s",
                "7.5",
                "--threads",
                str(threads),
            ],
            work / "sequence-search.log",
        )
    excluded = set()
    with hits.open() as handle:
        for row in csv.DictReader(
            handle, fieldnames=SEQUENCE_FIELDS.split(","), delimiter="\t"
        ):
            if sequence_excluded(row):
                excluded.add(row["query"])
    (work / "sequence-exclusions.txt").write_text("\n".join(sorted(excluded)) + "\n")
    return {
        "excluded": len(excluded),
        "rule": "(identity >=0.30 and shorter coverage >=0.50) or E<=0.001",
    }


def structscreen(
    foldseek: Path,
    work: Path,
    reference: Path,
    threads: int,
    max_seqs: int = 50,
    evalue: float = 0.001,
) -> dict:
    """Exclude candidates that duplicate an evaluation structure.

    The first production run used `--max-seqs 1000 -e 10`, which against an
    888-structure reference TM-aligns every candidate to every evaluation chain:
    about 2.95M x 888 alignments, measured at 53 hours, and it excluded 41,370
    candidates. These tighter defaults cap the alignments per query instead. A
    structure matching at TM >= 0.8 leaves a strong 3Di signal, so the prefilter
    should keep it, but that is an expectation rather than a proof: compare the
    exclusion count against the exhaustive run's 41,370 before trusting a rerun,
    and raise max_seqs if it comes back materially lower.
    """
    hits = work / "eval-structure-hits.tsv"
    if not hits.exists():
        run(
            [
                foldseek,
                "easy-search",
                work / "pdb",
                reference / "eval-structures",
                hits,
                work / "eval-tmp",
                "--alignment-type",
                "1",
                "--format-output",
                SEARCH_FIELDS,
                "--max-seqs",
                str(max_seqs),
                "-e",
                str(evalue),
                "--threads",
                str(threads),
            ],
            work / "eval-search.log",
        )
    excluded = set()
    with hits.open() as handle:
        for row in csv.DictReader(
            handle, fieldnames=SEARCH_FIELDS.split(","), delimiter="\t"
        ):
            if structure_excluded(row):
                excluded.add(Path(row["query"]).stem)
    (work / "structure-exclusions.txt").write_text("\n".join(sorted(excluded)) + "\n")
    return {
        "excluded": len(excluded),
        "rule": f"TM>={TM_THRESHOLD} both normalizations and coverage>={COVERAGE_THRESHOLD} both chains",
    }


def survivors(work: Path) -> dict:
    """Load the index, dropping everything either screen excluded."""
    excluded = set()
    for name in ("sequence-exclusions.txt", "structure-exclusions.txt"):
        text = (work / name).read_text().strip()
        excluded |= set(text.splitlines()) if text else set()
    keep = {}
    for batch in pq.ParquetFile(work / "index.parquet").iter_batches():
        for row in batch.to_pylist():
            if row["stem"] not in excluded:
                keep[row["stem"]] = row
    return keep


def cluster(foldseek: Path, work: Path, threads: int) -> dict:
    """Cluster survivors and keep at most CLUSTER_CAP per fine cluster."""
    keep = survivors(work)
    eligible = work / "eligible"
    if not eligible.exists():
        eligible.mkdir(parents=True)
        for bucket in range(256):
            (eligible / f"{bucket:02x}").mkdir(exist_ok=True)
        for stem in keep:
            source = work / "pdb" / stem[-2:] / f"{stem}.pdb"
            (eligible / stem[-2:] / f"{stem}.pdb").hardlink_to(source)
    result = work / "cluster"
    if not (work / "cluster_cluster.tsv").exists():
        run(
            [
                foldseek,
                "easy-cluster",
                eligible,
                result,
                work / "cluster-tmp",
                "--alignment-type",
                "1",
                "--tmscore-threshold",
                str(TM_THRESHOLD),
                "-c",
                str(COVERAGE_THRESHOLD),
                "--cov-mode",
                "0",
                "--threads",
                str(threads),
            ],
            work / "cluster.log",
        )
    members = defaultdict(list)
    with (work / "cluster_cluster.tsv").open() as handle:
        for line in handle:
            representative, member = line.split()
            members[Path(representative).stem].append(Path(member).stem)
    selected, rows = apply_cap(members, keep)
    with (work / "selection.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stem", "cluster", "rank", "selected"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (work / "selected.txt").write_text("\n".join(sorted(selected)) + "\n")
    return {
        "eligible": len(keep),
        "clusters": len(members),
        "selected": len(selected),
        "cap": CLUSTER_CAP,
        "method": "foldseek easy-cluster (cascaded set-cover), not connected components",
    }


def assemble(fs, base: str, work: Path, out_prefix: str, threads: int) -> dict:
    """Emit the selected contacts-v1 documents as corpus shards."""
    selected = set((work / "selected.txt").read_text().split())
    sources = sorted(
        path
        for path in fs.find(base + "/cases")
        if "/folded/documents-provisional/" in path and path.endswith(".parquet")
    )
    shard, rows, written, kept = 0, [], 0, 0

    def flush(rows: list[dict]) -> None:
        nonlocal shard
        table = pa.Table.from_pylist(rows)
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="zstd")
        fs.pipe_file(
            f"{out_prefix}/documents/shard-{shard:05d}.parquet", buffer.getvalue()
        )
        shard += 1

    def read(source: str) -> list[dict]:
        table = pq.read_table(io.BytesIO(fs.cat(source)))
        return [row for row in table.to_pylist() if row.get("stem") in selected]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for batch in pool.map(read, sources):
            rows.extend(batch)
            written += 1
            kept += len(batch)
            if len(rows) >= 50_000:
                flush(rows)
                rows = []
            if written % 5000 == 0:
                print(
                    f"  assembled {written:,}/{len(sources):,}, {kept:,} documents",
                    flush=True,
                )
    if rows:
        flush(rows)
    return {"documents": kept, "shards": shard}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work", type=Path, default=Path("/tmp/exp278-select"))
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SELECT_THREADS", "96"))
    )
    parser.add_argument("--only", choices=STAGES, action="append")
    parser.add_argument("--eval-max-seqs", type=int, default=50)
    parser.add_argument("--eval-evalue", type=float, default=0.001)
    args = parser.parse_args()

    with fsspec.open(args.manifest, "rt") as handle:
        plan = json.load(handle)
    fs, base = fsspec.core.url_to_fs(plan["output"])
    prefix = f"{base}/selection"
    args.work.mkdir(parents=True, exist_ok=True)
    foldseek, mmseqs = fetch_tools(fs, args.work)
    reference = fetch_reference(fs, args.work)

    for stage in args.only or STAGES:
        if stage_done(fs, prefix, stage, args.work):
            print(f"[stage] {stage} already complete", flush=True)
            continue
        started = time.time()
        if stage == "extract":
            payload = extract(fs, base, args.work, args.threads)
        elif stage == "seqscreen":
            payload = seqscreen(mmseqs, args.work, reference, args.threads)
        elif stage == "structscreen":
            payload = structscreen(
                foldseek,
                args.work,
                reference,
                args.threads,
                args.eval_max_seqs,
                args.eval_evalue,
            )
        elif stage == "cluster":
            payload = cluster(foldseek, args.work, args.threads)
        else:
            payload = assemble(fs, base, args.work, prefix, args.threads)
        payload["elapsed_seconds"] = round(time.time() - started, 1)
        persist_stage(fs, prefix, stage, args.work)
        mark_stage(fs, prefix, stage, payload)


if __name__ == "__main__":
    main()
