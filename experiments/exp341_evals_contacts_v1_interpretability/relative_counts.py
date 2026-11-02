# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A3/A4 panel, stage 2: count each candidate's relatives in exp232's training corpus.

A3 bins held-out structures by how many of their relatives survived #225's
decontamination, i.e. are in the corpus exp232 trained on. That count cannot
be read off ``struct_cluster_id`` (#225 purged whole Foldseek clusters, so a
removed structure's surviving relatives are outside its cluster), so it is
measured by sequence search:

1. Decode every training sequence from the two decontaminated native corpora
   on the HF bucket (``contacts_v1_decontam``: 3,963,003 AFDB documents;
   ``contacts_v1_esm_atlas_decontam``: 65,553,178 ESM Atlas documents) into
   one FASTA, reading only ``entry_id``, ``document``, ``seq_len`` and
   ``n_term_index``.
2. MMseqs2-search the stage 1 candidates against it at ``-s 7.5`` (exp213's
   and exp225's sensitivity), uncensored: ``-e 1000`` and a large
   ``--max-seqs``, as exp225's filter search did.
3. Per candidate, count training sequences that are relatives under two
   rules: the plan's (``fident >= 0.30`` and ``qcov >= 0.50``) and #225's
   removal rule (that, or ``evalue <= 1e-3``; ``decontam_lib``). A kept
   candidate's hit on its own training sequence is not counted.

Writes ``<work>/relative_counts.parquet``: ``entry_id``, ``set``,
``n_relatives`` (plan rule), ``n_relatives_225`` (#225 rule), ``n_hits``
(all reported hits) and ``censored`` (``n_hits`` reached ``--max-seqs``).

Needs ~170 GB of streamed input, ~40 GB of FASTA and a few hundred GB of
MMseqs2 scratch: run it on a large-disk CPU instance (``launch_aws.sh
relatives``), not a workstation.

    uv run python relative_counts.py --candidates candidates.parquet --work /data/rel
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from hf_bucket import list_parquet, read_columns
from marinfold.document_structures.contacts_v1.read import sequence_from_document

CORPORA = {
    "afdb": "data/document_structures/contacts_v1_decontam/train",
    "esm": "data/document_structures/contacts_v1_esm_atlas_decontam/train",
}
EXPECTED_SEQUENCES = 3_963_003 + 65_553_178
MIN_IDENTITY = 0.30
MIN_QCOV = 0.50
MAX_EVALUE_225 = 1e-3
MMSEQS_URL = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
FORMAT = "query,target,fident,qcov,evalue"


def ensure_mmseqs(work: Path) -> str:
    on_path = shutil.which("mmseqs")
    if on_path:
        return on_path
    binary = work / "mmseqs" / "bin" / "mmseqs"
    if not binary.exists():
        archive = work / "mmseqs.tar.gz"
        urllib.request.urlretrieve(MMSEQS_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter="data")
    return str(binary)


def shard_fasta(path: str, arm: str, out_dir: Path) -> int:
    """Decode one corpus shard to ``<out_dir>/<shard>.fasta``; returns its sequence count."""
    out = out_dir / f"{arm}-{Path(path).stem}.fasta"
    done = out.with_suffix(".done")
    if done.exists():
        return int(done.read_text())
    table = read_columns(path, ["entry_id", "document", "seq_len", "n_term_index"], block_size=1 << 23)
    count = 0
    partial = out.with_suffix(".partial")
    with partial.open("w") as handle:
        for entry_id, document, seq_len, n_term in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            handle.write(f">{entry_id}\n{sequence_from_document(document, seq_len, n_term)}\n")
            count += 1
    partial.replace(out)
    done.write_text(str(count))
    return count


def build_target_fasta(work: Path, workers: int) -> Path:
    target = work / "train.fasta"
    if target.exists():
        return target
    shard_dir = work / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(path, arm) for arm, prefix in CORPORA.items() for path in list_parquet(prefix)]
    total = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, count in enumerate(pool.map(lambda job: shard_fasta(job[0], job[1], shard_dir), jobs), start=1):
            total += count
            if done % 200 == 0 or done == len(jobs):
                print(f"[fasta] {done}/{len(jobs)} shards, {total:,} sequences, {time.time() - start:.0f}s", flush=True)
    if total != EXPECTED_SEQUENCES:
        raise ValueError(f"decoded {total:,} training sequences, expected {EXPECTED_SEQUENCES:,}")
    partial = target.with_suffix(".partial")
    with partial.open("wb") as out:
        for path in sorted(shard_dir.glob("*.fasta")):
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, out)
    partial.replace(target)
    shutil.rmtree(shard_dir)
    return target


def run(command: list[str]) -> None:
    print("[mmseqs]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True, help="stage 1 candidates.parquet")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, help="parallel shard downloads")
    parser.add_argument("--threads", type=int, default=0, help="mmseqs threads (0 = all cores)")
    parser.add_argument("--max-seqs", type=int, default=100_000)
    parser.add_argument("--split-memory-limit", default="", help="mmseqs --split-memory-limit, e.g. 200G")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(args.candidates, columns=["set", "entry_id", "sequence"])
    if candidates["entry_id"].duplicated().any():
        raise ValueError("candidate entry_ids must be unique")
    queries = args.work / "candidates.fasta"
    queries.write_text("".join(f">{e}\n{s}\n" for e, s in zip(candidates["entry_id"], candidates["sequence"])))

    target = build_target_fasta(args.work, args.workers)
    mmseqs = ensure_mmseqs(args.work)
    query_db, target_db, result_db = args.work / "queryDB", args.work / "trainDB", args.work / "resultDB"
    run([mmseqs, "createdb", str(queries), str(query_db)])
    if not target_db.exists():
        run([mmseqs, "createdb", str(target), str(target_db)])
    search = [mmseqs, "search", str(query_db), str(target_db), str(result_db), str(args.work / "tmp"),
              "-s", "7.5", "-e", "1000", "--max-seqs", str(args.max_seqs)]
    if args.threads:
        search += ["--threads", str(args.threads)]
    if args.split_memory_limit:
        search += ["--split-memory-limit", args.split_memory_limit]
    run(search)
    m8 = args.work / "hits.m8"
    run([mmseqs, "convertalis", str(query_db), str(target_db), str(result_db), str(m8), "--format-output", FORMAT])

    hits = pd.read_csv(m8, sep="\t", names=FORMAT.split(","))
    hits = hits[hits["query"] != hits["target"]]  # a kept candidate's own training sequence
    plan_rule = (hits["fident"] >= MIN_IDENTITY) & (hits["qcov"] >= MIN_QCOV)
    rule_225 = plan_rule | (hits["evalue"] <= MAX_EVALUE_225)
    counts = pd.DataFrame({
        "n_relatives": plan_rule.groupby(hits["query"]).sum(),
        "n_relatives_225": rule_225.groupby(hits["query"]).sum(),
        "n_hits": hits.groupby("query").size(),
    })
    out = candidates[["entry_id", "set"]].merge(counts, left_on="entry_id", right_index=True, how="left")
    out[["n_relatives", "n_relatives_225", "n_hits"]] = out[["n_relatives", "n_relatives_225", "n_hits"]].fillna(0).astype(int)
    out["censored"] = out["n_hits"] >= args.max_seqs - 1
    out.to_parquet(args.work / "relative_counts.parquet", index=False)
    print(out.groupby("set")[["n_relatives", "n_relatives_225"]].describe().T, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
