# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence-leakage check: candidates vs the AFDB-24M training representatives.

The sequence analogue of the exp41 Foldseek run. For each candidate we MMseqs2-
search its sequence against the 1.33M training cluster-representative sequences
(``notes/eval-dataset-design.md`` §5, axis 1) and ask: is there a training
sequence close enough that the candidate is effectively memorised?

The rep sequences are extracted with ``mmseqs convert2fasta`` straight from the
Foldseek ``targetDB`` (its base component is the amino-acid DB), so no extra
download is needed; the rep id -> split map comes from ``reps_manifest.csv``.

Verdict (tunable): a best **train** hit with identity >= 0.30 over query
coverage >= 0.50 = ``redundant_seq`` (the twilight-zone novel-family boundary);
otherwise ``novel_seq``. Hits to val/test reps are reported but don't count as
training leakage. Writes ``data/candidate_seq_leakage.csv``.

Needs the ``mmseqs`` binary (``$MMSEQS_BIN``, ``$PATH``, or the cached static
build under ``~/.cache/marinfold/mmseqs``).
"""

import argparse
import csv
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP41_DB = HERE.parent / "exp41_evals_foldseek_train_similarity" / "db_full"
DEFAULT_TARGETDB = EXP41_DB / "db" / "targetDB"
DEFAULT_REPS_MANIFEST = EXP41_DB / "reps_manifest.csv"

MMSEQS_DOWNLOAD = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
_CACHE = Path.home() / ".cache" / "marinfold" / "mmseqs"

REDUNDANT_ID = 0.30   # twilight-zone novel-family boundary
MIN_QCOV = 0.50


def ensure_mmseqs() -> str:
    """Return a path to the mmseqs binary, installing the static build if needed."""
    env_bin = os.environ.get("MMSEQS_BIN")
    if env_bin and Path(env_bin).exists():
        return str(Path(env_bin).resolve())
    on_path = shutil.which("mmseqs")
    if on_path:
        return on_path
    binary = _CACHE / "mmseqs" / "bin" / "mmseqs"
    if not binary.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        tar = _CACHE / "mmseqs.tar.gz"
        print(f"[seq_leakage] downloading mmseqs from {MMSEQS_DOWNLOAD}")
        urllib.request.urlretrieve(MMSEQS_DOWNLOAD, tar)
        with tarfile.open(tar) as tf:
            try:
                tf.extractall(_CACHE, filter="data")
            except TypeError:
                tf.extractall(_CACHE)
    return str(binary.resolve())


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def load_splits(reps_manifest: Path) -> dict[str, str]:
    """representative_id -> split (train/val/test)."""
    with reps_manifest.open() as fh:
        return {r["representative_id"]: r["split"] for r in csv.DictReader(fh)}


def search(mmseqs: str, candidates_fasta: Path, targetdb: Path, work: Path,
           sensitivity: float) -> Path:
    """Build DBs, search candidates vs reps, return the convertalis .m8 path."""
    work.mkdir(parents=True, exist_ok=True)
    reps_fasta = work / "reps.fasta"
    if not reps_fasta.exists() or reps_fasta.stat().st_size == 0:
        print("  extracting rep sequences from targetDB (convert2fasta) ...")
        run([mmseqs, "convert2fasta", str(targetdb), str(reps_fasta)])
    train_db, cand_db, aln_db = work / "trainDB", work / "candDB", work / "alnDB"
    # Clear any stale search outputs: mmseqs refuses to overwrite an existing
    # result DB, so a re-run (e.g. after adding candidates) would fail on the
    # previous run's alnDB / search tmp. Remove them so re-runs are idempotent.
    search_tmp = work / "tmp"
    if search_tmp.exists():
        shutil.rmtree(search_tmp)
    for stale in work.glob("alnDB*"):
        stale.unlink()
    run([mmseqs, "createdb", str(reps_fasta), str(train_db)])
    run([mmseqs, "createdb", str(candidates_fasta), str(cand_db)])
    run([mmseqs, "search", str(cand_db), str(train_db), str(aln_db), str(search_tmp),
         "-s", str(sensitivity), "--max-seqs", "300", "-e", "10"])
    m8 = work / "aln.m8"
    run([mmseqs, "convertalis", str(cand_db), str(train_db), str(aln_db), str(m8),
         "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits"])
    return m8


def best_hits(m8: Path, splits: dict[str, str]) -> dict[str, dict]:
    """Per query: the best hit overall and the best TRAIN hit (by bitscore)."""
    best: dict[str, dict] = {}
    best_train: dict[str, dict] = {}
    with m8.open() as fh:
        for line in fh:
            q, t, fident, alnlen, qcov, tcov, evalue, bits = line.rstrip("\n").split("\t")
            rec = {"target": t, "fident": float(fident), "qcov": float(qcov),
                   "tcov": float(tcov), "evalue": float(evalue), "bits": float(bits),
                   "split": splits.get(t, "")}
            if q not in best or rec["bits"] > best[q]["bits"]:
                best[q] = rec
            if rec["split"] == "train" and (q not in best_train or rec["bits"] > best_train[q]["bits"]):
                best_train[q] = rec
    return {"best": best, "best_train": best_train}


def verdict(train_hit: dict | None) -> str:
    if not train_hit:
        return "novel_seq"
    if train_hit["fident"] >= REDUNDANT_ID and train_hit["qcov"] >= MIN_QCOV:
        return "redundant_seq"
    return "novel_seq"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates-fasta", type=Path, default=HERE / "data" / "candidates.fasta")
    p.add_argument("--candidates-csv", type=Path, default=HERE / "data" / "candidate_sequences.csv")
    p.add_argument("--targetdb", type=Path, default=DEFAULT_TARGETDB)
    p.add_argument("--reps-manifest", type=Path, default=DEFAULT_REPS_MANIFEST)
    p.add_argument("--work", type=Path, default=HERE / "tmp" / "leakage")
    p.add_argument("--out", type=Path, default=HERE / "data" / "candidate_seq_leakage.csv")
    p.add_argument("--sensitivity", type=float, default=7.5, help="mmseqs -s (higher = more remote homologs).")
    args = p.parse_args()

    mmseqs = ensure_mmseqs()
    print(f"mmseqs: {mmseqs}")
    splits = load_splits(args.reps_manifest)
    print(f"loaded {len(splits)} rep->split entries")
    m8 = search(mmseqs, args.candidates_fasta, args.targetdb, args.work, args.sensitivity)
    hits = best_hits(m8, splits)

    # source/length per candidate from the sequences CSV.
    meta = {r["stem"]: r for r in csv.DictReader(args.candidates_csv.open())}
    rows = []
    for stem, r in meta.items():
        b = hits["best"].get(stem)
        bt = hits["best_train"].get(stem)
        rows.append({
            "source": r["source"], "stem": stem, "pdb_id": r["pdb_id"], "query_len": r["length"],
            "best_hit": b["target"] if b else "", "best_fident": f"{b['fident']:.3f}" if b else "",
            "best_split": b["split"] if b else "",
            "best_train_hit": bt["target"] if bt else "",
            "best_train_fident": f"{bt['fident']:.3f}" if bt else "",
            "best_train_qcov": f"{bt['qcov']:.3f}" if bt else "",
            "verdict": verdict(bt),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "stem", "pdb_id", "query_len", "best_hit", "best_fident", "best_split",
              "best_train_hit", "best_train_fident", "best_train_qcov", "verdict"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    import collections
    print(f"\nWrote {len(rows)} rows -> {args.out}")
    print("verdicts:", dict(collections.Counter(r["verdict"] for r in rows)))
    by_src = collections.defaultdict(collections.Counter)
    for r in rows:
        by_src[r["source"]][r["verdict"]] += 1
    for src, c in by_src.items():
        print(f"  {src}: {dict(c)}")


if __name__ == "__main__":
    main()
