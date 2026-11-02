# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 — MMseqs2-search the 554 eval proteins against exp199's training set.

Builds ``data/eval_queries.fasta`` from the two #78 eval manifests (header =
``{dataset}__{stem}`` — keyed on *both*, because a few stems recur across
foldbench and exp65 with different sequences), concatenates the two per-arm
training FASTAs from :mod:`fetch_train_sequences` into one ~17 Gaa target DB,
and searches at high sensitivity.

Then reduces the alignments to **one row per eval protein**
(``data/eval_train_identity.csv``) — the reusable artifact. Per protein and
also per training arm it records:

* ``n_hits`` / ``n_hits_significant`` — alignments, and those at
  ``evalue <= HOMOLOGY_EVALUE``. Censored at ``--max-seqs``; the column
  ``hits_censored`` flags any protein that hit the cap.
* ``best_identity_covered`` — the highest ``fident`` among *significant* hits
  that cover at least ``MIN_QCOV`` of the query. **This is the identity axis**;
  see :mod:`overlap_lib` for why coverage gating is not optional.
* ``best_identity_any`` — the highest ``fident`` among significant hits at any
  coverage. A paranoid upper bound, reported so a short high-identity match
  can't hide behind the coverage gate.
* ``best_bitscore`` / ``best_evalue`` / ``best_target`` — mmseqs' own best hit.

Sensitivity note: ``-s 7.5`` is mmseqs' most sensitive setting and is what
exp65 and exp94 used, so the three experiments' numbers are comparable. It
finds homologs down to roughly the 25–30 % twilight zone; a protein this
reports as ``no_homolog`` may still have an undetectable relative in training,
which makes the homology-free subset a slightly *optimistic* bound on novelty.

    uv run python search_overlap.py --work /data/exp213_overlap
    uv run python search_overlap.py --work /tmp/exp213_smoke --skip-search  # reduce only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from overlap_lib import (
    ARM_AFDB,
    ARM_ESM,
    ARMS,
    HOMOLOGY_EVALUE,
    MIN_QCOV,
    arm_of,
    ensure_mmseqs,
    identity_stratum,
    is_designed,
    run,
)

HERE = Path(__file__).resolve().parent
EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/data")
DEFAULT_MANIFESTS = [EXP78 / "eval_manifest_foldbench.csv", EXP78 / "eval_manifest_exp65.csv"]
#: exp65's manifest already carries a Foldseek `fold_verdict`; FoldBench's does
#: not, so its verdicts are joined from exp41's committed similarity table. With
#: both, the *structural* novelty axis is complete across all 554 proteins and
#: can be crossed against the sequence axis this experiment measures.
EXP41_FOLDBENCH = (HERE.parent / "exp41_evals_foldseek_train_similarity"
                   / "data" / "foldbench_vs_full_reps_similarity.csv")

FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits"
FIELDS = FORMAT.split(",")

#: Strata columns carried through from the eval manifests (exp65's axes), so
#: the identity table joins onto the orthogonal novelty labels with no extra step.
MANIFEST_STRATA = ["neff_tier", "fold_verdict", "seq_leakage", "msa_neff", "length"]


def build_query_fasta(manifests: list[Path], out: Path) -> int:
    """One record per (dataset, stem); header ``{dataset}__{stem}``."""
    seen: set[str] = set()
    lines: list[str] = []
    for manifest in manifests:
        with manifest.open() as fh:
            for row in csv.DictReader(fh):
                key = f"{row['dataset']}__{row['stem']}"
                if key in seen:
                    raise ValueError(f"duplicate query key {key} in {manifest}")
                seen.add(key)
                lines.append(f">{key}\n{row['input_seq'].strip().upper()}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    return len(lines)


def read_manifest_meta(manifests: list[Path],
                       exp41_foldbench: Path | None = None) -> dict[str, dict]:
    """``{dataset}__{stem}`` -> the manifest row's strata columns."""
    meta: dict[str, dict] = {}
    for manifest in manifests:
        with manifest.open() as fh:
            for row in csv.DictReader(fh):
                key = f"{row['dataset']}__{row['stem']}"
                meta[key] = {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "query_len": len(row["input_seq"].strip()),
                    **{c: row.get(c, "") for c in MANIFEST_STRATA},
                }

    if exp41_foldbench and exp41_foldbench.exists():
        filled = 0
        with exp41_foldbench.open() as fh:
            for row in csv.DictReader(fh):
                key = f"foldbench100__{row['stem']}"
                entry = meta.get(key)
                if entry is not None and not entry.get("fold_verdict"):
                    entry["fold_verdict"] = row["verdict"]
                    filled += 1
        print(f"[strata] filled {filled} FoldBench fold_verdict values from exp41",
              flush=True)
    return meta


def concat_targets(work: Path, arms: tuple[str, ...] = ARMS) -> Path:
    """One FASTA over the selected arms; reused if newer than all of them.

    ``arms`` exists so a single arm can be searched on its own — used to
    cross-check the AFDB-only result against exp94's published numbers, and to
    quantify what the ESM-Atlas half adds.
    """
    parts = [work / f"train_{arm}.fasta" for arm in arms]
    missing = [p for p in parts if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}; run fetch_train_sequences.py first")
    out = work / ("train_all.fasta" if tuple(arms) == ARMS
                  else f"train_only_{'_'.join(arms)}.fasta")
    if out.exists() and out.stat().st_mtime >= max(p.stat().st_mtime for p in parts):
        print(f"[targets] reusing {out} ({out.stat().st_size / 1e9:.1f} GB)", flush=True)
        return out
    tmp = out.with_suffix(".partial")
    t0 = time.time()
    with tmp.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
    tmp.rename(out)
    print(f"[targets] {out} ({out.stat().st_size / 1e9:.1f} GB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return out


def search(work: Path, query_fasta: Path, target_fasta: Path, *,
           sensitivity: float, max_seqs: int, evalue: float, threads: int,
           split_memory_limit: str, tag: str = "") -> Path:
    """Search the queries against ``target_fasta``; returns the ``.m8`` path.

    ``tag`` namespaces the mmseqs databases and the output so an arm-only run
    can sit beside the full one without clobbering its (expensive) target DB.
    """
    mmseqs = ensure_mmseqs()
    print(f"[mmseqs] binary: {mmseqs}", flush=True)
    suffix = f"_{tag}" if tag else ""
    query_db = work / f"queryDB{suffix}"
    target_db = work / f"targetDB{suffix}"
    aln_db = work / f"alnDB{suffix}"
    tmp = work / f"mmseqs_tmp{suffix}"

    # mmseqs refuses to overwrite an existing result DB or search tmp, so a
    # re-run must clear them; the target DB is expensive and is reused.
    shutil.rmtree(tmp, ignore_errors=True)
    for stale in list(work.glob(f"alnDB{suffix}*")) + list(work.glob(f"queryDB{suffix}*")):
        stale.unlink()

    run([mmseqs, "createdb", query_fasta, query_db])
    if not target_db.exists():
        t0 = time.time()
        run([mmseqs, "createdb", target_fasta, target_db])
        print(f"[mmseqs] target createdb in {time.time() - t0:.0f}s", flush=True)
    else:
        print("[mmseqs] reusing existing targetDB", flush=True)

    t0 = time.time()
    run([mmseqs, "search", query_db, target_db, aln_db, tmp,
         "-s", sensitivity, "--max-seqs", max_seqs, "-e", evalue,
         "--threads", threads, "--split-memory-limit", split_memory_limit])
    print(f"[mmseqs] search in {time.time() - t0:.0f}s", flush=True)

    m8 = work / f"aln{suffix}.m8"
    run([mmseqs, "convertalis", query_db, target_db, aln_db, m8,
         "--format-output", FORMAT, "--threads", threads])
    print(f"[mmseqs] alignments -> {m8}", flush=True)
    return m8


def reduce_alignments(m8: Path, meta: dict[str, dict], max_seqs: int) -> list[dict]:
    """One row per eval protein: per-arm and overall best hits + counts."""
    blank = lambda: {"n_hits": 0, "n_significant": 0, "best_identity_covered": None,
                     "best_identity_any": None, "best_bits": None,
                     "best_evalue": None, "best_target": ""}
    per_arm: dict[str, dict[str, dict]] = defaultdict(lambda: {a: blank() for a in ARMS})
    overall: dict[str, dict] = defaultdict(blank)

    with m8.open() as fh:
        for line in fh:
            values = line.rstrip("\n").split("\t")
            hit = dict(zip(FIELDS, values))
            query = hit["query"]
            fident, qcov = float(hit["fident"]), float(hit["qcov"])
            evalue, bits = float(hit["evalue"]), float(hit["bits"])
            arm = arm_of(hit["target"])
            for acc in (per_arm[query][arm], overall[query]):
                acc["n_hits"] += 1
                if evalue > HOMOLOGY_EVALUE:
                    continue
                acc["n_significant"] += 1
                if acc["best_bits"] is None or bits > acc["best_bits"]:
                    acc["best_bits"] = bits
                    acc["best_evalue"] = evalue
                    acc["best_target"] = hit["target"]
                if acc["best_identity_any"] is None or fident > acc["best_identity_any"]:
                    acc["best_identity_any"] = fident
                if qcov >= MIN_QCOV and (
                    acc["best_identity_covered"] is None
                    or fident > acc["best_identity_covered"]
                ):
                    acc["best_identity_covered"] = fident

    rows = []
    for key, info in sorted(meta.items()):
        total, arms = overall[key], per_arm[key]
        row = {
            "dataset": info["dataset"], "stem": info["stem"],
            "query_len": info["query_len"],
            "designed": int(is_designed(info["dataset"])),
            "n_hits": total["n_hits"],
            "n_hits_significant": total["n_significant"],
            "hits_censored": int(total["n_hits"] >= max_seqs),
            "best_identity_covered": total["best_identity_covered"],
            "best_identity_any": total["best_identity_any"],
            "best_evalue": total["best_evalue"],
            "best_bitscore": total["best_bits"],
            "best_target": total["best_target"],
            "best_arm": arm_of(total["best_target"]) if total["best_target"] else "",
            "stratum": identity_stratum(total["n_significant"],
                                        total["best_identity_covered"]),
        }
        for arm in ARMS:
            acc = arms[arm]
            row[f"{arm}_n_hits_significant"] = acc["n_significant"]
            row[f"{arm}_best_identity_covered"] = acc["best_identity_covered"]
            row[f"{arm}_best_identity_any"] = acc["best_identity_any"]
            row[f"{arm}_best_evalue"] = acc["best_evalue"]
            row[f"{arm}_best_target"] = acc["best_target"]
        row.update({c: info[c] for c in MANIFEST_STRATA})
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, nargs="+", default=DEFAULT_MANIFESTS)
    ap.add_argument("--exp41-foldbench", type=Path, default=EXP41_FOLDBENCH,
                    help="exp41 similarity CSV, to fill FoldBench fold_verdict")
    ap.add_argument("--query-fasta", type=Path, default=HERE / "data/eval_queries.fasta")
    ap.add_argument("--out", type=Path, default=HERE / "data/eval_train_identity.csv")
    ap.add_argument("-s", "--sensitivity", type=float, default=7.5)
    ap.add_argument("--max-seqs", type=int, default=2000,
                    help="prefilter hits kept per query; also the hit-count censor")
    ap.add_argument("--evalue", type=float, default=10.0,
                    help="mmseqs reporting threshold; significance is applied later "
                         f"at {HOMOLOGY_EVALUE:g}")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--split-memory-limit", default="200G")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                    help="training arms to search against (default: both). A "
                         "single arm cross-checks against exp94's AFDB-only run.")
    ap.add_argument("--skip-search", action="store_true",
                    help="reuse an existing aln.m8 and only rebuild the table")
    args = ap.parse_args()

    arms = tuple(args.arms)
    tag = "" if arms == ARMS else "only_" + "_".join(arms)
    n_queries = build_query_fasta(args.manifests, args.query_fasta)
    meta = read_manifest_meta(args.manifests, args.exp41_foldbench)
    print(f"[queries] {n_queries} eval proteins -> {args.query_fasta}", flush=True)
    print(f"[targets] arms: {', '.join(arms)}", flush=True)

    m8 = args.work / (f"aln_{tag}.m8" if tag else "aln.m8")
    if not args.skip_search:
        target_fasta = concat_targets(args.work, arms)
        m8 = search(args.work, args.query_fasta, target_fasta,
                    sensitivity=args.sensitivity, max_seqs=args.max_seqs,
                    evalue=args.evalue, threads=args.threads,
                    split_memory_limit=args.split_memory_limit, tag=tag)
    elif not m8.exists():
        raise SystemExit(f"--skip-search but {m8} does not exist")

    rows = reduce_alignments(m8, meta, args.max_seqs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table] {len(rows)} rows -> {args.out}", flush=True)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["stratum"]] += 1
    print("[strata]", dict(counts), flush=True)
    for arm in ARMS:
        n = sum(1 for r in rows if r[f"{arm}_n_hits_significant"] > 0)
        print(f"[{arm}] {n}/{len(rows)} eval proteins have a significant hit", flush=True)
    censored = sum(r["hits_censored"] for r in rows)
    if censored:
        print(f"[warn] {censored} proteins hit the --max-seqs {args.max_seqs} cap; "
              "their n_hits columns are censored (best-hit columns are not)", flush=True)

    provenance = args.out.with_suffix(".provenance.json")
    provenance.write_text(json.dumps({
        "sensitivity": args.sensitivity, "max_seqs": args.max_seqs,
        "reporting_evalue": args.evalue, "homology_evalue": HOMOLOGY_EVALUE,
        "min_qcov": MIN_QCOV, "mmseqs": ensure_mmseqs(),
        "train_stats": json.loads((args.work / "train_sequences_stats.json").read_text())
        if (args.work / "train_sequences_stats.json").exists() else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
