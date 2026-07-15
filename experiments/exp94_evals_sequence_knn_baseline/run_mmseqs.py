# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 — mmseqs search of the 554 eval queries vs the train index.

Builds ``data/eval_queries.fasta`` from the two eval manifests (header =
``{dataset}__{stem}`` — keyed on *both* because ``7ur7_A``/``8ah9_A`` recur across
foldbench and exp65 with different sequences), then searches it against
``_scratch/train_seqs.fasta`` (Step 1). The alignment must carry per-residue
columns, so two pitfalls are non-negotiable:

* ``-a`` (== ``--alignment-mode 3``) on ``search`` — without it ``convertalis``
  returns empty ``qaln``/``taln`` and every contact maps to nothing.
* ``qstart,qend,tstart,tend,qaln,taln`` in the ``--format-output`` — Step 3 needs
  them to walk the alignment.

Outputs ``_scratch/aln.m8``. Re-runs are idempotent (stale result DBs/tmp are
cleared first; mmseqs refuses to overwrite them otherwise).

    uv run python run_mmseqs.py --scratch _scratch -s 7.5
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from knn_lib import ensure_mmseqs, run

EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts/data")
DEFAULT_MANIFESTS = [EXP78 / "eval_manifest_foldbench.csv", EXP78 / "eval_manifest_exp65.csv"]
FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits,qstart,qend,tstart,tend,qaln,taln"


def build_query_fasta(manifests: list[Path], out: Path) -> int:
    """One record per (dataset, stem); header `{dataset}__{stem}`, seq = input_seq."""
    seen: set[str] = set()
    lines: list[str] = []
    for man in manifests:
        with man.open() as fh:
            for r in csv.DictReader(fh):
                key = f"{r['dataset']}__{r['stem']}"
                if key in seen:
                    raise ValueError(f"duplicate query key {key} in {man}")
                seen.add(key)
                lines.append(f">{key}\n{r['input_seq'].strip().upper()}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, nargs="+", default=DEFAULT_MANIFESTS)
    ap.add_argument("--query-fasta", type=Path, default=Path("data/eval_queries.fasta"))
    ap.add_argument("-s", "--sensitivity", type=float, default=7.5)
    ap.add_argument("--max-seqs", type=int, default=300)
    ap.add_argument("--evalue", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()

    mmseqs = ensure_mmseqs()
    print(f"[mmseqs] binary: {mmseqs}", flush=True)

    n_q = build_query_fasta(args.manifests, args.query_fasta)
    print(f"[mmseqs] {n_q} eval queries -> {args.query_fasta}", flush=True)

    train_fasta = args.scratch / "train_seqs.fasta"
    if not train_fasta.exists():
        raise SystemExit(f"missing {train_fasta}; run build_train_index.py first")

    work = args.scratch
    query_db, train_db, aln_db = work / "queryDB", work / "trainDB", work / "alnDB"
    tmp = work / "tmp"
    # Idempotency: mmseqs won't overwrite an existing result DB or search tmp.
    if tmp.exists():
        shutil.rmtree(tmp)
    for stale in work.glob("alnDB*"):
        stale.unlink()

    run([mmseqs, "createdb", str(args.query_fasta), str(query_db)])
    if not (work / "trainDB").exists():
        run([mmseqs, "createdb", str(train_fasta), str(train_db)])
    else:
        print("[mmseqs] reusing existing trainDB", flush=True)
    run([mmseqs, "search", str(query_db), str(train_db), str(aln_db), str(tmp),
         "-s", str(args.sensitivity), "--max-seqs", str(args.max_seqs),
         "-e", str(args.evalue), "-a", "--threads", str(args.threads)])
    m8 = work / "aln.m8"
    run([mmseqs, "convertalis", str(query_db), str(train_db), str(aln_db), str(m8),
         "--format-output", FORMAT, "--threads", str(args.threads)])

    n_lines = sum(1 for _ in m8.open())
    print(f"[mmseqs] wrote {n_lines:,} alignment rows -> {m8}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
