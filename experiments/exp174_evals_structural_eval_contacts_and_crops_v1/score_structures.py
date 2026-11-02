# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step B — score predicted structures against the ground-truth bundle.

This is Component 2 of issue #174, and it is deliberately decoupled from
Component 1 (how a document becomes coordinates): its only input is a
directory of predicted structures in the ``canonical_pdb`` contract, laid out
the same way as the ground-truth bundle::

    <pred-dir>/<dataset>/<stem>.pdb

Any predictor — a MarinFold checkpoint under any of the ``PLANS.md``
inference regimes, a quantization baseline, ESMFold, whatever — is scored by
pointing this script at its output directory.

What it reports, per record and in aggregate:

* **Coverage** — ``atom_coverage`` / ``ca_coverage`` / ``residue_coverage``,
  and ``frac_refined_of_gt`` (atoms the predictor claims at fine resolution).
  Nothing else in the table is interpretable without these.
* **Coverage-penalized accuracy** — ``lddt_all``, ``lddt_ca``, ``tm_score``.
  Their denominators come from the ground truth, so an unplaced atom costs
  score. **These are the model-comparison numbers.**
* **Covered-only accuracy** — ``lddt_all_covered``, ``lddt_ca_covered``,
  ``rmsd_all``, ``rmsd_ca``. "How good is the part it did emit." A predictor
  that places three atoms perfectly gets ``rmsd_all = 0``; read these only
  next to the coverage columns.

A record with no prediction file is **not skipped** — it is scored as a total
miss (zero coverage, zero lDDT, zero TM-score) and marked ``status=missing``,
because dropping it would quietly inflate the mean over whatever the
predictor happened to finish.

Usage::

    uv run python score_structures.py \\
        --gt-dir _scratch/gt --pred-dir _scratch/pred/<name> \\
        --model-name <name> --out data/scores_<name>.csv
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import canonical_pdb
import usalign
from structure_metrics import score_prediction

# Length strata for the aggregate table. contacts-and-crops-v1's coverage
# falls off with chain length by construction (SPEC → Coverage), and #93's
# base-rate finding means length has to be held fixed before two predictors
# can be compared — so every headline number is also reported per bin.
LENGTH_BINS: tuple[tuple[str, int, int], ...] = (
    ("<=100", 0, 100),
    ("101-200", 101, 200),
    ("201-400", 201, 400),
    (">400", 401, 10**9),
)

# Metrics the aggregate table averages, in report order.
SUMMARY_METRICS: tuple[str, ...] = (
    "atom_coverage",
    "ca_coverage",
    "residue_coverage",
    "frac_refined_of_gt",
    "lddt_all",
    "lddt_ca",
    "tm_score",
    "lddt_all_covered",
    "lddt_ca_covered",
    "rmsd_all",
    "rmsd_ca",
)

# US-align needs at least three CA atoms in each file — below that it prints
# "Sequence is too short <3!" and dies on a signal. A prediction that thin has
# a TM-score of at most 3/L_ref against the reference anyway, so we score it 0
# and record why, rather than letting a crash take down the whole run. This is
# reachable: a heavily truncated document can leave a protein with one or two
# CA atoms placed.
_TM_MIN_CA = 3

# What a record scores when the predictor produced nothing for it. Zero rather
# than NaN for the coverage-penalized metrics: "no structure" really is a
# TM-score of 0 against the reference, and averaging it in is the honest
# accounting. The covered-only metrics stay NaN — there is no covered set.
_EMPTY_METRICS = {
    "atom_coverage": 0.0,
    "ca_coverage": 0.0,
    "residue_coverage": 0.0,
    "frac_refined_of_covered": float("nan"),
    "frac_refined_of_gt": 0.0,
    "lddt_all": 0.0,
    "lddt_ca": 0.0,
    "lddt_all_refined": float("nan"),
    "lddt_all_covered": float("nan"),
    "lddt_ca_covered": float("nan"),
    "rmsd_all": float("nan"),
    "rmsd_ca": float("nan"),
    "tm_score": 0.0,
    "tm_score_pred_normalized": float("nan"),
    "n_tm_aligned": 0.0,
    "tm_below_min_ca": True,
    "n_pred_atoms": 0.0,
    "n_pred_extra": 0.0,
    "n_covered_atoms": 0.0,
    "n_covered_ca": 0.0,
}


@dataclass(frozen=True)
class Job:
    """One record to score."""

    record: dict
    gt_path: Path
    pred_path: Path
    refined_max_sigma: float
    binary: Path


def length_bin(length: int) -> str:
    """Name of the length stratum ``length`` falls in."""
    for name, lo, hi in LENGTH_BINS:
        if lo <= length <= hi:
            return name
    raise ValueError(f"length {length} fell outside every bin")


def score_one(job: Job) -> dict:
    """Score a single record; never raises for a missing/empty prediction."""
    started = time.time()
    row = {
        "record_id": job.record["record_id"],
        "stem": job.record["stem"],
        "dataset": job.record["dataset"],
        "L": job.record["L"],
        "length_bin": length_bin(int(job.record["L"])),
        "n_gt_atoms": float(job.record["n_gt_atoms"]),
        "n_gt_ca": float(job.record["n_gt_ca"]),
        "n_gt_residues": float(job.record["n_gt_residues"]),
    }
    if not job.pred_path.exists():
        row.update(_EMPTY_METRICS)
        row["status"] = "missing"
        row["score_seconds"] = time.time() - started
        return row

    gt = canonical_pdb.read_structure(job.gt_path)
    pred = canonical_pdb.read_structure(job.pred_path)
    if len(pred) == 0:
        row.update(_EMPTY_METRICS)
        row["status"] = "empty"
        row["score_seconds"] = time.time() - started
        return row

    row.update(
        score_prediction(gt, pred, refined_max_sigma=job.refined_max_sigma)
    )
    if row["n_covered_ca"] < _TM_MIN_CA:
        row["tm_score"] = 0.0
        row["tm_score_pred_normalized"] = float("nan")
        row["n_tm_aligned"] = row["n_covered_ca"]
        row["tm_below_min_ca"] = True
    else:
        tm = usalign.tm_score(job.pred_path, job.gt_path, binary=job.binary)
        row["tm_score"] = tm.tm_score
        row["tm_score_pred_normalized"] = tm.tm_score_pred_normalized
        row["n_tm_aligned"] = float(tm.n_aligned)
        row["tm_below_min_ca"] = False
    row["status"] = "ok"
    row["score_seconds"] = time.time() - started
    return row


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate table: one row per (stratum, length bin) plus the overalls."""
    frames = []

    def block(label: str, group: pd.DataFrame) -> dict:
        out = {"stratum": label, "n": int(len(group))}
        out["n_missing"] = int((group["status"] != "ok").sum())
        for metric in SUMMARY_METRICS:
            out[f"mean_{metric}"] = float(group[metric].mean())
        # RMSD distributions have a long right tail; the median says more.
        for metric in ("rmsd_all", "rmsd_ca"):
            out[f"median_{metric}"] = float(group[metric].median())
        return out

    frames.append(block("all", scores))
    for name, _, _ in LENGTH_BINS:
        group = scores[scores["length_bin"] == name]
        if len(group):
            frames.append(block(f"length {name}", group))
    for dataset in sorted(scores["dataset"].unique()):
        frames.append(block(f"dataset {dataset}", scores[scores["dataset"] == dataset]))
    return pd.DataFrame(frames)


def provenance(model_name: str, binary: Path, pred_dir: Path) -> dict:
    """Everything needed to trace a results CSV back to what produced it."""
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    return {
        "model_nickname": model_name,
        "pred_dir": str(pred_dir),
        "usalign_version": usalign.binary_version(binary),
        "git_sha": git_sha,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-dir", type=Path, required=True, help="prepare_gt_structures.py output")
    ap.add_argument("--pred-dir", type=Path, required=True, help="<dataset>/<stem>.pdb tree")
    ap.add_argument("--model-name", required=True, help="nickname for the results CSV")
    ap.add_argument("--out", type=Path, required=True, help="per-record CSV")
    ap.add_argument("--summary", type=Path, default=None, help="aggregate CSV (default: <out>.summary.csv)")
    ap.add_argument(
        "--refined-max-sigma",
        type=float,
        default=1.0,
        help="B-factor (predicted positional sigma, Å) at or below which an "
        "atom counts as refined rather than coarse-box-only",
    )
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--usalign", type=Path, default=usalign.DEFAULT_BINARY)
    args = ap.parse_args(argv)

    binary = usalign.require_binary(args.usalign)
    index_path = args.gt_dir / "gt_index.jsonl"
    records = [json.loads(line) for line in index_path.open()]
    if args.limit is not None:
        records = records[: args.limit]

    jobs = [
        Job(
            record=record,
            gt_path=args.gt_dir / "gt_structures" / record["dataset"] / f"{record['stem']}.pdb",
            pred_path=args.pred_dir / record["dataset"] / f"{record['stem']}.pdb",
            refined_max_sigma=args.refined_max_sigma,
            binary=binary,
        )
        for record in records
    ]

    started = time.time()
    rows: list[dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for i, row in enumerate(pool.map(score_one, jobs), start=1):
                rows.append(row)
                if i % 50 == 0:
                    print(f"  ...{i}/{len(jobs)} scored ({time.time() - started:.0f}s)", flush=True)
    else:
        for i, job in enumerate(jobs, start=1):
            rows.append(score_one(job))
            if i % 50 == 0:
                print(f"  ...{i}/{len(jobs)} scored ({time.time() - started:.0f}s)", flush=True)

    scores = pd.DataFrame(rows)
    for key, value in provenance(args.model_name, binary, args.pred_dir).items():
        scores[key] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out, index=False)

    summary = summarize(scores)
    summary.insert(0, "model_nickname", args.model_name)
    summary_path = args.summary or args.out.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    n_ok = int((scores["status"] == "ok").sum())
    print(
        f"[score] {n_ok}/{len(scores)} predictions present "
        f"({len(scores) - n_ok} missing/empty) in {time.time() - started:.0f}s"
    )
    print(f"[score] per-record -> {args.out}")
    print(f"[score] aggregate  -> {summary_path}")
    columns = [
        "stratum",
        "n",
        "mean_atom_coverage",
        "mean_lddt_all",
        "mean_lddt_ca",
        "mean_tm_score",
        "mean_rmsd_ca",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
