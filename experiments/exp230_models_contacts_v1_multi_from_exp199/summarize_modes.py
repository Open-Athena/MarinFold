# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Turn ``eval_modes_worker`` output into the Gate B table and the RL hand-off report.

Gate B is a single number with a threshold, so it is reported as pass/fail:

    plain-mode mean sections <= 1.05  AND  >= 95% of rollouts emit exactly one

#163's arm F read **2.94** on the first of those.  Anything materially above 1
means the multi-draft habit is still leaking into plain generation, which is the
defect this experiment exists to fix.

The multi arm is reported, not gated, because best-of-N is an oracle metric --
but two of its numbers ARE kill criteria borrowed from #200: mean pairwise
Jaccard above 0.30 is diversity collapse, and sections shorter than 60% of
baseline is reward-hacking-toward-silence in the SFT stage.

Paired statistics are per (target_id, r) so plain and multi are compared on
identical proteins and rollout indices.

    uv run python summarize_modes.py --rollouts gs://.../eval/step-2500 \\
        --label step-2500 --out data/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

#: Gate B, from the issue's success criteria.
MAX_MEAN_SECTIONS = 1.05
MIN_SINGLE_SECTION_FRACTION = 0.95
#: #200's diversity-collapse kill criterion.
MAX_MEAN_JACCARD = 0.30


def load(uri: str) -> pd.DataFrame:
    import fsspec

    fs = fsspec.core.url_to_fs(uri)[0]
    frames = []
    for mode in ("plain", "multi"):
        pattern = f"{uri.rstrip('/')}/{mode}/*.parquet"
        try:
            paths = fs.glob(pattern)
        except FileNotFoundError:
            paths = []
        for p in paths:
            full = p if "://" in p else f"{uri.split('://')[0]}://{p}"
            with fs.open(full, "rb") as fh:
                frames.append(pq.read_table(fh).to_pandas())
    if not frames:
        raise SystemExit(f"no rollout parquets under {uri}")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    df = load(a.rollouts)
    a.out.mkdir(parents=True, exist_ok=True)

    per_mode = (df.groupby("mode")
                  .agg(n_rollouts=("r", "size"),
                       n_proteins=("target_id", "nunique"),
                       mean_sections=("n_sections", "mean"),
                       mean_sections_raw=("n_sections_raw", "mean"),
                       max_sections=("n_sections", "max"),
                       frac_single=("n_sections", lambda s: float((s == 1).mean())),
                       finished=("finished", "mean"),
                       mean_jaccard=("mean_jaccard", "mean"),
                       first_f1=("first_f1", "mean"),
                       best_f1=("best_f1", "mean"),
                       last_f1=("last_f1", "mean"),
                       tokens=("n_gen_tokens", "mean"))
                  .reset_index())
    per_mode.insert(0, "label", a.label)
    per_mode.to_csv(a.out / f"modes_{a.label}.csv", index=False)
    print(per_mode.to_string(index=False))

    plain = df[df["mode"] == "plain"]
    verdict = {"label": a.label}
    if len(plain):
        mean_sections = float(plain["n_sections"].mean())
        frac_single = float((plain["n_sections"] == 1).mean())
        verdict.update({
            "plain_mean_sections": mean_sections,
            "plain_frac_single_section": frac_single,
            "gate_b_pass": bool(mean_sections <= MAX_MEAN_SECTIONS
                                and frac_single >= MIN_SINGLE_SECTION_FRACTION),
        })
        print(f"\nGATE B  mean sections {mean_sections:.3f} (<= {MAX_MEAN_SECTIONS}) - "
              f"single-section {frac_single:.1%} (>= {MIN_SINGLE_SECTION_FRACTION:.0%}) - "
              f"{'PASS' if verdict['gate_b_pass'] else 'FAIL'}")
        print("        #163 arm F read 2.94 mean sections here.")

    multi = df[df["mode"] == "multi"]
    if len(multi):
        jac = float(multi["mean_jaccard"].mean())
        # Paired on (protein, rollout): the spread best-of-N is paid for by.
        gap = float((multi["best_f1"] - multi["last_f1"]).mean())
        se = float((multi["best_f1"] - multi["last_f1"]).std(ddof=1) / np.sqrt(len(multi)))
        verdict.update({
            "multi_mean_sections": float(multi["n_sections"].mean()),
            "multi_mean_jaccard": jac,
            "multi_best_minus_last": gap,
            "multi_best_minus_last_se": se,
            "diversity_collapse": bool(jac > MAX_MEAN_JACCARD),
        })
        print(f"\nMULTI   sections {multi['n_sections'].mean():.2f} - "
              f"jaccard {jac:.3f} ({'COLLAPSE' if jac > MAX_MEAN_JACCARD else 'ok'}, "
              f"#163 arm F 0.071) - best-last {gap:+.4f} +/- {se:.4f}")

    (a.out / f"modes_{a.label}.verdict.json").write_text(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
