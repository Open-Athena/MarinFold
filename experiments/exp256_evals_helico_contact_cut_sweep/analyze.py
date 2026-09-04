# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""How many MarinFold contacts should Helico be given? The lDDT-versus-cut curve.

Reads the Helico runs for issue #256 and produces the curve plus its paired
statistics. The folding itself happens in the helico repo -- `modal/bench_byclass.py`
against `experiments/exp14_foldbench_held_out_monomers`, whose targets, ground
truths, index map and model settings this reuses unchanged so the new cuts land
on the same curve as exp14's published ones. Only the analysis lives here.

Three cuts (`L`, `L/2`, `L/5`) come from exp14 and are not re-run; five
(`1.5L`, `2L`, `3L`, `5L`, `union`) are this experiment's. `off`, `oracle`,
`v2ss` and `v2msa` are exp14's reference arms, carried over for context.

**Everything is eval-val.** eval-test is not read.

The comparison that matters is paired: every arm folds the same 96 targets with
the same weights and the same seed, differing only in the contact list, so the
per-target difference against `mf_L` is the statistic — a per-arm interval on
the means is roughly twice as wide and is not the right one.

    uv run python analyze.py --helico <helico-worktree> --out data
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_SET = "eval-val"
#: Arm -> how many pairs it hands the model, as a multiple of L. `union` is not
#: a fixed multiple; its measured value goes in from the accuracy table.
CUT_ORDER = ("mf_L5", "mf_L2", "mf_L", "mf_1p5L", "mf_2L", "mf_3L", "mf_5L",
             "mf_union")
CUT_MULTIPLE = {"mf_L5": 0.2, "mf_L2": 0.5, "mf_L": 1.0, "mf_1p5L": 1.5,
                "mf_2L": 2.0, "mf_3L": 3.0, "mf_5L": 5.0}
REFERENCE_ARMS = ("off", "v2ss", "oracle", "v2msa")
REFERENCE_LABEL = {
    "off": "Helico, no contacts",
    "v2ss": "Protenix-v2 single sequence",
    "oracle": "Helico + oracle contacts",
    "v2msa": "Protenix-v2 + MSA",
}
BASELINE = "mf_L"
BOOTSTRAP_DRAWS = 10_000
SEED = 256


def load_arm(results: Path, tag: str) -> pd.DataFrame:
    """One arm's per-target lDDT on eval-val."""
    path = results / f"{tag}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["target_id", "lddt"])
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if r.get("dataset") == EVAL_SET and r["status"] == "ok" and r["lddt"]]
    return pd.DataFrame({"target_id": [r["target_id"] for r in rows],
                         "lddt": [float(r["lddt"]) for r in rows],
                         "arm": tag})


def paired(wide: pd.DataFrame, a: str, b: str) -> dict:
    """Mean per-target ``a - b`` with a 95 % bootstrap interval."""
    delta = (wide[a] - wide[b]).dropna().to_numpy()
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    means = delta[index].mean(axis=1)
    return dict(arm=a, versus=b, n=len(delta), mean_delta=float(delta.mean()),
                lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)),
                targets_better=float((delta > 0).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--helico", type=Path, required=True,
                    help="helico worktree holding the #256 runs")
    ap.add_argument("--exp14", type=Path, required=True,
                    help="helico worktree holding exp14's published arms")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    exp14_results = args.exp14 / "experiments/exp14_foldbench_held_out_monomers/results"
    new_results = args.helico / "experiments/exp14_foldbench_held_out_monomers/results"

    frames = []
    provenance = {}
    for tag in CUT_ORDER + REFERENCE_ARMS:
        # This experiment's arms take precedence; anything it did not run comes
        # from exp14 unchanged, so a cut is never silently re-derived.
        source = new_results if (new_results / f"{tag}.csv").exists() else exp14_results
        frame = load_arm(source, tag)
        if frame.empty:
            print(f"[cut-sweep] MISSING {tag}")
            continue
        provenance[tag] = "exp256" if source == new_results else "exp14"
        frames.append(frame)
    per_target = pd.concat(frames, ignore_index=True)

    # Contact accuracy of each cut, so the curve can be read against the
    # operating point rather than against k.
    accuracy = pd.read_csv(
        args.helico / "experiments/exp14_foldbench_held_out_monomers"
        / "data/cut_sweep_accuracy.csv")
    exp14_accuracy = pd.read_csv(
        args.exp14 / "experiments/exp14_foldbench_held_out_monomers"
        / "data/marinfold_arm_accuracy.csv")
    exp14_accuracy = exp14_accuracy[exp14_accuracy.eval_set == EVAL_SET]

    stats = {}
    for tag in CUT_ORDER:
        label = tag.removeprefix("mf_")
        if f"precision_{label}" in accuracy.columns:
            stats[tag] = dict(
                precision=float(accuracy[f"precision_{label}"].mean()),
                recall=float(accuracy[f"recall_{label}"].mean()),
                pairs_over_L=float((accuracy[f"n_{label}"] / accuracy["L"]).mean()))
        elif f"precision_{label}" in exp14_accuracy.columns:
            stats[tag] = dict(
                precision=float(exp14_accuracy[f"precision_{label}"].mean()),
                recall=float("nan"),
                pairs_over_L=CUT_MULTIPLE.get(tag, float("nan")))

    # Wide table over the targets every arm folded, so all comparisons are on
    # the same set rather than each on its own.
    wide = per_target.pivot_table(index="target_id", columns="arm", values="lddt")
    common = wide.dropna()
    print(f"[cut-sweep] {len(wide)} targets seen, {len(common)} folded by every arm")

    curve = []
    for tag in CUT_ORDER:
        if tag not in common.columns:
            continue
        row = dict(arm=tag, source=provenance[tag], n=len(common),
                   lddt=float(common[tag].mean()), **stats.get(tag, {}))
        if tag != BASELINE:
            row.update({f"vs_{BASELINE}_{k}": v
                        for k, v in paired(common, tag, BASELINE).items()
                        if k in ("mean_delta", "lo", "hi", "targets_better")})
        curve.append(row)
    curve_frame = pd.DataFrame(curve)
    curve_frame.to_csv(args.out / "cut_sweep_curve.csv", index=False)

    references = [dict(arm=tag, label=REFERENCE_LABEL[tag], source=provenance[tag],
                       n=len(common), lddt=float(common[tag].mean()))
                  for tag in REFERENCE_ARMS if tag in common.columns]
    pd.DataFrame(references).to_csv(args.out / "reference_arms.csv", index=False)

    per_target.to_csv(args.out / "per_target_lddt.csv.gz", index=False)
    (args.out / "provenance.json").write_text(
        json.dumps({"eval_set": EVAL_SET, "n_common": len(common),
                    "arm_source": provenance}, indent=2) + "\n")

    print("\n[cut-sweep] lDDT versus contact cut on eval-val:")
    print(curve_frame.round(4).to_string(index=False))
    print("\n[cut-sweep] reference arms:")
    print(pd.DataFrame(references).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
