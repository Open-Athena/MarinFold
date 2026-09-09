# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool the ESMFold2 refold shards: design arm against the native control.

The design pass rate is only interpretable next to the native one. Native
sequences refolded onto their OWN AFDB backbones are the ceiling this
measurement can reach, and it is far below 100% because ESMFold2 at 1 sample
/ 100 steps against a whole-chain 2 A gate is a strict test — not because
AFDB backbones are wrong. Report the ratio, not the absolute.

**The ratio is computed on backbones present in both arms.** An earlier
version pooled every design refold against every native refold without joining
on `entry_id`, and the README described the result as "paired" when it was
not: the design arm covers 375 backbones and the native arm 250, so 125 design
backbones had no control and the two rates were measured over different
protein sets. It moved the headline scTM ratio from 0.907 to 0.872. Design
refolds outside the shared set are still reported, separately and labelled as
the design-only sample.

Uncertainty is a **bootstrap over backbones, not over refolds**. The eight
designs of one backbone share its geometry and are nowhere near independent —
resampling refolds would treat 2,000 correlated observations as 2,000
independent ones and produce an interval several times too narrow. Resampling
backbones keeps each backbone's designs together, which is the unit that was
actually sampled.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import random
import statistics as st
from pathlib import Path

import pyarrow.parquet as pq

PASSES = (("sc_rmsd", 2.0, False, "scRMSD<2A"), ("sc_tm", 0.5, True, "scTM>0.5"))


def load(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        rows += pq.read_table(path).to_pylist()
    return rows


def rate(rows: list[dict], key: str, thr: float, greater: bool) -> float:
    if not rows:
        return float("nan")
    return sum((r[key] > thr) if greater else (r[key] < thr) for r in rows) / len(rows)


def by_entry(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        out[r["entry_id"]].append(r)
    return dict(out)


def bootstrap_ratio(design: dict[str, list[dict]], native: dict[str, list[dict]],
                    key: str, thr: float, greater: bool, *,
                    draws: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for design_rate / native_rate, resampling backbones.

    Each draw takes a backbone and carries *all* of its design refolds plus its
    native refold, so sibling designs stay together and the correlation between
    them is inside the interval rather than ignored by it.
    """
    keys = sorted(set(design) & set(native))
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        picks = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        d = [r for k in picks for r in design[k]]
        n = [r for k in picks for r in native[k]]
        nr = rate(n, key, thr, greater)
        if nr:
            ratios.append(rate(d, key, thr, greater) / nr)
    if not ratios:
        return (float("nan"), float("nan"))
    ratios.sort()
    return (ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))])


def summarise(rows: list[dict]) -> dict:
    return {
        "n_refolds": len(rows),
        "n_backbones": len({r["entry_id"] for r in rows}),
        "sc_rmsd_pass_pct": round(100 * rate(rows, "sc_rmsd", 2.0, False), 1),
        "sc_tm_pass_pct": round(100 * rate(rows, "sc_tm", 0.5, True), 1),
        "median_sc_rmsd": round(st.median(r["sc_rmsd"] for r in rows), 2),
        "median_sc_tm": round(st.median(r["sc_tm"] for r in rows), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", default="/data/exp266/design_refold-*.parquet")
    ap.add_argument("--native", default="/data/exp266/native_refold-*.parquet")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--draws", type=int, default=10_000)
    args = ap.parse_args()

    d, n = load(args.design), load(args.native)
    dby, nby = by_entry(d), by_entry(n)
    shared = sorted(set(dby) & set(nby))
    dm = [r for k in shared for r in dby[k]]

    print(f"design arm : {len(d):,} refolds over {len(dby)} backbones")
    print(f"native ctrl: {len(n):,} refolds over {len(nby)} backbones")
    print(f"shared     : {len(shared)} backbones "
          f"({len(set(dby) - set(nby))} design-only, {len(set(nby) - set(dby))} native-only)")

    print(f"\nMATCHED comparison — the {len(shared)} backbones present in both arms")
    print(f"{'arm':<16}{'n':>7}{'scRMSD<2A':>11}{'scTM>0.5':>10}{'med RMSD':>10}{'med TM':>8}")
    for label, rows in (("design", dm), ("native control", n)):
        s = summarise(rows)
        print(f"{label:<16}{s['n_refolds']:>7,}{s['sc_rmsd_pass_pct']:>10.1f}%"
              f"{s['sc_tm_pass_pct']:>9.1f}%{s['median_sc_rmsd']:>10.2f}{s['median_sc_tm']:>8.3f}")

    rows_out = []
    for key, thr, greater, label in PASSES:
        r = rate(dm, key, thr, greater) / rate(n, key, thr, greater)
        lo, hi = bootstrap_ratio(dby, nby, key, thr, greater, draws=args.draws)
        print(f"  design/native {label:<10} ratio {r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        rows_out.append({"metric": label, "ratio": round(r, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "n_backbones": len(shared)})

    print(f"\nDESIGN-ONLY sample (all {len(dby)} design backbones, no control)")
    s = summarise(d)
    print(f"  {s['n_refolds']:,} refolds  scRMSD<2A {s['sc_rmsd_pass_pct']}%  "
          f"scTM>0.5 {s['sc_tm_pass_pct']}%")
    print("  Not comparable to the native arm: different protein set.")

    best = sum(any(x["sc_rmsd"] < 2.0 for x in v) for v in dby.values()) / len(dby)
    best_m = sum(any(x["sc_rmsd"] < 2.0 for x in dby[k]) for k in shared) / len(shared)
    print(f"\nper-backbone designability (any of 8): "
          f"{best:.1%} over all {len(dby)}, {best_m:.1%} over the shared {len(shared)}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "refold_selfconsistency.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["arm", "n_refolds", "n_backbones", "sc_rmsd_pass_pct",
                                "sc_tm_pass_pct", "median_sc_rmsd", "median_sc_tm"])
        w.writeheader()
        w.writerow({"arm": "design_matched", **summarise(dm)})
        w.writerow({"arm": "native_control", **summarise(n)})
        w.writerow({"arm": "design_all", **summarise(d)})
    with (args.out / "refold_ratio_ci.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["metric", "ratio", "ci_lo", "ci_hi", "n_backbones"])
        w.writeheader()
        w.writerows(rows_out)

    # Per-temperature and per-length cuts, both on the matched set.
    with (args.out / "refold_by_temperature.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["mpnn_temperature", "n", "sc_rmsd_pass_pct", "sc_tm_pass_pct"])
        w.writeheader()
        bt = collections.defaultdict(list)
        for r in dm:
            bt[r["mpnn_temperature"]].append(r)
        for t in sorted(bt):
            w.writerow({"mpnn_temperature": t, "n": len(bt[t]),
                        "sc_rmsd_pass_pct": round(100 * rate(bt[t], "sc_rmsd", 2.0, False), 1),
                        "sc_tm_pass_pct": round(100 * rate(bt[t], "sc_tm", 0.5, True), 1)})

    def binof(length: int) -> str:
        for lo, hi in ((0, 100), (100, 200), (200, 400), (400, 10 ** 6)):
            if lo <= length < hi:
                return f"{lo}-{hi if hi < 10 ** 6 else '+'}"
        return "?"

    with (args.out / "refold_by_length.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["length_bin", "n_design", "design_sc_tm_pass_pct",
                                "n_native", "native_sc_tm_pass_pct"])
        w.writeheader()
        bd, bn = collections.defaultdict(list), collections.defaultdict(list)
        for r in dm:
            bd[binof(r["seq_len"])].append(r)
        for r in n:
            bn[binof(r["seq_len"])].append(r)
        for k in sorted(set(bd) | set(bn), key=lambda x: int(x.split("-")[0])):
            w.writerow({"length_bin": k, "n_design": len(bd.get(k, [])),
                        "design_sc_tm_pass_pct": round(100 * rate(bd[k], "sc_tm", 0.5, True), 1) if bd.get(k) else "",
                        "n_native": len(bn.get(k, [])),
                        "native_sc_tm_pass_pct": round(100 * rate(bn[k], "sc_tm", 0.5, True), 1) if bn.get(k) else ""})
    print(f"\nwrote {args.out}/refold_selfconsistency.csv, refold_ratio_ci.csv, "
          f"refold_by_temperature.csv, refold_by_length.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
