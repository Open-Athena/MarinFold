# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot contact-set count under <contacts-v1> vs <contacts-v1.multi>, per checkpoint.

H2 predicted that the plain-mode leak (#163's arm F emitted ~2.94 sections under
the PLAIN sentinel after 405 steps) closes with optimization, and that #175's
completely clean switch at 2,070 steps is the same mechanism given ~5x the
training.  This is the measurement that decides it, and it needs BOTH curves:

* plain falling to 1.0 alone would be consistent with the model having simply
  lost the multi-draft format;
* multi staying high alone says nothing about the leak.

Only the two together show a *mode switch* rather than a mode collapse, which is
why they are drawn on one axis.

Counts are ``n_sections_raw`` -- uncapped.  ``n_sections`` is clipped to
``--max-sections`` when the worker decodes, and a clipped count can only
understate a leak.

    python plot_leak_curve.py --curve ~/exp230_data/eval/curve --out plots/
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load(curve: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(curve.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"(base|step-(\d+))-(plain|multi)$", d.name)
        if not m:
            continue
        step = 0 if m.group(1) == "base" else int(m.group(2))
        mode = m.group(3)
        # eval_modes_worker nests its output under a {mode} subdirectory, so
        # this has to recurse -- a flat glob silently finds nothing and the
        # checkpoint is skipped without an error.
        parts = sorted(glob.glob(str(d / "**" / "*.parquet"), recursive=True))
        if not parts:
            continue
        df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
        rows.append(dict(
            step=step, mode=mode, label=m.group(1), n_rollouts=len(df),
            mean_sections=float(df["n_sections_raw"].mean()),
            median_sections=float(df["n_sections_raw"].median()),
            p90_sections=float(df["n_sections_raw"].quantile(0.90)),
            max_sections=int(df["n_sections_raw"].max()),
            frac_single=float((df["n_sections_raw"] == 1).mean()),
            # SE over rollouts; the curve is paired across checkpoints by subset,
            # so this is the honest spread for a single point.
            se_sections=float(df["n_sections_raw"].std(ddof=1) / np.sqrt(len(df))),
            frac_finished=float(df["finished"].mean()),
            mean_best_f1=float(df["best_f1"].mean()),
            mean_last_f1=float(df["last_f1"].mean()),
        ))
    return pd.DataFrame(rows).sort_values(["mode", "step"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    df = load(a.curve)
    if df.empty:
        raise SystemExit(f"no curve output under {a.curve}")
    missing = {"plain", "multi"} - set(df["mode"])
    if missing:
        raise SystemExit(f"curve is missing mode(s) {missing} -- both are needed: "
                         "plain falling alone cannot distinguish a closed leak "
                         "from a lost format")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "leak_curve.csv", index=False)
    print(df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for mode, colour, marker in (("plain", "#c1121f", "o"), ("multi", "#0353a4", "s")):
        d = df[df["mode"] == mode]
        if d.empty:
            continue
        lbl = f"<contacts-v1{'.multi' if mode == 'multi' else ''}>"
        ax.errorbar(d["step"], d["mean_sections"], yerr=d["se_sections"],
                    color=colour, marker=marker, capsize=3, lw=1.8, ms=5, label=lbl)
        ax2.plot(d["step"], 100 * d["frac_single"], color=colour, marker=marker,
                 lw=1.8, ms=5, label=lbl)
        # Label every point: on a linear axis the plain curve is pinned to the
        # floor and its shape is unreadable without the numbers.
        for x, y in zip(d["step"], d["mean_sections"]):
            ax.annotate(f"{y:.2f}" if y < 10 else f"{y:.1f}",
                        xy=(x, y), textcoords="offset points",
                        xytext=(0, 9 if mode == "multi" else -15),
                        ha="center", fontsize=7.5, color=colour)

    ax.axhline(1.0, color="0.55", ls=":", lw=1)
    ax.axhline(2.94, color="#e07a5f", ls="-.", lw=1)
    ax.annotate("#163 arm F leak (2.94, 405 steps)", xy=(0.30, 2.94),
                xycoords=("axes fraction", "data"), fontsize=8,
                color="#e07a5f", va="bottom")
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("mean contact sets per rollout")
    ax.set_title("Contact sets emitted, by mode")
    ax.legend(fontsize=9, frameon=False, loc="center right")
    # A small negative floor is deliberate: both curves sit at 1.0 at step 0 and
    # the plain curve never leaves it, so its point labels need clear space BELOW
    # the line. With ylim starting at 0 they collide with the x-axis.
    ax.set_ylim(-2.0, max(26, float(df["mean_sections"].max()) * 1.12))
    ax.spines[["top", "right"]].set_visible(False)

    ax2.axhline(95, color="0.4", ls=":", lw=1)
    ax2.annotate("Gate B: 95%", xy=(0.55, 95), xycoords=("axes fraction", "data"),
                 fontsize=8, color="0.35", va="bottom")
    ax2.set_xlabel("fine-tuning step")
    ax2.set_ylabel("% of rollouts emitting exactly one set")
    ax2.set_title("Single-set fraction")
    ax2.set_ylim(-3, 105)
    ax2.legend(fontsize=9, frameon=False, loc="center right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    png = a.out / "leak_curve.png"
    fig.savefig(png, dpi=170)
    print(f"wrote {png} and {a.out / 'leak_curve.csv'}")

    meta = {
        "caption": ("Contact sets emitted per rollout under the plain "
                    "<contacts-v1> sentinel versus <contacts-v1.multi>, across the "
                    "fine-tuning run. Step 0 is the exp199 base, read for BOTH modes: "
                    "token id 7 is renamed in place, so the base sees the same integer "
                    "and simply has no multi-draft behaviour attached to it yet. Left: "
                    "mean count, linear axis, every point labelled, with #163's arm F "
                    "leak at 2.94 for reference. Right: fraction of rollouts emitting "
                    "exactly one set, against Gate B's 95% bar. Counts are uncapped "
                    "(n_sections_raw). 200 proteins, seeded and identical across "
                    "checkpoints; 4 rollouts each."),
        "script": "plot_leak_curve.py",
    }
    (a.out / "leak_curve.png.meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
