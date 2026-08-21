# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The lDDT-versus-contact-cut curve, and what the extra contacts actually buy.

Left: folding accuracy against how many contacts Helico was handed, from top-L/5
out to every pair any rollout proposed. The reference arms bracket it -- no
contacts at the bottom, oracle contacts and Protenix-v2-with-MSA at the top --
so the whole curve can be read against the range it lives in, which is narrow.

Right: the same points against the *precision* of the list rather than its
length, with recall annotated. This is the panel that says why the curve turns:
lDDT tracks precision and ignores the recall the extra contacts buy.

    uv run python plot_results.py --data data --out plots
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LABEL = {"mf_L5": "L/5", "mf_L2": "L/2", "mf_L": "L", "mf_1p5L": "1.5L",
         "mf_2L": "2L", "mf_3L": "3L", "mf_5L": "5L", "mf_union": "union\n(14L)"}
CURVE_COLOR = "#d55e00"
BASELINE_COLOR = "#8f8b86"


def stamp(path: Path, sources: dict[str, Path], caption: str) -> None:
    meta = {
        "script": Path(sys.argv[0]).name, "args": sys.argv[1:], "caption": caption,
        "plot": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {name: {"path": str(source.resolve().relative_to(REPO)),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                    for name, source in sources.items()},
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")


def curve_panel(axis, curve: pd.DataFrame, references: pd.DataFrame) -> None:
    positions = range(len(curve))
    for _, row in references.iterrows():
        axis.axhline(row["lddt"], color=BASELINE_COLOR, linewidth=1.0,
                     linestyle="--", zorder=1)
        axis.text(len(curve) - 0.4, row["lddt"], f"  {row['label']} {row['lddt']:.3f}",
                  fontsize=8, color="#33312e", va="center")
    axis.plot(positions, curve["lddt"], color=CURVE_COLOR, marker="o", markersize=7,
              linewidth=2.2, zorder=3)
    best = curve["lddt"].idxmax()
    axis.scatter([list(curve.index).index(best)], [curve.loc[best, "lddt"]], s=170,
                 facecolor="none", edgecolor=CURVE_COLOR, linewidth=2, zorder=4)
    for position, (_, row) in zip(positions, curve.iterrows()):
        axis.annotate(f"{row['lddt']:.3f}", (position, row["lddt"]),
                      textcoords="offset points", xytext=(0, 11), ha="center",
                      fontsize=8.5, color="#33312e")
    axis.set_xticks(list(positions))
    axis.set_xticklabels([LABEL[a] for a in curve["arm"]], fontsize=9)
    axis.set_xlabel("contacts handed to Helico (cut of the vote ranking)")
    axis.set_ylabel("lDDT")
    axis.set_xlim(-0.5, len(curve) + 1.9)
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("Folding accuracy against how many contacts we hand over\n"
                   "circled = best; the whole usable range spans 0.01 lDDT",
                   fontsize=10.5)


def precision_panel(axis, curve: pd.DataFrame) -> None:
    axis.plot(curve["precision"], curve["lddt"], color=CURVE_COLOR, marker="o",
              markersize=7, linewidth=2.2, zorder=3)
    for _, row in curve.iterrows():
        recall = "" if pd.isna(row["recall"]) else f"\nrecall {row['recall']:.2f}"
        axis.annotate(f"{LABEL[row['arm']].splitlines()[0]}{recall}",
                      (row["precision"], row["lddt"]), textcoords="offset points",
                      xytext=(0, -26 if row["arm"] in ("mf_L2", "mf_2L", "mf_1p5L")
                              else 12),
                      ha="center", fontsize=8, color="#33312e")
    axis.set_xlabel("precision of the contact list handed over")
    axis.set_ylabel("lDDT")
    axis.invert_xaxis()
    span = curve["lddt"].max() - curve["lddt"].min()
    axis.set_ylim(curve["lddt"].min() - 0.22 * span, curve["lddt"].max() + 0.18 * span)
    axis.grid(color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_title("lDDT follows precision, not recall\n"
                   "recall rises 0.52 -> 0.92 left to right and buys nothing",
                   fontsize=10.5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(args.data / "cut_sweep_curve.csv")
    references = pd.read_csv(args.data / "reference_arms.csv")

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    curve_panel(axes[0], curve, references)
    precision_panel(axes[1], curve)
    figure.suptitle("How many MarinFold contacts should Helico be given? "
                    "(eval-val, n=95)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    dest = args.out / "contact_cut_sweep.png"
    figure.savefig(dest, dpi=200)
    plt.close(figure)
    stamp(dest, {"cut_sweep_curve": args.data / "cut_sweep_curve.csv",
                 "reference_arms": args.data / "reference_arms.csv"},
          "Helico lDDT on eval-val against the MarinFold contact cut, from "
          "top-L/5 to every pair any rollout proposed, with the precision and "
          "recall of each list.")
    print(f"[plot] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
