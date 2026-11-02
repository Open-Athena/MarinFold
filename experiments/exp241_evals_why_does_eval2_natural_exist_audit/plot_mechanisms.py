# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5 — the four figures.

1. ``mechanism_ladder.png`` — where the 78 go once each one is checked: 15 are
   not natural, and the rest split across three reasons none of which is
   "the sequence is new".
2. ``known_but_unsampled.png`` — the headline. Of the 63 audited-natural
   proteins, how many are in UniProt, how many AlphaFold folded, and how many we
   trained on; beside the year UniProt first published each sequence.
3. ``kingdom_gap.png`` — survival into eval2 by kingdom against each arm's hit
   rate. Viral proteins are the hole in both corpora.
4. ``base_rate.png`` — the unconditioned rate: random recent PDB protein chains
   put through eval2's own filter, next to the eval sets.

    uv run python plot_mechanisms.py
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import upstream as U  # noqa: E402
from build_summary import save_plot_with_meta  # noqa: E402

DATA = U.HERE / "data"
PLOTS = U.HERE / "plots"

#: The series palette this experiment family already uses. Identity is carried
#: by position and a direct value label on every mark, never by hue alone — the
#: red/orange pair is the one a CVD reader would struggle with.
BLUE = "#2980b9"
RED = "#c0392b"
ORANGE = "#d68910"
GREY = "#95a5a6"
GREEN = "#16a085"

MECHANISM_LABELS = {
    "designed_not_natural": "not natural\n(designed protein)",
    "not_in_uniprot": "natural, but not\na UniProt sequence",
    "afdb_absent": "in UniProt, but\nAlphaFold never folded it",
    "unsampled_corpus": "folded by AlphaFold —\nwe just did not train on it",
    "search_miss": "search miss\n(pipeline defect)",
}
CAPTIONS = {
    "mechanism_ladder":
        "Each of the 78 proteins published as eval2-natural, charged to the earliest checked reason it escaped the 40% filter. Hatching marks the share drawn from the novelty-curated CASP-FM / CAMEO-hard sources.",
    "known_but_unsampled":
        "The 63 audited-natural eval2 proteins: how many are in UniProt, how many AlphaFold folded, and how many our AFDB training arm contains. Right: the year UniProt first published each sequence.",
    "kingdom_gap":
        "Share of eval proteins with a hit in each training arm, and share surviving eval2's filter, by source kingdom. Viral proteins are missing from both corpora and survive at 4x the bacterial rate.",
    "base_rate":
        "Random recent PDB protein chains put through eval2's own filter and target database, beside the eval-set slices. The unconditioned survival rate is the base rate eval2-natural has to be judged against.",
}

MECHANISM_COLORS = {
    "designed_not_natural": RED,
    "not_in_uniprot": GREY,
    "afdb_absent": ORANGE,
    "unsampled_corpus": BLUE,
    "search_miss": "#000000",
}


def read(path):
    with (DATA / path).open() as fh:
        return list(csv.DictReader(fh))


def plot_mechanism_ladder(out: Path, argv: list[str]) -> None:
    rows = [r for r in read("mechanism_counts.csv") if int(r["n"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels = [MECHANISM_LABELS[r["escape_mechanism"]] for r in rows]
    counts = np.array([int(r["n"]) for r in rows])
    curated = np.array([int(r["n_novelty_curated"]) for r in rows])
    y = np.arange(len(rows))[::-1]

    ax.barh(y, counts, 0.6,
            color=[MECHANISM_COLORS[r["escape_mechanism"]] for r in rows])
    # The hatched part is the share drawn from CASP-FM / CAMEO-hard, which are
    # curated for difficulty; the plain part is a slice of recent PDB.
    ax.barh(y, curated, 0.6, color="none", edgecolor="white",
            hatch="///", linewidth=0.0)
    for yi, count, cur in zip(y, counts, curated):
        ax.text(count + 0.7, yi, f"{count}   ({cur} from CASP/CAMEO)",
                va="center", fontsize=9, color="#2c3e50")
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlim(0, max(counts) * 1.6)
    ax.set_xlabel("proteins (of the 78 published as eval2-natural)")
    ax.set_title("Why each of the 78 escaped a 40 % identity filter\n"
                 "checked per protein, charged to the earliest reason that applies",
                 fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_plot_with_meta(fig, out, caption=CAPTIONS[out.stem], args=argv)


def plot_known_but_unsampled(out: Path, argv: list[str]) -> None:
    rows = [r for r in read("mechanism_table.csv")
            if r["escape_mechanism"] != "designed_not_natural"]
    n = len(rows)
    stages = [
        ("audited natural\nin eval2", n, GREY),
        ("has a UniProt\nsequence entry", sum(1 for r in rows if r["uniprot_accessions"]), BLUE),
        ("AlphaFold DB has\na model for it", sum(1 for r in rows if r["in_afdb_full"] == "1"), ORANGE),
        ("in our AFDB\ntraining arm", sum(1 for r in rows if r["in_afdb_arm"] == "1"), RED),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(stages))
    ax.bar(x, [s[1] for s in stages], 0.62, color=[s[2] for s in stages])
    for xi, (_, value, _) in zip(x, stages):
        ax.text(xi, value + 1.2, str(value), ha="center", fontsize=11,
                fontweight="bold", color="#2c3e50")
    ax.set_xticks(x, [s[0] for s in stages], fontsize=9)
    ax.set_ylim(0, n * 1.22)
    ax.set_ylabel("proteins")
    ax.set_title("The sequences are known. We did not train on them.", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    years = sorted(int(r["uniprot_first_public"][:4])
                   for r in rows if r["uniprot_first_public"][:4].isdigit())
    ax2.hist(years, bins=np.arange(min(years), 2027, 2), color=BLUE,
             edgecolor="white", linewidth=0.8)
    median = years[len(years) // 2]
    ax2.axvline(median, color=RED, linewidth=2)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.2)
    ax2.text(median, ax2.get_ylim()[1] * 0.94, f" median {median}",
             color=RED, fontsize=10, fontweight="bold", va="top")
    ax2.set_xlabel("year UniProt first published the sequence")
    ax2.set_ylabel("proteins")
    ax2.set_title(f"...and they are not new sequences (n={len(years)})", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save_plot_with_meta(fig, out, caption=CAPTIONS[out.stem], args=argv)


def plot_kingdom_gap(out: Path, argv: list[str]) -> None:
    rows = [r for r in read("kingdom_by_arm.csv")
            if r["kingdom"] not in ("unknown", "unclassified")
            and int(r["n"]) >= 5]
    rows.sort(key=lambda r: -float(r["share_in_eval2"]))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(rows))
    width = 0.27
    series = [
        ("has an AFDB-arm hit", [int(r["n_afdb_hit"]) / int(r["n"]) for r in rows], BLUE),
        ("has an ESM-Atlas hit", [int(r["n_esm_atlas_hit"]) / int(r["n"]) for r in rows], GREEN),
        ("survives into eval2", [float(r["share_in_eval2"]) for r in rows], RED),
    ]
    for i, (label, values, color) in enumerate(series):
        offset = (i - 1) * width
        ax.bar(x + offset, values, width * 0.92, color=color, label=label)
        for xi, value in zip(x, values):
            ax.text(xi + offset, value + 0.02, f"{value:.0%}", ha="center",
                    fontsize=8, color="#2c3e50")
    ax.set_xticks(x, [f"{r['kingdom']}\n(n={r['n']})" for r in rows], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of eval proteins")
    ax.set_title("Viral proteins are the hole in both training corpora\n"
                 "(‘synthetic’ = designed protein, shown for contrast)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_plot_with_meta(fig, out, caption=CAPTIONS[out.stem], args=argv)


def plot_base_rate(out: Path, argv: list[str]) -> None:
    summary = {(r["subset"], r["filter"]): r for r in read("base_rate_summary.csv")}
    natural = summary[("natural", "passes_40")]
    strict = summary[("natural", "passes_30")]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))

    bars = [
        (f"random recent PDB\nnatural chains\n(n={natural['n']})",
         float(natural["rate"]), BLUE),
        (f"...at <30 % id\n(n={strict['n']})", float(strict["rate"]), GREY),
    ]
    identity = U.read_identity_table()
    for label, keys in (("FoldBench-100", ("foldbench100",)),
                        ("FoldBench rest\n(+222)", ("foldbench_rest",)),
                        ("CASP-FM +\nCAMEO-hard", ("casp_fm", "cameo_hard"))):
        subset = [r for k, r in identity.items() if k[0] in keys]
        survive = sum(1 for r in subset
                      if not r["best_identity_covered"]
                      or float(r["best_identity_covered"]) < U.EVAL2_THRESHOLD)
        bars.append((f"{label}\n(n={len(subset)})", survive / len(subset),
                     ORANGE if "FoldBench" in label else RED))

    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], 0.6, color=[b[2] for b in bars])
    for xi, (_, value, _) in zip(x, bars):
        ax.text(xi, value + 0.015, f"{value:.0%}", ha="center", fontsize=11,
                fontweight="bold", color="#2c3e50")
    ax.set_xticks(x, [b[0] for b in bars], fontsize=9)
    ax.set_ylabel("share with no >= 40 % training relative")
    ax.set_ylim(0, max(b[1] for b in bars) * 1.22)
    ax.set_title("eval2-natural is not an anomaly: the base rate is already "
                 f"{float(natural['rate']):.0%}\n"
                 "same filter, same target database, unconditioned sample",
                 fontsize=11, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_plot_with_meta(fig, out, caption=CAPTIONS[out.stem], args=argv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    args = list(sys.argv[1:])
    PLOTS.mkdir(exist_ok=True)
    plot_mechanism_ladder(PLOTS / "mechanism_ladder.png", args)
    plot_known_but_unsampled(PLOTS / "known_but_unsampled.png", args)
    plot_kingdom_gap(PLOTS / "kingdom_gap.png", args)
    if (DATA / "base_rate_summary.csv").exists():
        plot_base_rate(PLOTS / "base_rate.png", args)
    else:
        print("[skip] base_rate.png — run measure_base_rate.py first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
