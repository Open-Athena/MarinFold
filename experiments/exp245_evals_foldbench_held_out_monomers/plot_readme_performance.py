# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The repository README's headline performance figure.

Two panels, because one set cannot answer both questions a reader arrives with.

**Left — the legacy 554-protein benchmark.** Every published MarinFold number
since #75 lives on this set, so it is the only place the model generations can be
compared to each other. It is also *not* homology-controlled and is 75 % de novo
designed protein, so it flatters single-sequence methods; that is exactly why the
right panel exists.

**Right — eval-test.** FoldBench's 217 natural monomers that nothing in this repo
had ever scored, and which #225 verifiably removed from the #232 training corpora
at 30 % identity. It is a held-out confirmation set, read rarely and logged in
``data/eval_test_reads.md``; publishing this figure is one of those reads. Routine
tracking happens on eval-val and the legacy 554. Two sequence-KNN nulls are drawn beside the models: copying the
contacts of a protein's ten nearest training sequences, once out of the corpus
#199 trained on and once out of the decontaminated corpus #232 trained on. A
model only earns a number by clearing the null over its own corpus.

Every value is a mean over per-protein rows that some experiment already
published; nothing is recomputed here.

    uv run python plot_readme_performance.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import upstream as U  # noqa: E402

DATA = U.DATA
OUT = U.HERE / "plots" / "readme_performance.png"
TABLE = DATA / "readme_performance.csv"

#: Per-protein legacy-554 rows: MarinFold generations and Protenix single-seq
#: from #244's comparison table, the other baselines from #213's.
LEGACY_ROWS = U.EXP232_ROLLOUT_DIR / "data" / "all_r_rows.csv.gz"
EXP213_WIDE = (U.EXPERIMENTS / "exp213_evals_train_sequence_overlap_audit"
               / "data" / "per_protein_wide.csv.gz")

#: (key in #244's table, label, colour role).
LEGACY_MARINFOLD = (
    ("exp75-reproduced", "MarinFold #75", "history"),
    ("exp146", "MarinFold #146 (3B)", "history"),
    ("exp166", "MarinFold #166", "history"),
    ("cw-p06-aug", "MarinFold #199", "history"),
    ("cw-p06-cool", "MarinFold #199 cooldown\n(default model)", "default"),
    ("exp232-m2-p06-decontam", "MarinFold #232\n(decontaminated data)", "decontam"),
)
LEGACY_BASELINES = (
    ("Protenix-v2 single-seq", "Protenix-v2\nsingle-seq"),
    ("ESMFold", "ESMFold"),
    ("ESMFold2", "ESMFold2"),
    ("Protenix-v2 + MSA", "Protenix-v2\n+ MSA"),
    ("seq-KNN k=10 (null)", "seq-KNN null"),
)

#: eval-test panel: predictor name in exp245's table -> (label, colour role).
TEST_PANEL = (
    ("#199 cooldown (contaminated)", "MarinFold #199 cooldown\n(default model)", "default"),
    ("#232 m2-p06 (decontaminated)", "MarinFold #232\n(decontaminated data)", "decontam"),
    ("Protenix-v2 single-seq", "Protenix-v2\nsingle-seq", "baseline"),
    ("ESMFold", "ESMFold", "baseline"),
    ("ESMFold2", "ESMFold2", "baseline"),
    ("Protenix-v2 + MSA", "Protenix-v2\n+ MSA", "baseline"),
    ("seq-KNN (unfiltered corpus)", "seq-KNN null\n(#199's corpus)", "null"),
    ("seq-KNN (decontaminated corpus)", "seq-KNN null\n(#232's corpus)", "null"),
)

COLORS = {
    "history": "#b9b5b0",
    "default": "#d55e00",
    "decontam": "#e69f00",
    "baseline": "#5a5754",
    "null": "#9fc8e8",
}
BOOTSTRAP_DRAWS = 4_000
SEED = 245


def interval(values: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(SEED)
    index = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def legacy_series() -> list[dict]:
    rows = pd.read_csv(LEGACY_ROWS)
    wide = pd.read_csv(EXP213_WIDE)
    wide = wide[(wide["range"] == "all") & (wide["cut"] == "R")]
    series = []
    for key, label, role in LEGACY_MARINFOLD:
        values = rows.loc[rows.key == key, "precision"].to_numpy()
        low, high = interval(values)
        series.append({"panel": "legacy_554", "label": label, "role": role,
                       "n": len(values), "value": float(values.mean()),
                       "ci_low": low, "ci_high": high})
    for column, label in LEGACY_BASELINES:
        values = wide[column].dropna().to_numpy()
        low, high = interval(values)
        role = "null" if "KNN" in label else "baseline"
        series.append({"panel": "legacy_554", "label": label, "role": role,
                       "n": len(values), "value": float(values.mean()),
                       "ci_low": low, "ci_high": high})
    return series


def test_series() -> list[dict]:
    per_protein = pd.read_csv(DATA / "per_protein.csv.gz")
    sets = pd.read_csv(DATA / "eval_sets.csv")
    keep = set(sets.loc[(sets.scorable == 1) & (sets.eval_set == "eval-test"), "stem"])
    frame = per_protein[(per_protein["range"] == "all") & (per_protein["cut"] == "R")
                        & per_protein.stem.isin(keep)]
    series = []
    for predictor, label, role in TEST_PANEL:
        values = frame.loc[frame.predictor == predictor, "precision"].to_numpy()
        low, high = interval(values)
        series.append({"panel": "eval_test", "label": label, "role": role,
                       "n": len(values), "value": float(values.mean()),
                       "ci_low": low, "ci_high": high})
    return series


def draw(table: pd.DataFrame, out: Path) -> None:
    panels = (
        ("legacy_554", "The legacy 554-protein benchmark\n"
                       "every MarinFold number is quoted on\n"
                       "(75 % de novo designed, no homology control)"),
        ("eval_test", "eval-test: 217 natural FoldBench monomers,\n"
                      "decontaminated at 30 % identity and held out\n"
                      "(read rarely, on the record)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharey=True,
                               gridspec_kw={"width_ratios": [11, 8]})
    for axis, (panel, title) in zip(axes, panels, strict=True):
        part = table[table.panel == panel].reset_index(drop=True)
        positions = np.arange(len(part))
        colors = [COLORS[role] for role in part.role]
        axis.bar(positions, part.value, color=colors, width=0.7)
        axis.errorbar(positions, part.value,
                      yerr=[part.value - part.ci_low, part.ci_high - part.value],
                      fmt="none", ecolor="#33312e", elinewidth=1.1, capsize=3)
        for position, value, high in zip(positions, part.value, part.ci_high,
                                         strict=True):
            axis.text(position, high + 0.016, f"{value:.3f}", ha="center",
                      fontsize=9, color="#33312e")
        axis.set_xticks(positions)
        axis.set_xticklabels(part.label, rotation=38, ha="right", fontsize=8.6)
        axis.set_title(title, fontsize=10.5)
        axis.grid(axis="y", color="#dddad6", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Contact R-precision (all ranges)")
    axes[0].set_ylim(0, 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[role]) for role in
               ("default", "decontam", "history", "baseline", "null")]
    axes[1].legend(handles, ["default model", "trained on decontaminated data",
                             "earlier MarinFold", "structure predictor",
                             "sequence-KNN null"],
                   frameon=False, fontsize=8.8, loc="upper left")
    figure.suptitle(
        "Predicting residue-residue contacts from a single sequence, no MSA and "
        "no PLM", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    table = pd.DataFrame(legacy_series() + test_series())
    table.to_csv(TABLE, index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    draw(table, OUT)
    meta = {
        "script": Path(sys.argv[0]).name,
        "args": sys.argv[1:],
        "caption": (
            "Contact R-precision on the legacy 554-protein benchmark (left, where "
            "every published MarinFold number lives) and on exp245's eval-test "
            "(right, 217 natural FoldBench monomers held out and decontaminated "
            "at 30 % identity). Error bars are 95 % bootstrap intervals over "
            "proteins. The two sequence-KNN nulls copy the contacts of each "
            "protein's ten nearest training sequences, out of the corpus each "
            "model family actually trained on."
        ),
        "plot": OUT.name,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "sources": {
            "legacy_rows": {"path": str(LEGACY_ROWS.relative_to(U.REPO)),
                            "sha256": U.sha256(LEGACY_ROWS)},
            "exp213_wide": {"path": str(EXP213_WIDE.relative_to(U.REPO)),
                            "sha256": U.sha256(EXP213_WIDE)},
            "per_protein": {"path": str((DATA / "per_protein.csv.gz").relative_to(U.REPO)),
                            "sha256": U.sha256(DATA / "per_protein.csv.gz")},
        },
    }
    OUT.with_suffix(OUT.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(table.round(4).to_string(index=False))
    print(f"\n[plots] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
