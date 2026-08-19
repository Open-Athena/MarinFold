# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 — three figures, one per claim the analysis actually supports.

``family_abundance.png``
    Mean R-precision by quartile of MSA depth, one line per predictor. This is
    the result: every predictor rises with how many relatives a protein has, and
    the slopes order the predictors by how much homology they depend on.

``explainable_variance.png``
    Cross-validated R² of a model that sees only protein properties, per
    predictor. How much of the per-protein spread is a property of the protein
    rather than of the model — and the same ordering falls out.

``feature_heatmap.png``
    Spearman ρ for a readable subset of features against every predictor. The
    block structure is the point: homology features are dark, geometry and
    biology features are not.

    uv run python plot_properties.py
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
PLOTS = U.HERE / "plots"

COLORS = {
    "#232 m2-p06 (decontaminated)": "#d55e00",
    "#232 m1-p02 (decontaminated)": "#e69f00",
    "#199 cooldown (contaminated)": "#0072b2",
    "ESMFold": "#7f7c78",
    "ESMFold2": "#4a4744",
    "Protenix-v2 + MSA": "#009e73",
    "Protenix-v2 single-seq": "#a8a5a1",
    "seq-KNN (unfiltered corpus)": "#9fc8e8",
    "seq-KNN (decontaminated corpus)": "#c9dff0",
}
#: Features worth showing by name, grouped the way the analysis groups them.
HEATMAP_FEATURES = [
    ("family abundance", [
        ("msa_log_depth", "MSA depth (log)"),
        ("train_log_n_hits", "training homologs (log)"),
        ("knn_n_hits", "KNN neighbours"),
        ("n_surviving_alignments", "alignments surviving decontamination"),
        ("knn_best_identity", "best training identity"),
    ]),
    ("size and shape", [
        ("length", "length"),
        ("contacts_per_residue", "contacts per residue"),
        ("relative_contact_order", "relative contact order"),
        ("frac_long_contacts", "fraction long-range contacts"),
        ("resolved_fraction", "resolved fraction"),
        ("radius_of_gyration", "radius of gyration"),
    ]),
    ("secondary structure", [
        ("frac_helix", "helix fraction"),
        ("frac_sheet", "sheet fraction"),
        ("n_sse", "number of SSEs"),
    ]),
    ("biology", [
        ("is_viral", "viral"),
        ("kingdom_bacteria", "bacterial"),
        ("n_uniprot_domains", "UniProt domains"),
        ("is_membrane", "membrane"),
        ("has_ec", "enzyme (has EC)"),
        ("frac_low_complexity", "low-complexity fraction"),
    ]),
]


def stamp(path: Path, caption: str, sources: dict[str, Path]) -> None:
    meta = {
        "script": Path(sys.argv[0]).name, "args": sys.argv[1:], "caption": caption,
        "plot": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {name: str(source.relative_to(U.REPO)) for name, source in sources.items()},
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load() -> pd.DataFrame:
    features = pd.read_csv(DATA / "protein_features.csv").set_index("stem")
    scores = pd.read_csv(U.PER_PROTEIN)
    scores = scores[(scores["range"] == "all") & (scores["cut"] == "R")]
    wide = scores.pivot_table(index="stem", columns="predictor", values="precision")
    return features.join(wide)


def family_abundance(frame: pd.DataFrame, out: Path) -> None:
    quartiles, edges = pd.qcut(frame.msa_log_depth, 4, retbins=True,
                               labels=["Q1 shallowest", "Q2", "Q3", "Q4 deepest"])
    # Bin boundaries in sequence counts, so a reader can place a protein of
    # their own without going back to the table.
    bounds = [int(round(10 ** edge)) for edge in edges]
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    for predictor in U.PREDICTORS:
        if predictor not in frame:
            continue
        means = frame.groupby(quartiles, observed=False)[predictor].mean()
        axis.plot(range(4), means.to_numpy(), marker="o", markersize=6,
                  linewidth=2.6 if "#232 m2" in predictor or "#199" in predictor else 1.6,
                  color=COLORS[predictor], zorder=3 if "#232 m2" in predictor else 2)
        axis.annotate(predictor, (3, means.iloc[-1]), textcoords="offset points",
                      xytext=(9, 0), va="center", fontsize=8.6, color=COLORS[predictor])
    counts = quartiles.value_counts().sort_index()
    axis.set_xticks(range(4))
    axis.set_xticklabels(
        [f"{label}\n{low:,}–{high:,} seqs\nn={n}"
         for label, low, high, n in zip(quartiles.cat.categories, bounds[:-1],
                                        bounds[1:], counts, strict=True)],
        fontsize=8.6)
    axis.set_xlim(-0.15, 4.9)
    axis.set_xlabel("MSA depth quartile — how many relatives the protein has")
    axis.set_ylabel("Mean R-precision (all ranges)")
    axis.set_title("Every predictor gets better on proteins with more relatives.\n"
                   "MarinFold's slope is steeper than the MSA methods', not shallower.",
                   fontsize=11.5)
    axis.grid(axis="y", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def explainable_variance(out: Path) -> None:
    performance = pd.read_csv(DATA / "model_performance.csv")
    best = (performance.groupby("predictor").cv_r2_mean.max()
            .sort_values(ascending=False))
    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    colors = [COLORS.get(name, "#8f8b86") for name in best.index]
    axis.barh(range(len(best)), best.clip(lower=0).to_numpy(), color=colors)
    for index, (name, value) in enumerate(best.items()):
        axis.text(max(value, 0) + 0.012, index, f"{value:.2f}", va="center", fontsize=9)
    axis.set_yticks(range(len(best)))
    axis.set_yticklabels(best.index, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("Cross-validated R² from protein properties alone")
    axis.set_title("How much of each predictor's per-protein accuracy is\n"
                   "a property of the protein rather than of the model",
                   fontsize=11.5)
    axis.grid(axis="x", color="#dddad6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def feature_heatmap(out: Path) -> None:
    associations = pd.read_csv(DATA / "associations.csv")
    pooled = associations[associations.subset == "natural (pooled)"]
    labels, names, boundaries = [], [], []
    for group, entries in HEATMAP_FEATURES:
        boundaries.append((group, len(names)))
        for feature, label in entries:
            if feature in set(pooled.feature):
                names.append(feature)
                labels.append(label)
    matrix = pooled.pivot_table(index="feature", columns="predictor", values="rho")
    columns = [p for p in U.PREDICTORS if p in matrix.columns]
    matrix = matrix.reindex(index=names, columns=columns)

    figure, axis = plt.subplots(figsize=(11.5, 8.4))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.7, vmax=0.7,
                        aspect="auto")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns, rotation=38, ha="right", fontsize=8.8)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(labels, fontsize=8.8)
    for row in range(len(names)):
        for column in range(len(columns)):
            value = matrix.iat[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7.4,
                          color="white" if abs(value) > 0.42 else "#33312e")
    for _, start in boundaries[1:]:
        axis.axhline(start - 0.5, color="#33312e", linewidth=1.1)
    for group, start in boundaries:
        axis.text(-0.62, start - 0.42, group, transform=axis.get_yaxis_transform(),
                  fontsize=9, style="italic", color="#33312e")
    figure.colorbar(image, ax=axis, shrink=0.62, label="Spearman ρ with R-precision")
    axis.set_title("Only one block of features explains contact accuracy — and it is\n"
                   "the same block for every predictor", fontsize=12)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    frame = load()
    sources = {"features": DATA / "protein_features.csv",
               "associations": DATA / "associations.csv",
               "scores": U.PER_PROTEIN}

    path = PLOTS / "family_abundance.png"
    family_abundance(frame, path)
    stamp(path, "Mean all-range R-precision by quartile of MSA depth over the 314 "
                "natural FoldBench monomers. Quartile boundaries are 784, 3,015 "
                "and 7,413 sequences; the set spans 2 to 19,393. MSA depth stands "
                "in for how many relatives a protein has — it correlates 0.80 with "
                "the number of homologs in our training corpus and 0.87 with KNN "
                "neighbour count.", sources)

    path = PLOTS / "explainable_variance.png"
    explainable_variance(path)
    stamp(path, "Cross-validated R² of the best of a ridge and a gradient-boosted "
                "model predicting each predictor's per-protein R-precision from "
                "60 protein properties, 5-fold CV over 314 proteins. Negative "
                "values are clipped to zero.", sources)

    path = PLOTS / "feature_heatmap.png"
    feature_heatmap(path)
    stamp(path, "Spearman ρ between each protein property and each predictor's "
                "per-protein R-precision, pooled over the 314 natural monomers.",
          sources)
    print("[plots] three figures written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
