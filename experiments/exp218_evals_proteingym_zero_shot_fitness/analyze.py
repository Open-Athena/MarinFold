# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1, step 2 — aggregate, compare to the leaderboard, plot.

CPU only. Scores the cached conditionals over the ``(K, context-threshold)``
grid, aggregates each point the way ProteinGym does, and puts the result next
to every published baseline **re-aggregated on exactly the assays we scored** —
the only honest way to compare when 5 of 217 assays are out of format range.

Outputs into ``data/`` and ``plots/``:

- ``marinfold_spearman_dms_level.csv`` — per-assay Spearman for every rule,
  same shape as ProteinGym's own file.
- ``leaderboard_comparison.csv`` — us vs the baselines on the same 212.
- ``sweep_orderings.png`` / ``sweep_context.png`` — the two knobs.
- ``category_profile.png`` — our per-category shape against a sequence-only
  and a structure-trained baseline, which is where the "does it behave like a
  structure model?" prediction is settled.

Usage::

    uv run python analyze.py                 # score (cached) + aggregate + plot
    uv run python analyze.py --rescore       # force re-scoring from the .npz
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import proteingym  # noqa: E402
import score as scoring  # noqa: E402
from build_summary import save_plot_with_meta  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"

# The rule grid. K is swept at the best context threshold and vice versa; the
# full cross-product is cheap enough to just take.
ORDERINGS = (1, 4, 16, 64, 200)
CONTEXT_THRESHOLDS = (0.0, 0.5, 0.75, 0.9)

# Baselines worth naming in the comparison, spanning the model classes: two
# single-sequence PLMs (the direct comparison), a structure model, an MSA
# model, and the two floors.
HEADLINE_BASELINES = (
    "ESM2 (650M)",
    "ESM-1v (ensemble)",
    "ESM2 (150M)",
    "ESM2 (35M)",
    "ESM2 (8M)",
    "ESM-IF1",
    "SaProt (650M)",
    "GEMME",
    "TranceptEVE L",
    "Site-Independent",
)


# A residue's statement clears a context threshold `c` in a fraction `1 - c` of
# orderings, so a (K, c) pair yields about `K * (1 - c)` samples per residue.
# Below this many, a non-trivial share of residues get *no* qualifying ordering,
# their log-ratio is undefined, and the variants touching them are dropped — so
# the rule would be compared on a different variant subset than its neighbours.
# Rather than let that happen quietly, the grid excludes those combinations.
MIN_EXPECTED_SAMPLES = 4.0


def rules() -> list[scoring.ScoringRule]:
    out = []
    for k in ORDERINGS:
        for c in CONTEXT_THRESHOLDS:
            # c = 0 keeps every slot by construction, so K alone is the budget.
            if c > 0 and k * (1 - c) < MIN_EXPECTED_SAMPLES:
                continue
            out.append(scoring.ScoringRule(orderings=k, min_context_fraction=c))
    return out


def per_assay_table(rescore: bool) -> pd.DataFrame:
    path = DATA / "marinfold_spearman_dms_level.csv"
    if path.exists() and not rescore:
        return pd.read_csv(path)
    available = {p.stem for p in (DATA / "conditionals").glob("*.npz")}
    if not available:
        raise SystemExit(
            "No cached conditionals found. Run cache_conditionals.py first."
        )
    print(f"Scoring {len(available)} assays over {len(rules())} rules...")
    frame = scoring.score_all(rules(), DATA / "conditionals")
    frame.to_csv(path, index=False)
    return frame


def label_for(orderings: int, context: float) -> str:
    return f"K={orderings}, ctx>={context:g}"


# The headline rule is chosen *a priori*, not from the sweep. "Masked
# marginals" means conditioning on everything else, so the faithful
# implementation is the largest ensemble at the highest context threshold that
# still has enough samples per residue — which the grid filter fixes at
# K=200 / ctx>=0.9 (20 qualifying orderings per residue). Picking the best cell
# of a 13-point grid by its benchmark score would be selection on the test set;
# that number is still reported, labelled as the upper bound it is.
PRIMARY_RULE = (200, 0.9)


def aggregate_rules(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Aggregate every rule in the sweep, keyed by label."""
    results = {}
    for (k, c), chunk in frame.groupby(["orderings", "min_context_fraction"]):
        results[label_for(int(k), float(c))] = {
            "orderings": int(k),
            "min_context_fraction": float(c),
            "variants_dropped": int(chunk.n_dropped.sum()),
            **proteingym.aggregate(chunk[["DMS_id", "spearman"]], reference),
        }
    return results


def pick_rules(results: dict) -> tuple[dict, dict]:
    """Return (the pre-registered primary rule, the best cell of the grid)."""
    primary_label = label_for(*PRIMARY_RULE)
    if primary_label not in results:
        raise SystemExit(
            f"primary rule {primary_label} missing from the sweep; the cache "
            f"probably holds fewer than {PRIMARY_RULE[0]} orderings."
        )
    best_label, best = max(
        results.items(), key=lambda kv: kv[1]["average_spearman"]
    )
    return (
        {"label": primary_label, **results[primary_label]},
        {"label": best_label, **best},
    )


def leaderboard(frame: pd.DataFrame, reference: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Us against every published baseline, all on the same assay set."""
    scored_ids = sorted(frame.DMS_id.unique())
    baselines = proteingym.baseline_table(scored_ids)
    rows = [
        {
            "model": "MarinFold contacts-v1-exp199-1.5B",
            "type": "Single sequence (structure-trained)",
            "average_spearman": rule["average_spearman"],
            **{f"Function_{k}": v for k, v in rule["by_function"].items()},
        }
    ]
    for name in baselines.columns:
        if name == "DMS_id":
            continue
        per = baselines[["DMS_id", name]].rename(columns={name: "spearman"}).dropna()
        if len(per) < len(scored_ids):
            continue  # baseline did not cover every assay we scored
        aggregated = proteingym.aggregate(per, reference)
        rows.append(
            {
                "model": name,
                "type": "",
                "average_spearman": aggregated["average_spearman"],
                **{f"Function_{k}": v for k, v in aggregated["by_function"].items()},
            }
        )
    table = pd.DataFrame(rows).sort_values("average_spearman", ascending=False)
    return table.reset_index(drop=True)


def depth_breakdown(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Spearman by mutational depth — the additive-approximation axis.

    Every single-sequence baseline on the leaderboard scores a k-mutant by
    summing k independent single-site terms, and their published numbers fall
    off sharply with depth. Ours does the same thing at this stage (Phase 2 is
    where the exact joint replaces it), so this is the *baseline* for that
    comparison, not yet the result.
    """
    reference_frame = proteingym.reference()
    scorable = reference_frame[reference_frame.scorable]
    scorable = scorable[scorable.DMS_id.isin(frame.DMS_id.unique())]
    selected = scoring.ScoringRule(
        orderings=rule["orderings"],
        min_context_fraction=rule["min_context_fraction"],
    )
    rows = []
    for _, meta in scorable.iterrows():
        path = DATA / "conditionals" / f"{meta.DMS_id}.npz"
        assay = proteingym.load_assay(meta)
        conditionals = scoring.load_conditionals(path, selected.orderings)
        scores, _ = scoring.score_assay(conditionals, assay, selected)
        depths = scoring.mutational_depth(assay)
        measured = assay.variants.DMS_score.values
        for depth in range(1, 6):
            mask = (depths == depth) if depth < 5 else (depths >= 5)
            mask &= np.isfinite(scores)
            if mask.sum() < 20:
                continue
            rows.append(
                {
                    "DMS_id": meta.DMS_id,
                    "depth": depth if depth < 5 else "5+",
                    "spearman": proteingym.assay_spearman(
                        scores[mask], measured[mask]
                    ),
                    "n_variants": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_sweeps(results: dict) -> None:
    frame = pd.DataFrame(results.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for threshold, chunk in frame.groupby("min_context_fraction"):
        chunk = chunk.sort_values("orderings")
        axes[0].plot(
            chunk.orderings,
            chunk.average_spearman,
            marker="o",
            label=f"context ≥ {threshold:g}",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("orderings ensembled (K)")
    axes[0].set_ylabel("average Spearman")
    axes[0].set_title("Ordering ensemble")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for k, chunk in frame.groupby("orderings"):
        chunk = chunk.sort_values("min_context_fraction")
        axes[1].plot(
            chunk.min_context_fraction,
            chunk.average_spearman,
            marker="o",
            label=f"K={k}",
        )
    axes[1].set_xlabel("minimum context fraction")
    axes[1].set_ylabel("average Spearman")
    axes[1].set_title("Conditioning context")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for axis in axes:
        axis.axhline(0.4152, color="crimson", ls="--", lw=1)
        axis.text(
            0.02, 0.4152, "ESM2 650M", color="crimson", fontsize=7, va="bottom",
            transform=axis.get_yaxis_transform(),
        )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "sweeps.png",
        caption=(
            "Average Spearman on the 212 in-range ProteinGym substitution assays "
            "vs the two knobs an any-order model has and a masked LM does not: "
            "how many document orderings are ensembled (left) and how much of the "
            "rest of the protein each conditional saw (right). Dashed line is "
            "ESM-2 650M re-aggregated on the same 212 assays."
        ),
        dpi=150,
    )
    plt.close(fig)


def plot_category_profile(table: pd.DataFrame) -> None:
    """Our per-category shape vs a sequence-only and a structure-trained model.

    The prediction on record in #218 is that a structure-objective model is
    tilted toward Stability relative to a sequence-only PLM, whatever its
    overall level. Normalizing each model by its own average is what makes that
    a statement about *shape* rather than about who is better.
    """
    wanted = ["MarinFold contacts-v1-exp199-1.5B", "ESM2 (650M)", "ESM-IF1"]
    present = [m for m in wanted if m in set(table.model)]
    categories = list(proteingym.FUNCTION_CATEGORIES)
    fig, axis = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / max(len(present), 1)
    for offset, model in enumerate(present):
        row = table[table.model == model].iloc[0]
        values = [row.get(f"Function_{c}", np.nan) / row.average_spearman for c in categories]
        axis.bar(
            np.arange(len(categories)) + offset * width,
            values,
            width=width,
            label=model,
        )
    axis.axhline(1.0, color="black", lw=1)
    axis.set_xticks(np.arange(len(categories)) + width * (len(present) - 1) / 2)
    axis.set_xticklabels(categories, rotation=20, ha="right")
    axis.set_ylabel("category Spearman / model's own average")
    axis.set_title("Function-category profile (shape, not level)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "category_profile.png",
        caption=(
            "Each model's per-category Spearman divided by its own overall "
            "average, so the bars compare *shape*. A structure-trained model is "
            "expected to lean on Stability (ESM-IF1 does); the question is "
            "whether MarinFold, trained on a structure objective but reading out "
            "sequence, leans the same way."
        ),
        dpi=150,
    )
    plt.close(fig)


def plot_vs_esm(frame: pd.DataFrame, rule: dict) -> None:
    """Per-assay agreement with ESM-2 650M, and what an ensemble would buy."""
    chunk = frame[
        (frame.orderings == rule["orderings"])
        & (frame.min_context_fraction == rule["min_context_fraction"])
    ]
    baselines = proteingym.baseline_table(sorted(chunk.DMS_id.unique()))
    merged = chunk.merge(
        baselines[["DMS_id", "ESM2 (650M)"]], on="DMS_id", how="inner"
    ).dropna()
    fig, axis = plt.subplots(figsize=(5.2, 5))
    axis.scatter(merged["ESM2 (650M)"], merged.spearman, s=14, alpha=0.7)
    limits = [-0.2, 0.9]
    axis.plot(limits, limits, color="black", lw=1, ls="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("ESM-2 650M Spearman (published)")
    axis.set_ylabel("MarinFold Spearman")
    correlation = float(np.corrcoef(merged["ESM2 (650M)"], merged.spearman)[0, 1])
    axis.set_title(f"Per-assay agreement (r = {correlation:.2f})")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "vs_esm2.png",
        caption=(
            "Per-assay Spearman, MarinFold vs ESM-2 650M, on the 212 shared "
            "assays. Points below the diagonal are assays ESM-2 wins. A low "
            "correlation would mean the two models fail on different assays — "
            "which is what would make a structure-objective model useful in an "
            "ensemble even at a lower overall level."
        ),
        dpi=150,
    )
    plt.close(fig)
    return correlation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    reference = proteingym.reference()
    frame = per_assay_table(args.rescore)
    results = aggregate_rules(frame, reference)
    primary, best = pick_rules(results)

    table = leaderboard(frame, reference, primary)
    table.to_csv(DATA / "leaderboard_comparison.csv", index=False)
    depths = depth_breakdown(frame, primary)
    depths.to_csv(DATA / "depth_breakdown.csv", index=False)

    plot_sweeps(results)
    plot_category_profile(table)
    correlation = plot_vs_esm(frame, primary)

    esm2 = float(table[table.model == "ESM2 (650M)"].average_spearman.iloc[0])
    summary = {
        "n_assays_scored": int(frame.DMS_id.nunique()),
        "primary_rule": primary,
        "best_rule_upper_bound": best,
        "cheapest_rule": results.get(label_for(1, 0.0)),
        "per_rule": results,
        "esm2_650m_on_same_assays": esm2,
        "gap_to_esm2_650m": primary["average_spearman"] - esm2,
        "per_assay_correlation_with_esm2": correlation,
        "spearman_by_depth": {
            str(d): float(c.spearman.mean()) for d, c in depths.groupby("depth")
        },
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nassays scored: {summary['n_assays_scored']}")
    print(f"PRIMARY (pre-registered) {primary['label']:<22s} -> {primary['average_spearman']:.4f}")
    print(f"best cell of the grid    {best['label']:<22s} -> {best['average_spearman']:.4f}  (test-set selection)")
    if summary["cheapest_rule"]:
        print(f"cheapest (one pass)      {label_for(1, 0.0):<22s} -> {summary['cheapest_rule']['average_spearman']:.4f}")
    print(f"\nESM-2 650M on the same {summary['n_assays_scored']} assays: {esm2:.4f}"
          f"   (gap {summary['gap_to_esm2_650m']:+.4f})")
    print(f"\n{table.head(14).to_string(index=False)}")
    print(f"\nper-assay correlation with ESM-2 650M: r = {correlation:.3f}")
    print("\nSpearman by mutational depth (additive approximation):")
    for depth, value in summary["spearman_by_depth"].items():
        print(f"  depth {depth}: {value:+.4f}")


if __name__ == "__main__":
    main()
