# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run adjusted ANOVA/OLS feature-importance summaries for exp324.

The target is the paired R-precision gap between a MarinFold checkpoint and
ESMFold2 on the same protein. We fit small adjusted OLS models of the form

    model_minus_esmfold2 ~ feature + length + contact_count + resolution + uniref50

and rank features by partial eta-squared. Direction is reported separately:
positive effects are MarinFold-favorable and negative effects are
ESMFold2-favorable.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ESMFOLD_R = "esmfold2_n100_all_r"
CONTACTS_V2_R = "delta_v2_step78499_n100_all_r"
EXP277_R = "exp277_full_epoch_step266344_n100_all_r"
BASE_COVAR_COLUMNS = ["log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]
BASE_COVARS = " + ".join(BASE_COVAR_COLUMNS)

CATEGORICAL_FEATURES = [
    "lineage_level_1",
    "lineage_level_2",
    "method",
    "structure_status",
    "function_status",
    "lineage_status",
    "is_synthetic",
]
NUMERIC_FEATURES = [
    "seq_len",
    "contacts_emitted",
    "highest_contact_degree",
    "lowest_included_contact_degree",
    "resolution",
    "uniref100_size",
    "uniref90_size",
    "uniref50_size",
]
TAG_FEATURES = ["go_slim_tags", "cath_class_tags", "cath_topology_tags"]
CATEGORY_LABELS = {
    "All proteins": ("all", None),
    "Synthetic": ("column_eq", ("is_synthetic", True)),
    "Solution NMR": ("column_eq", ("method", "SOLUTION NMR")),
    "Riboviria": ("column_eq", ("lineage_level_2", "Riboviria")),
    "Duplodnaviria": ("column_eq", ("lineage_level_2", "Duplodnaviria")),
    "Varidnaviria": ("column_eq", ("lineage_level_2", "Varidnaviria")),
    "CATH topology 1.20.5": ("tag", ("cath_topology_tags", "1.20.5")),
    "GO: cell wall organization/biogenesis": (
        "tag",
        ("go_slim_tags", "cell wall organization or biogenesis"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/anova_feature_importance"),
    )
    parser.add_argument("--min-feature-count", type=int, default=50)
    return parser.parse_args()


def tag_memberships(series: pd.Series) -> pd.Series:
    """Return a set of semicolon-separated tags for each row."""
    return series.fillna("").astype(str).map(lambda value: {tag.strip() for tag in value.split(";") if tag.strip()})


def add_analysis_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add paired deltas and common covariates used by all adjusted models."""
    out = table.copy()
    out["contacts_v2_minus_esmfold2"] = out[CONTACTS_V2_R] - out[ESMFOLD_R]
    out["exp277_minus_esmfold2"] = out[EXP277_R] - out[ESMFOLD_R]
    out["log_seq_len"] = np.log(out["seq_len"].clip(lower=1))
    out["log_contacts_emitted"] = np.log1p(out["contacts_emitted"].clip(lower=0))
    out["log_uniref50"] = np.log10(out["uniref50_size"].fillna(0) + 1)
    out["resolution_filled"] = out["resolution"].fillna(out["resolution"].median())
    return out


def partial_eta_squared(anova_row: pd.Series, residual_ss: float) -> float:
    """Return partial eta-squared for one ANOVA row."""
    sum_sq = float(anova_row["sum_sq"])
    return sum_sq / (sum_sq + float(residual_ss))


def categorical_importance(table: pd.DataFrame, delta_col: str, feature: str, min_count: int) -> dict[str, Any] | None:
    """Rank one categorical feature with type-II ANOVA."""
    required = [delta_col, feature, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]
    work = table.dropna(subset=required).copy()
    counts = work[feature].fillna("unknown").astype(str).value_counts()
    keep = set(counts[counts >= min_count].index)
    work[feature] = work[feature].fillna("unknown").astype(str).where(lambda values: values.isin(keep), "__rare__")
    if work[feature].nunique() < 2:
        return None

    model = smf.ols(f"{delta_col} ~ C({feature}) + {BASE_COVARS}", data=work).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    row = anova.loc[f"C({feature})"]
    means = work.groupby(feature)[delta_col].mean().sort_values(ascending=False)
    return {
        "comparison": delta_col,
        "feature_family": "categorical",
        "feature": feature,
        "top_positive_level_or_tag": means.index[0],
        "bottom_level_or_tag": means.index[-1],
        "direction_effect": means.iloc[0] - means.iloc[-1],
        "n": len(work),
        "f_stat": float(row["F"]),
        "p_value": float(row["PR(>F)"]),
        "partial_eta2": partial_eta_squared(row, anova.loc["Residual", "sum_sq"]),
    }


def numeric_covars(feature: str) -> str:
    """Return covariates for a numeric feature, excluding its own transform."""
    related_covars = {
        "seq_len": "log_seq_len",
        "contacts_emitted": "log_contacts_emitted",
        "resolution": "resolution_filled",
        "uniref50_size": "log_uniref50",
    }
    covars = [col for col in BASE_COVAR_COLUMNS if col != related_covars.get(feature)]
    return " + ".join(covars)


def numeric_importance(table: pd.DataFrame, delta_col: str, feature: str) -> dict[str, Any] | None:
    """Rank one numeric feature with a one-standard-deviation OLS coefficient."""
    covar_formula = numeric_covars(feature)
    covar_columns = [col.strip() for col in covar_formula.split("+")]
    required = [delta_col, feature, *covar_columns]
    work = table.dropna(subset=required).copy()
    if len(work) < 100 or work[feature].nunique() < 5:
        return None
    std = work[feature].std()
    if not np.isfinite(std) or std == 0:
        return None
    work["z_feature"] = (work[feature] - work[feature].mean()) / std
    model = smf.ols(f"{delta_col} ~ z_feature + {covar_formula}", data=work).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    row = anova.loc["z_feature"]
    coef = float(model.params["z_feature"])
    return {
        "comparison": delta_col,
        "feature_family": "numeric",
        "feature": feature,
        "top_positive_level_or_tag": "higher values" if coef > 0 else "lower values",
        "bottom_level_or_tag": "lower values" if coef > 0 else "higher values",
        "direction_effect": coef,
        "n": len(work),
        "f_stat": float(row["F"]),
        "p_value": float(row["PR(>F)"]),
        "partial_eta2": partial_eta_squared(row, anova.loc["Residual", "sum_sq"]),
    }


def tag_importance(table: pd.DataFrame, delta_col: str, tag_col: str, min_count: int) -> list[dict[str, Any]]:
    """Rank individual tag indicators within a semicolon-separated tag column."""
    tag_sets = tag_memberships(table[tag_col])
    counts: dict[str, int] = {}
    for tags in tag_sets:
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1

    rows: list[dict[str, Any]] = []
    for tag, count in sorted(counts.items()):
        if count < min_count:
            continue
        required = [delta_col, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]
        work = table.dropna(subset=required).copy()
        work["has_tag"] = tag_sets.loc[work.index].map(lambda tags: tag in tags).astype(int)
        if work["has_tag"].nunique() < 2:
            continue
        model = smf.ols(f"{delta_col} ~ has_tag + {BASE_COVARS}", data=work).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        row = anova.loc["has_tag"]
        rows.append(
            {
                "comparison": delta_col,
                "feature_family": tag_col,
                "feature": tag_col,
                "top_positive_level_or_tag": tag,
                "bottom_level_or_tag": f"not {tag}",
                "direction_effect": float(model.params["has_tag"]),
                "n": count,
                "f_stat": float(row["F"]),
                "p_value": float(row["PR(>F)"]),
                "partial_eta2": partial_eta_squared(row, anova.loc["Residual", "sum_sq"]),
            }
        )
    return rows


def rank_feature_importance(table: pd.DataFrame, delta_col: str, min_count: int) -> pd.DataFrame:
    """Return one ranked feature-importance table for a model-vs-ESMFold2 gap."""
    rows: list[dict[str, Any]] = []
    for feature in CATEGORICAL_FEATURES:
        if feature in table:
            row = categorical_importance(table, delta_col, feature, min_count)
            if row is not None:
                rows.append(row)
    for feature in NUMERIC_FEATURES:
        if feature in table:
            row = numeric_importance(table, delta_col, feature)
            if row is not None:
                rows.append(row)
    for tag_col in TAG_FEATURES:
        if tag_col in table:
            rows.extend(tag_importance(table, delta_col, tag_col, min_count))

    out = pd.DataFrame(rows)
    out["abs_direction_effect"] = out["direction_effect"].abs()
    return out.sort_values(["partial_eta2", "abs_direction_effect"], ascending=False)


def mask_for_category(table: pd.DataFrame, kind: str, spec: Any) -> pd.Series:
    """Return a boolean mask for one named category summary."""
    if kind == "all":
        return pd.Series(True, index=table.index)
    if kind == "column_eq":
        column, value = spec
        return table[column].eq(value).fillna(False)
    if kind == "tag":
        column, tag = spec
        return tag_memberships(table[column]).map(lambda tags: tag in tags)
    raise ValueError(f"unknown category mask kind {kind!r}")


def category_gap_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed category comparison table used in the README."""
    rows = []
    for label, (kind, spec) in CATEGORY_LABELS.items():
        work = table.loc[mask_for_category(table, kind, spec)].dropna(subset=[ESMFOLD_R, CONTACTS_V2_R, EXP277_R])
        if len(work) == 0:
            continue
        rows.append(
            {
                "category": label,
                "n": len(work),
                "esmfold2_r": work[ESMFOLD_R].mean(),
                "contacts_v2_r": work[CONTACTS_V2_R].mean(),
                "exp277_r": work[EXP277_R].mean(),
                "contacts_v2_gap": work[CONTACTS_V2_R].mean() - work[ESMFOLD_R].mean(),
                "exp277_gap": work[EXP277_R].mean() - work[ESMFOLD_R].mean(),
                "contacts_v2_win_frac": (work[CONTACTS_V2_R] > work[ESMFOLD_R]).mean(),
                "exp277_win_frac": (work[EXP277_R] > work[ESMFOLD_R]).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    table = add_analysis_columns(pd.read_parquet(args.table))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    importance = pd.concat(
        [
            rank_feature_importance(table, "contacts_v2_minus_esmfold2", args.min_feature_count),
            rank_feature_importance(table, "exp277_minus_esmfold2", args.min_feature_count),
        ],
        ignore_index=True,
    )
    categories = category_gap_summary(table)

    importance.to_csv(args.out_dir / "anova_feature_importance.csv", index=False)
    categories.to_csv(args.out_dir / "category_gap_summary.csv", index=False)

    for comparison in ["contacts_v2_minus_esmfold2", "exp277_minus_esmfold2"]:
        print(f"\n## {comparison}")
        cols = ["feature_family", "feature", "top_positive_level_or_tag", "direction_effect", "partial_eta2", "p_value"]
        print(importance[importance["comparison"] == comparison].head(12)[cols].to_string(index=False))
    print("\n## category_gap_summary")
    print(categories.to_string(index=False))


if __name__ == "__main__":
    main()
