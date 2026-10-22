# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Fisher LDA screens for exp324 model advantage slices.

The target for each comparison is a binary win/loss label from the per-protein
R-precision-all delta. Small near-ties are dropped with --margin. Features are
coarse biological/structural annotations plus simple protein metadata; model
performance columns are intentionally excluded from X.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Comparison:
    name: str
    left: str
    right: str


COMPARISONS = [
    Comparison("delta_stream_minus_esmfold2", "delta_v2_step78499_n100_all_r", "esmfold2_n100_all_r"),
    Comparison("old_baseline_minus_esmfold2", "exp117_e16_final_step35679_n100_all_r", "esmfold2_n100_all_r"),
    Comparison("new_baseline_minus_esmfold2", "exp277_full_epoch_step266344_n100_all_r", "esmfold2_n100_all_r"),
    Comparison("delta_stream_minus_old_baseline", "delta_v2_step78499_n100_all_r", "exp117_e16_final_step35679_n100_all_r"),
    Comparison("delta_stream_minus_new_baseline", "delta_v2_step78499_n100_all_r", "exp277_full_epoch_step266344_n100_all_r"),
]

NUMERIC_FEATURES = [
    "seq_len",
    "resolution",
    "contacts_emitted",
    "highest_contact_degree",
    "lowest_included_contact_degree",
    "n_cath_domains",
    "uniref100_size",
    "uniref90_size",
    "uniref50_size",
]
CATEGORICAL_FEATURES = [
    "lineage_level_1",
    "lineage_level_2",
    "lineage_status",
    "function_status",
    "structure_status",
    "method",
    "is_synthetic",
    "is_multi_domain",
    "ec_top_classes",
]
MULTITAG_FEATURES = [
    "go_slim_tags",
    "cath_class_tags",
    "cath_topology_tags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/fisher_lda_partial"))
    parser.add_argument("--margin", type=float, default=0.02, help="Drop examples with abs(R-precision delta) <= margin.")
    parser.add_argument("--min-feature-count", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def split_tags(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def build_feature_matrix(df: pd.DataFrame, min_feature_count: int) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    pieces: list[pd.DataFrame] = []

    numeric = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            numeric[f"num:{col}"] = pd.to_numeric(df[col], errors="coerce")
    for col in ["num:uniref100_size", "num:uniref90_size", "num:uniref50_size"]:
        if col in numeric.columns:
            numeric[col] = np.log10(numeric[col].fillna(0) + 1)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype(float)
    medians = numeric.median(numeric_only=True).to_dict()
    numeric = numeric.fillna(medians)
    scaler = StandardScaler()
    if len(numeric.columns):
        numeric = pd.DataFrame(scaler.fit_transform(numeric), index=numeric.index, columns=numeric.columns)
        pieces.append(numeric)

    categorical = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        values = df[col].fillna("unknown").astype(str).replace({"": "unknown"})
        counts = values.value_counts()
        kept = set(counts[counts >= min_feature_count].index)
        values = values.where(values.isin(kept), other="__rare__")
        dummies = pd.get_dummies(values, prefix=f"cat:{col}", dtype=float)
        categorical = pd.concat([categorical, dummies], axis=1)
    if len(categorical.columns):
        pieces.append(categorical)

    multitag = pd.DataFrame(index=df.index)
    for col in MULTITAG_FEATURES:
        if col not in df.columns:
            continue
        tag_lists = df[col].map(split_tags)
        counts: dict[str, int] = {}
        for tags in tag_lists:
            for tag in set(tags):
                counts[tag] = counts.get(tag, 0) + 1
        kept = sorted(tag for tag, count in counts.items() if count >= min_feature_count)
        for tag in kept:
            multitag[f"tag:{col}:{tag}"] = [float(tag in tags) for tags in tag_lists]
    if len(multitag.columns):
        pieces.append(multitag)

    features = pd.concat(pieces, axis=1)
    # Drop constants after subsetting/rare filtering.
    nunique = features.nunique(dropna=False)
    features = features.loc[:, nunique > 1]
    means = features.mean().to_dict()
    stds = features.std(ddof=0).replace(0, 1).to_dict()
    return features.astype(float), means, stds


def fit_one(df: pd.DataFrame, features: pd.DataFrame, comparison: Comparison, margin: float, folds: int, top_k: int, out_dir: Path) -> dict[str, object]:
    needed = [comparison.left, comparison.right]
    available = df.dropna(subset=needed).copy()
    available["advantage"] = available[comparison.left] - available[comparison.right]
    selected = available[available["advantage"].abs() > margin].copy()
    selected["left_better"] = (selected["advantage"] > 0).astype(int)
    X = features.loc[selected.index].to_numpy(dtype=float)
    y = selected["left_better"].to_numpy(dtype=int)

    result: dict[str, object] = {
        "comparison": comparison.name,
        "left": comparison.left,
        "right": comparison.right,
        "margin": margin,
        "n_available": int(len(available)),
        "n_used": int(len(selected)),
        "n_left_better": int(y.sum()),
        "n_right_better": int((1 - y).sum()),
        "mean_advantage_all_available": float(available["advantage"].mean()),
        "median_advantage_all_available": float(available["advantage"].median()),
    }
    if len(np.unique(y)) < 2 or min(result["n_left_better"], result["n_right_better"]) < folds:
        result.update({"cv_auc": None, "cv_balanced_accuracy": None})
        return result

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=324)
    proba = cross_val_predict(lda, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    result["cv_auc"] = float(roc_auc_score(y, proba))
    result["cv_balanced_accuracy"] = float(balanced_accuracy_score(y, pred))

    lda.fit(X, y)
    coefficients = pd.DataFrame({
        "feature": features.columns,
        "coef_left_better": lda.coef_[0],
    })
    coefficients["abs_coef"] = coefficients["coef_left_better"].abs()
    coefficients = coefficients.sort_values("abs_coef", ascending=False)
    coefficients.to_csv(out_dir / f"{comparison.name}.coefficients.csv", index=False)

    scored = selected[["stem", comparison.left, comparison.right, "advantage", "left_better"]].copy()
    scored["lda_score_left_better"] = lda.decision_function(X)
    scored["lda_proba_left_better_cv"] = proba
    scored.to_csv(out_dir / f"{comparison.name}.protein_scores.csv", index=False)

    result["top_positive_features"] = coefficients[coefficients["coef_left_better"] > 0].head(top_k)[["feature", "coef_left_better"]].to_dict("records")
    result["top_negative_features"] = coefficients[coefficients["coef_left_better"] < 0].head(top_k)[["feature", "coef_left_better"]].to_dict("records")
    return result


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.table)
    features, means, stds = build_feature_matrix(df, args.min_feature_count)
    feature_meta = {
        "n_rows": int(len(df)),
        "n_features": int(features.shape[1]),
        "features": list(features.columns),
        "numeric_feature_means_after_scaling": means,
        "numeric_feature_stds_after_scaling": stds,
    }
    (args.out_dir / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2) + "\n")

    summaries = []
    for comparison in COMPARISONS:
        if comparison.left not in df.columns or comparison.right not in df.columns:
            continue
        summaries.append(fit_one(df, features, comparison, args.margin, args.folds, args.top_k, args.out_dir))

    summary_df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in summaries])
    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
