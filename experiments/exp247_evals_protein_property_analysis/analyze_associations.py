# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 2 — which protein properties move which predictor, and by how much.

Three passes over the same feature matrix, in increasing order of how much they
assume:

**Univariate.** Spearman ρ between every numeric feature and every predictor's
per-protein R-precision, with Benjamini-Hochberg q-values over the whole
(feature × predictor) grid, computed on the pooled natural set and again
separately on eval-val and eval-test. A correlation that flips sign between the
two sets is a sample artefact, and the split is the cheapest way to see it.

**Multivariate.** Per predictor, a ridge model and a gradient-boosted one on the
same standardised features, scored by 5-fold cross-validated R². The point is not
prediction — it is how much of the per-protein spread is explainable at all, and
which features carry it once they compete. Permutation importance is computed on
held-out folds so a feature cannot look important by memorising 314 rows.

**Partial.** For the biology features (kingdom, localisation, function, domain
count), the partial Spearman controlling for length, contact order and training
support — H4 says most of their apparent effect is those three in disguise.

Missing values are median-imputed **inside each CV fold** and a missingness
indicator is added, so "this protein has no UniProt entry" can itself be a
feature rather than a silent bias.

    uv run python analyze_associations.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import upstream as U

DATA = U.DATA
FEATURES = DATA / "protein_features.csv"
ASSOCIATIONS = DATA / "associations.csv"
MODELS = DATA / "model_performance.csv"
IMPORTANCE = DATA / "feature_importance.csv"
PARTIAL = DATA / "partial_associations.csv"
SUMMARY = DATA / "analysis_summary.json"

#: Columns that identify a protein rather than describe it.
IDENTIFIERS = {
    "stem", "eval_set", "pdb_id", "chain_id", "entity_id", "entry_id", "sequence",
    "title", "source_organisms", "domain_annotations", "subcellular_location",
    "uniprot_keywords", "protein_existence", "deposit_date", "exp199_stratum",
    "kingdom", "structure_error",
}
#: The three axes H4 asks the biology features to survive.
CONTROLS = ("length", "relative_contact_order", "knn_best_identity")
BIOLOGY = ("is_viral", "is_membrane", "is_secreted", "is_cytoplasmic", "is_nuclear",
           "n_uniprot_domains", "n_pfam", "n_cath", "has_ec", "n_transmembrane",
           "has_signal_peptide", "kingdom_eukaryote", "kingdom_bacteria")
SEED = 247
FOLDS = 5


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Feature matrix, per-protein scores (all-range R), and the feature names."""
    features = pd.read_csv(FEATURES)
    # One-hot the one categorical worth keeping; the rest are free text.
    for kingdom in ("bacteria", "eukaryote", "archaea", "virus"):
        features[f"kingdom_{kingdom}"] = (features.kingdom == kingdom).astype(int)

    scores = pd.read_csv(U.PER_PROTEIN)
    scores = scores[(scores["range"] == "all") & (scores["cut"] == "R")]
    wide = scores.pivot_table(index="stem", columns="predictor", values="precision")
    wide = wide.loc[[s for s in features.stem if s in wide.index]]

    numeric = [
        c for c in features.columns
        if c not in IDENTIFIERS and pd.api.types.is_numeric_dtype(features[c])
        and features[c].notna().sum() >= 0.5 * len(features)
        and features[c].nunique(dropna=True) > 1
    ]
    return features.set_index("stem"), wide, numeric


def associations(features: pd.DataFrame, scores: pd.DataFrame,
                 names: list[str]) -> pd.DataFrame:
    """Spearman ρ per (feature, predictor, subset), with BH q-values."""
    rows = []
    subsets = {"natural (pooled)": features.index,
               "eval-val": features.index[features.eval_set == "eval-val"],
               "eval-test": features.index[features.eval_set == "eval-test"]}
    for subset, index in subsets.items():
        for predictor in U.PREDICTORS:
            if predictor not in scores:
                continue
            y = scores.loc[index, predictor]
            for name in names:
                x = features.loc[index, name]
                mask = x.notna() & y.notna()
                if mask.sum() < 30 or x[mask].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(x[mask], y[mask])
                rows.append({"subset": subset, "predictor": predictor,
                             "feature": name, "n": int(mask.sum()),
                             "rho": float(rho), "p": float(p)})
    frame = pd.DataFrame(rows)
    for subset, part in frame.groupby("subset"):
        order = part.p.rank(method="first")
        frame.loc[part.index, "q"] = (part.p * len(part) / order).clip(upper=1.0)
    return frame.sort_values(["subset", "predictor", "rho"])


def model_pipelines() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 30))),
        ]),
        "gbm": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(
                max_depth=3, max_iter=250, learning_rate=0.06, random_state=SEED)),
        ]),
    }


def models(features: pd.DataFrame, scores: pd.DataFrame,
           names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validated R² per predictor, and permutation importance."""
    X = features[names]
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    performance, importances = [], []
    for predictor in U.PREDICTORS:
        if predictor not in scores:
            continue
        y = scores[predictor].loc[X.index]
        mask = y.notna()
        for kind, pipeline in model_pipelines().items():
            r2 = cross_val_score(pipeline, X[mask], y[mask], cv=splitter, scoring="r2")
            performance.append({
                "predictor": predictor, "model": kind, "n": int(mask.sum()),
                "cv_r2_mean": float(r2.mean()), "cv_r2_std": float(r2.std()),
                "variance_explained": float(max(0.0, r2.mean())),
            })
            if kind != "gbm":
                continue
            # Importance on held-out folds only.
            fold_importances = []
            for train, test in splitter.split(X[mask]):
                fitted = pipeline.fit(X[mask].iloc[train], y[mask].iloc[train])
                result = permutation_importance(
                    fitted, X[mask].iloc[test], y[mask].iloc[test],
                    n_repeats=8, random_state=SEED, scoring="r2")
                fold_importances.append(result.importances_mean)
            mean = np.mean(fold_importances, axis=0)
            for name, value in zip(names, mean, strict=True):
                importances.append({"predictor": predictor, "feature": name,
                                    "importance": float(value)})
    return pd.DataFrame(performance), pd.DataFrame(importances)


def partial_associations(features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Biology features after regressing out size, contact order and homology."""
    controls = [c for c in CONTROLS if c in features]
    rows = []
    for predictor in U.PREDICTORS:
        if predictor not in scores:
            continue
        y = scores[predictor].loc[features.index]
        for name in BIOLOGY:
            if name not in features:
                continue
            frame = features[[*controls, name]].join(y.rename("y")).dropna()
            if len(frame) < 30 or frame[name].nunique() < 2:
                continue
            raw, _ = stats.spearmanr(frame[name], frame.y)
            # Partial Spearman: rank everything, then correlate the residuals of
            # feature and score after linear regression on the ranked controls.
            ranked = frame.rank()
            design = np.column_stack([np.ones(len(ranked)), ranked[controls].to_numpy()])
            def residual(values: np.ndarray) -> np.ndarray:
                beta, *_ = np.linalg.lstsq(design, values, rcond=None)
                return values - design @ beta
            rho, p = stats.spearmanr(residual(ranked[name].to_numpy()),
                                     residual(ranked.y.to_numpy()))
            rows.append({"predictor": predictor, "feature": name, "n": int(len(frame)),
                         "rho_raw": float(raw), "rho_partial": float(rho),
                         "p_partial": float(p),
                         "controls": ",".join(controls)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    features, scores, names = load()
    print(f"[analysis] {len(features)} proteins x {len(names)} usable features", flush=True)

    assoc = associations(features, scores, names)
    assoc.to_csv(ASSOCIATIONS, index=False)
    performance, importance = models(features, scores, names)
    performance.to_csv(MODELS, index=False)
    importance.to_csv(IMPORTANCE, index=False)
    partial = partial_associations(features, scores)
    partial.to_csv(PARTIAL, index=False)

    pooled = assoc[assoc.subset == "natural (pooled)"]
    top = {
        predictor: part.reindex(part.rho.abs().sort_values(ascending=False).index)
                       .head(6)[["feature", "rho", "q"]].round(3).to_dict("records")
        for predictor, part in pooled.groupby("predictor")
    }
    summary = {
        "n_proteins": int(len(features)),
        "n_features": len(names),
        "cv_r2": performance.pivot_table(index="predictor", columns="model",
                                         values="cv_r2_mean").round(3).to_dict(),
        "top_associations": top,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cv_r2"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
