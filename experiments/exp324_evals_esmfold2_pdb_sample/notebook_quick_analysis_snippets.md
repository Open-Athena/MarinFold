# Exp324 10k table — notebook snippets

Use this in a Jupyter notebook from the MarinFold repo root.

```python
import numpy as np
import pandas as pd

TABLE = "experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.parquet"
df = pd.read_parquet(TABLE)
print(df.shape)
```

## 1. Core model columns

```python
R = {
    "ESMFold2 n=100": "esmfold2_n100_all_r",
    "old baseline exp117": "exp117_e16_final_step35679_n100_all_r",
    "delta-stream v2": "delta_v2_step78499_n100_all_r",
    "new baseline exp277": "exp277_full_epoch_step266344_n100_all_r",
}

summary = []
for name, col in R.items():
    x = df[col].dropna()
    summary.append({"model": name, "n": len(x), "mean_R": x.mean(), "sem": x.sem(), "median_R": x.median()})
pd.DataFrame(summary).sort_values("mean_R", ascending=False)
```

## 2. Pairwise deltas

```python
df = df.copy()
df["delta_stream_minus_esmfold2"] = df[R["delta-stream v2"]] - df[R["ESMFold2 n=100"]]
df["exp117_minus_esmfold2"] = df[R["old baseline exp117"]] - df[R["ESMFold2 n=100"]]
df["exp277_minus_esmfold2"] = df[R["new baseline exp277"]] - df[R["ESMFold2 n=100"]]
df["delta_stream_minus_exp117"] = df[R["delta-stream v2"]] - df[R["old baseline exp117"]]
df["delta_stream_minus_exp277"] = df[R["delta-stream v2"]] - df[R["new baseline exp277"]]

deltas = [c for c in df.columns if "_minus_" in c]
df[deltas].describe().T
```

## 3. Quick grouped analyses: where do we win?

Use this for any categorical column (`lineage_level_1`, `lineage_level_2`, `structure_status`, etc.).

```python
def grouped_delta(table, group_col, delta_col, min_n=30):
    out = (
        table.dropna(subset=[group_col, delta_col])
        .groupby(group_col, dropna=False)[delta_col]
        .agg(n="size", mean="mean", median="median", sem="sem")
        .reset_index()
    )
    out = out[out.n >= min_n].sort_values("mean", ascending=False)
    return out

grouped_delta(df, "lineage_level_1", "delta_stream_minus_esmfold2", min_n=30)
grouped_delta(df, "lineage_level_2", "delta_stream_minus_esmfold2", min_n=50).head(20)
grouped_delta(df, "structure_status", "delta_stream_minus_exp117", min_n=30)
```

## 4. Multi-tag group analysis for GO-slim / CATH

```python
def explode_tags(table, tag_col, delta_col, min_n=50):
    tmp = table[["stem", tag_col, delta_col]].dropna(subset=[delta_col]).copy()
    tmp[tag_col] = tmp[tag_col].fillna("").astype(str).str.split(";")
    tmp = tmp.explode(tag_col)
    tmp[tag_col] = tmp[tag_col].str.strip()
    tmp = tmp[tmp[tag_col] != ""]
    out = (
        tmp.groupby(tag_col)[delta_col]
        .agg(n="size", mean="mean", median="median", sem="sem")
        .reset_index()
    )
    return out[out.n >= min_n].sort_values("mean", ascending=False)

explode_tags(df, "go_slim_tags", "delta_stream_minus_esmfold2", min_n=50).head(20)
explode_tags(df, "cath_topology_tags", "delta_stream_minus_exp117", min_n=30).head(20)
```

## 5. UniRef depth proxy slices

`uniref50_size` is a cheap homolog-family-size proxy, not true MSA depth.

```python
df["uniref50_tier"] = pd.cut(
    df["uniref50_size"],
    bins=[-1, 9, 99, 999, 9999, np.inf],
    labels=["<=9", "10-99", "100-999", "1k-9999", "10k+"],
)

grouped_delta(df, "uniref50_tier", "delta_stream_minus_esmfold2", min_n=1)
grouped_delta(df, "uniref50_tier", "delta_stream_minus_exp117", min_n=1)
```

## 6. Fisher LDA: combinations of dimensions

This reproduces the LDA-style screen. It treats “left model better by > margin” as class 1 and “right model better by > margin” as class 0.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

NUMERIC = [
    "seq_len", "resolution", "contacts_emitted", "highest_contact_degree",
    "lowest_included_contact_degree", "n_cath_domains",
    "uniref100_size", "uniref90_size", "uniref50_size",
]
CATEGORICAL = [
    "lineage_level_1", "lineage_level_2", "lineage_status", "function_status",
    "structure_status", "method", "is_synthetic", "is_multi_domain", "ec_top_classes",
]
TAG_COLS = ["go_slim_tags", "cath_class_tags", "cath_topology_tags"]

def split_tags(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def make_X(data, min_count=50):
    pieces = []
    num = data[[c for c in NUMERIC if c in data.columns]].apply(pd.to_numeric, errors="coerce")
    for c in ["uniref100_size", "uniref90_size", "uniref50_size"]:
        if c in num:
            num[c] = np.log10(num[c].fillna(0) + 1)
    num = num.replace([np.inf, -np.inf], np.nan).fillna(num.median()).astype(float)
    if len(num.columns):
        pieces.append(pd.DataFrame(StandardScaler().fit_transform(num), columns=[f"num:{c}" for c in num.columns], index=data.index))

    for c in CATEGORICAL:
        if c not in data.columns:
            continue
        values = data[c].fillna("unknown").astype(str).replace({"": "unknown"})
        keep = values.value_counts()[lambda s: s >= min_count].index
        values = values.where(values.isin(keep), "__rare__")
        pieces.append(pd.get_dummies(values, prefix=f"cat:{c}", dtype=float))

    tag_frames = []
    for c in TAG_COLS:
        tags = data[c].map(split_tags) if c in data.columns else pd.Series([[]] * len(data), index=data.index)
        counts = {}
        for ts in tags:
            for t in set(ts): counts[t] = counts.get(t, 0) + 1
        for t, n in counts.items():
            if n >= min_count:
                tag_frames.append(pd.Series([float(t in ts) for ts in tags], index=data.index, name=f"tag:{c}:{t}"))
    if tag_frames:
        pieces.append(pd.concat(tag_frames, axis=1))
    X = pd.concat(pieces, axis=1)
    return X.loc[:, X.nunique() > 1]

def lda_screen(data, left_col, right_col, margin=0.02, min_count=50, top=20):
    sub = data.dropna(subset=[left_col, right_col]).copy()
    sub["adv"] = sub[left_col] - sub[right_col]
    sub = sub[sub.adv.abs() > margin]
    y = (sub.adv > 0).astype(int).to_numpy()
    Xdf = make_X(sub, min_count=min_count)
    X = Xdf.to_numpy(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=324)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    clf.fit(X, y)
    coef = pd.DataFrame({"feature": Xdf.columns, "coef_left_better": clf.coef_[0]})
    coef["abs_coef"] = coef.coef_left_better.abs()
    coef = coef.sort_values("abs_coef", ascending=False)
    print({
        "n": len(sub), "left_better": int(y.sum()), "right_better": int((1-y).sum()),
        "cv_auc": roc_auc_score(y, p),
        "cv_balanced_acc": balanced_accuracy_score(y, p >= 0.5),
        "mean_delta": sub.adv.mean(),
    })
    return coef.head(top), coef[coef.coef_left_better > 0].head(top), coef[coef.coef_left_better < 0].head(top)

all_top, left_features, right_features = lda_screen(
    df,
    left_col=R["delta-stream v2"],
    right_col=R["ESMFold2 n=100"],
)
left_features, right_features
```

## 7. Shallow decision tree for readable feature combinations

```python
from sklearn.tree import DecisionTreeClassifier, export_text

left_col = R["delta-stream v2"]
right_col = R["ESMFold2 n=100"]
sub = df.dropna(subset=[left_col, right_col]).copy()
sub["adv"] = sub[left_col] - sub[right_col]
sub = sub[sub.adv.abs() > 0.02]
y = (sub.adv > 0).astype(int)
Xdf = make_X(sub, min_count=50)

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=80, class_weight="balanced", random_state=324)
tree.fit(Xdf, y)
print(export_text(tree, feature_names=list(Xdf.columns)))

leaf = tree.apply(Xdf)
leaf_stats = (
    sub.assign(left_better=y.values, leaf=leaf)
    .groupby("leaf")
    .agg(n=("left_better", "size"), left_rate=("left_better", "mean"), mean_delta=("adv", "mean"))
    .sort_values(["left_rate", "n"], ascending=[False, False])
)
leaf_stats.head(10)
```

## 9. ANOVA / OLS: higher-performing groups vs ESMFold2

This models the continuous paired delta and asks which groups/tags have positive adjusted effects after controlling for simple confounders.

```python
# Optional dependency in notebook env:
# %pip install statsmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

TABLE = "experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.parquet"
df = pd.read_parquet(TABLE).copy()

# Paired performance deltas: positive means MarinFold checkpoint beats ESMFold2.
df["delta_stream_minus_esmfold2"] = df["delta_v2_step78499_n100_all_r"] - df["esmfold2_n100_all_r"]
df["exp277_minus_esmfold2"] = df["exp277_full_epoch_step266344_n100_all_r"] - df["esmfold2_n100_all_r"]

# Simple covariates to adjust for. contacts_emitted is a strong proxy for contact-map difficulty/density.
df["log_seq_len"] = np.log(df["seq_len"].clip(lower=1))
df["log_contacts_emitted"] = np.log1p(df["contacts_emitted"].clip(lower=0))
df["log_uniref50"] = np.log10(df["uniref50_size"].fillna(0) + 1)
df["resolution_filled"] = df["resolution"].fillna(df["resolution"].median())

BASE_COVARS = "log_seq_len + log_contacts_emitted + resolution_filled + log_uniref50"
```

### 9a. Categorical ANOVA: taxonomy / method / status fields

```python
def anova_categorical(table, delta_col, feature, min_n=30):
    """Type-II ANOVA for one categorical feature, plus adjusted group effects."""
    work = table.dropna(subset=[delta_col, feature, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]).copy()
    counts = work[feature].fillna("unknown").astype(str).value_counts()
    keep = set(counts[counts >= min_n].index)
    work[feature] = work[feature].fillna("unknown").astype(str).where(lambda s: s.isin(keep), "__rare__")

    formula = f"{delta_col} ~ C({feature}) + {BASE_COVARS}"
    model = smf.ols(formula, data=work).fit()
    anova = sm.stats.anova_lm(model, typ=2).loc[[f"C({feature})"]]

    # Human-readable group means/effects. raw_mean is easiest to explain;
    # adjusted_effect is coefficient relative to the reference category.
    groups = (
        work.groupby(feature)[delta_col]
        .agg(n="size", raw_mean="mean", median="median", sem="sem")
        .reset_index()
        .sort_values("raw_mean", ascending=False)
    )
    coefs = model.params.filter(like=f"C({feature})[T.")
    coef_rows = []
    for name, value in coefs.items():
        level = name.split("[T.", 1)[1].rstrip("]")
        coef_rows.append({feature: level, "adjusted_effect_vs_ref": value, "p_value_vs_ref": model.pvalues[name]})
    coef_df = pd.DataFrame(coef_rows)
    groups = groups.merge(coef_df, on=feature, how="left")
    return anova, groups

for delta_col in ["delta_stream_minus_esmfold2", "exp277_minus_esmfold2"]:
    print("\n###", delta_col)
    for feature in ["lineage_level_1", "lineage_level_2", "method", "structure_status", "function_status"]:
        anova, groups = anova_categorical(df, delta_col, feature, min_n=50)
        print("\n", feature)
        display(anova)
        display(groups.head(15))  # highest-performing groups for this checkpoint vs ESMFold2
```

### 9b. Multi-tag ANOVA: GO-slim and CATH tags

This tests each tag as a binary feature, adjusted for the same covariates. Positive `adjusted_effect` means proteins with that tag have a higher MarinFold-minus-ESMFold2 delta.

```python
def tag_memberships(series):
    return series.fillna("").astype(str).map(lambda x: {t.strip() for t in x.split(";") if t.strip()})

def anova_tags(table, delta_col, tag_col, min_n=50):
    tag_sets = tag_memberships(table[tag_col])
    counts = {}
    for tags in tag_sets:
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    tags = sorted(tag for tag, n in counts.items() if n >= min_n)

    rows = []
    for tag in tags:
        work = table.dropna(subset=[delta_col, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]).copy()
        work["has_tag"] = tag_sets.loc[work.index].map(lambda xs: tag in xs).astype(int)
        if work["has_tag"].nunique() < 2:
            continue
        model = smf.ols(f"{delta_col} ~ has_tag + {BASE_COVARS}", data=work).fit()
        yes = work.loc[work.has_tag == 1, delta_col]
        no = work.loc[work.has_tag == 0, delta_col]
        rows.append({
            "tag_col": tag_col,
            "tag": tag,
            "n_with_tag": int(work.has_tag.sum()),
            "raw_mean_with_tag": yes.mean(),
            "raw_mean_without_tag": no.mean(),
            "raw_delta_with_minus_without": yes.mean() - no.mean(),
            "adjusted_effect": model.params["has_tag"],
            "p_value": model.pvalues["has_tag"],
        })
    return pd.DataFrame(rows).sort_values("adjusted_effect", ascending=False)

for delta_col in ["delta_stream_minus_esmfold2", "exp277_minus_esmfold2"]:
    print("\n###", delta_col)
    for tag_col in ["go_slim_tags", "cath_class_tags", "cath_topology_tags"]:
        result = anova_tags(df, delta_col, tag_col, min_n=50)
        print("\nPositive adjusted effects:", tag_col)
        display(result.head(20))
        print("\nNegative adjusted effects:", tag_col)
        display(result.tail(20).sort_values("adjusted_effect"))
```

### 9c. Compact table: top positive groups/tags

```python
def top_positive_anova_hits(table, delta_col):
    hits = []

    # Categorical fields: use raw group means for ranking, with ANOVA p-value for the whole feature.
    for feature in ["lineage_level_1", "lineage_level_2", "method", "structure_status", "function_status"]:
        anova, groups = anova_categorical(table, delta_col, feature, min_n=50)
        feature_p = float(anova["PR(>F)"].iloc[0])
        for _, row in groups.head(10).iterrows():
            hits.append({
                "feature_type": feature,
                "level_or_tag": row[feature],
                "n": int(row["n"]),
                "mean_delta": row["raw_mean"],
                "sem": row["sem"],
                "test_p_value": feature_p,
            })

    # Tags: use adjusted binary effect.
    for tag_col in ["go_slim_tags", "cath_class_tags", "cath_topology_tags"]:
        tags = anova_tags(table, delta_col, tag_col, min_n=50).head(15)
        for _, row in tags.iterrows():
            hits.append({
                "feature_type": tag_col,
                "level_or_tag": row["tag"],
                "n": int(row["n_with_tag"]),
                "mean_delta": row["raw_mean_with_tag"],
                "sem": np.nan,
                "adjusted_effect": row["adjusted_effect"],
                "test_p_value": row["p_value"],
            })

    return pd.DataFrame(hits).sort_values(["mean_delta", "adjusted_effect"], ascending=False)

delta_hits = top_positive_anova_hits(df, "delta_stream_minus_esmfold2")
exp277_hits = top_positive_anova_hits(df, "exp277_minus_esmfold2")

display(delta_hits.head(30))
display(exp277_hits.head(30))
```

## 10. Rank the most important features first (ANOVA/OLS effect size)

Use this when you want a single table ordered by feature importance across feature families. Importance here is **partial eta-squared** from an adjusted OLS/ANOVA model. Larger values mean the feature explains more variance in the paired model-vs-ESMFold2 delta after the covariates.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assumes df and BASE_COVARS from section 9 already exist.
# Positive direction means the feature/tag is associated with higher MarinFold-minus-ESMFold2 R-precision.

def partial_eta_squared(anova_row, residual_ss):
    ss = float(anova_row["sum_sq"])
    return ss / (ss + float(residual_ss))


def rank_categorical_features(table, delta_col, features, min_n=50):
    rows = []
    for feature in features:
        work = table.dropna(
            subset=[delta_col, feature, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]
        ).copy()
        counts = work[feature].fillna("unknown").astype(str).value_counts()
        keep = set(counts[counts >= min_n].index)
        work[feature] = work[feature].fillna("unknown").astype(str).where(lambda s: s.isin(keep), "__rare__")
        if work[feature].nunique() < 2:
            continue

        model = smf.ols(f"{delta_col} ~ C({feature}) + {BASE_COVARS}", data=work).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        row = anova.loc[f"C({feature})"]
        resid_ss = anova.loc["Residual", "sum_sq"]

        group_means = work.groupby(feature)[delta_col].mean().sort_values(ascending=False)
        top_level = group_means.index[0]
        bottom_level = group_means.index[-1]
        direction = group_means.iloc[0] - group_means.iloc[-1]

        rows.append({
            "feature_family": "categorical",
            "feature": feature,
            "top_positive_level_or_tag": top_level,
            "top_positive_mean_delta": group_means.iloc[0],
            "bottom_level_or_tag": bottom_level,
            "bottom_mean_delta": group_means.iloc[-1],
            "direction_effect": direction,
            "n": len(work),
            "F": float(row["F"]),
            "p_value": float(row["PR(>F)"]),
            "partial_eta2": partial_eta_squared(row, resid_ss),
        })
    return pd.DataFrame(rows)


def rank_numeric_features(table, delta_col, features):
    rows = []
    for feature in features:
        work = table.dropna(
            subset=[delta_col, feature, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]
        ).copy()
        if len(work) < 100 or work[feature].nunique() < 5:
            continue
        z = (work[feature] - work[feature].mean()) / work[feature].std()
        work["z_feature"] = z.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["z_feature"])

        model = smf.ols(f"{delta_col} ~ z_feature + {BASE_COVARS}", data=work).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        row = anova.loc["z_feature"]
        resid_ss = anova.loc["Residual", "sum_sq"]
        coef = model.params["z_feature"]

        rows.append({
            "feature_family": "numeric",
            "feature": feature,
            "top_positive_level_or_tag": "higher values" if coef > 0 else "lower values",
            "top_positive_mean_delta": np.nan,
            "bottom_level_or_tag": "lower values" if coef > 0 else "higher values",
            "bottom_mean_delta": np.nan,
            "direction_effect": coef,  # effect per +1 SD
            "n": len(work),
            "F": float(row["F"]),
            "p_value": float(row["PR(>F)"]),
            "partial_eta2": partial_eta_squared(row, resid_ss),
        })
    return pd.DataFrame(rows)


def rank_tag_features(table, delta_col, tag_cols, min_n=50):
    rows = []
    for tag_col in tag_cols:
        tag_sets = tag_memberships(table[tag_col])
        counts = {}
        for tags in tag_sets:
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        tags = sorted(tag for tag, n in counts.items() if n >= min_n)

        for tag in tags:
            work = table.dropna(
                subset=[delta_col, "log_seq_len", "log_contacts_emitted", "resolution_filled", "log_uniref50"]
            ).copy()
            work["has_tag"] = tag_sets.loc[work.index].map(lambda xs: tag in xs).astype(int)
            if work["has_tag"].nunique() < 2:
                continue

            model = smf.ols(f"{delta_col} ~ has_tag + {BASE_COVARS}", data=work).fit()
            anova = sm.stats.anova_lm(model, typ=2)
            row = anova.loc["has_tag"]
            resid_ss = anova.loc["Residual", "sum_sq"]
            yes = work.loc[work.has_tag == 1, delta_col]
            no = work.loc[work.has_tag == 0, delta_col]
            coef = model.params["has_tag"]

            rows.append({
                "feature_family": tag_col,
                "feature": tag_col,
                "top_positive_level_or_tag": tag,
                "top_positive_mean_delta": yes.mean(),
                "bottom_level_or_tag": f"not {tag}",
                "bottom_mean_delta": no.mean(),
                "direction_effect": coef,
                "n": int(work.has_tag.sum()),
                "F": float(row["F"]),
                "p_value": float(row["PR(>F)"]),
                "partial_eta2": partial_eta_squared(row, resid_ss),
            })
    return pd.DataFrame(rows)


def rank_all_feature_importance(table, delta_col, min_n=50):
    categorical = [
        "lineage_level_1",
        "lineage_level_2",
        "method",
        "structure_status",
        "function_status",
        "lineage_status",
        "is_synthetic",
    ]
    numeric = [
        "seq_len",
        "contacts_emitted",
        "highest_contact_degree",
        "lowest_included_contact_degree",
        "resolution",
        "uniref100_size",
        "uniref90_size",
        "uniref50_size",
    ]
    tag_cols = ["go_slim_tags", "cath_class_tags", "cath_topology_tags"]

    parts = [
        rank_categorical_features(table, delta_col, categorical, min_n=min_n),
        rank_numeric_features(table, delta_col, numeric),
        rank_tag_features(table, delta_col, tag_cols, min_n=min_n),
    ]
    out = pd.concat(parts, ignore_index=True)
    out["abs_direction_effect"] = out["direction_effect"].abs()
    return out.sort_values(["partial_eta2", "abs_direction_effect"], ascending=False)

# Most important adjusted features/tags for each checkpoint vs ESMFold2.
delta_importance = rank_all_feature_importance(df, "delta_stream_minus_esmfold2", min_n=50)
exp277_importance = rank_all_feature_importance(df, "exp277_minus_esmfold2", min_n=50)

display(delta_importance.head(30))
display(exp277_importance.head(30))
```

Quick side-by-side table of the top features:

```python
side_by_side = (
    delta_importance.head(30)[["feature_family", "feature", "top_positive_level_or_tag", "direction_effect", "partial_eta2", "p_value"]]
    .rename(columns={
        "top_positive_level_or_tag": "delta_top_level_or_tag",
        "direction_effect": "delta_direction_effect",
        "partial_eta2": "delta_partial_eta2",
        "p_value": "delta_p",
    })
    .merge(
        exp277_importance.head(30)[["feature_family", "feature", "top_positive_level_or_tag", "direction_effect", "partial_eta2", "p_value"]]
        .rename(columns={
            "top_positive_level_or_tag": "exp277_top_level_or_tag",
            "direction_effect": "exp277_direction_effect",
            "partial_eta2": "exp277_partial_eta2",
            "p_value": "exp277_p",
        }),
        on=["feature_family", "feature"],
        how="outer",
    )
)
display(side_by_side)
```

## 11. R-precision comparisons for selected protein groups

These groups are interpretable slices where MarinFold tends to **close the ESMFold2 gap**. Most still have ESMFold2 ahead on mean R-precision; use the gap and win-fraction columns to distinguish absolute wins from smaller losses.

```python
MF_DELTA = "delta_v2_step78499_n100_all_r"
MF_277 = "exp277_full_epoch_step266344_n100_all_r"
ESM = "esmfold2_n100_all_r"

# Positive means MarinFold beats ESMFold2 on that protein.
df["delta_stream_minus_esmfold2"] = df[MF_DELTA] - df[ESM]
df["exp277_minus_esmfold2"] = df[MF_277] - df[ESM]

def has_tag(series, tag):
    return series.fillna("").astype(str).str.split(";").map(lambda xs: tag in {x.strip() for x in xs if x.strip()})

CATEGORY_MASKS = {
    "All proteins": pd.Series(True, index=df.index),
    "Synthetic": df["is_synthetic"].fillna(False).astype(bool),
    "Solution NMR": df["method"].eq("SOLUTION NMR"),
    "Riboviria": df["lineage_level_2"].eq("Riboviria"),
    "Duplodnaviria": df["lineage_level_2"].eq("Duplodnaviria"),
    "Varidnaviria": df["lineage_level_2"].eq("Varidnaviria"),
    "CATH topo 1.20.5": has_tag(df["cath_topology_tags"], "1.20.5"),
    "GO: cell wall org/biogenesis": has_tag(df["go_slim_tags"], "cell wall organization or biogenesis"),
}

def summarize_selected_categories(table, masks):
    rows = []
    for label, mask in masks.items():
        w = table.loc[mask].dropna(subset=[ESM, MF_DELTA, MF_277]).copy()
        if len(w) == 0:
            continue
        rows.append({
            "category": label,
            "n": len(w),
            "esmfold2_r": w[ESM].mean(),
            "delta_stream_r": w[MF_DELTA].mean(),
            "exp277_r": w[MF_277].mean(),
            "delta_stream_gap": w[MF_DELTA].mean() - w[ESM].mean(),
            "exp277_gap": w[MF_277].mean() - w[ESM].mean(),
            "delta_stream_win_frac": (w[MF_DELTA] > w[ESM]).mean(),
            "exp277_win_frac": (w[MF_277] > w[ESM]).mean(),
        })
    return pd.DataFrame(rows)

category_summary = summarize_selected_categories(df, CATEGORY_MASKS)
display(category_summary)
```

Gap / win-fraction comparison table:

```python
comparison_table = category_summary[[
    "category", "n",
    "esmfold2_r", "delta_stream_r", "exp277_r",
    "delta_stream_gap", "delta_stream_win_frac",
    "exp277_gap", "exp277_win_frac",
]].copy()
comparison_table["delta_stream_win_frac"] *= 100
comparison_table["exp277_win_frac"] *= 100
comparison_table = comparison_table.sort_values("delta_stream_gap", ascending=False)
display(comparison_table)
```
