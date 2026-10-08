"""Put AF3 sampling on the same TM-score scale and population as the predictors.

No inference or structural scoring: reuse archived per-protein scores, select
AF3 candidates by saved confidence, and bootstrap proteins within each MSA bin.
The 1,000 samples never count as 1,000 independent biological examples.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import EXP311, FIG250, REPO, Sources, bootstrap, sha256

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def score_rows(raw: pd.DataFrame, method: str, column: str = "tm_score") -> pd.DataFrame:
    """Retain the exact source cell behind each protein score."""
    out = raw[["stem", column, "source", "source_row"]].rename(columns={column: "tm_score"}).copy()
    out["method"] = method
    out["source_column"] = column
    out["uses_ground_truth"] = method == "oracle"
    out["series"] = "archived_predictor"
    return out


def select_samples(samples: pd.DataFrame, budget: int, selector: str) -> pd.DataFrame:
    """Select within a fixed seed prefix, breaking exact ties by earliest seed."""
    parts = []
    for stem, group in samples.groupby("stem", sort=True):
        group = group.sort_values("seed").head(budget)
        if len(group) != budget or group.seed.duplicated().any():
            raise ValueError(f"Incomplete or duplicate AF3 pool for {stem}, N={budget}")
        selected = group.sort_values([selector, "seed"], ascending=[False, True]).iloc[[0]]
        method = f"af3_{budget}_{selector}"
        row = score_rows(selected, method)
        row["series"] = "extended_af3"
        row["budget"] = budget
        row["selected_seed"] = selected.seed.to_numpy()
        row["selector"] = selector
        row["uses_ground_truth"] = selector == "tm_score"
        parts.append(row)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """Write source-traced comparisons and cached protein-level uncertainty."""
    sources = Sources()
    targets = sources.csv(DATA / "targets.csv")
    existing = sources.csv(DATA / "figure_rows.csv")
    stems = set(existing.loc[(existing.figure == "01_predictors") & (existing.designed == 0), "stem"])
    if len(stems) != 305:
        raise ValueError("Expected the existing 305 matched natural proteins")
    targets = targets[targets.stem.isin(stems)].copy()
    targets = targets.rename(columns={"source": "target_source", "source_row": "target_source_row"})
    targets = targets[["stem", "eval_set", "msa_depth", "tier", "L", "target_source", "target_source_row"]]
    archived = sources.csv(REPO / FIG250 / "3_structure_accuracy/per_target.csv", round_trip=True)
    archived = archived.assign(stem=archived.target_id)
    archived = archived[(archived.status == "ok") & archived.stem.isin(stems)]
    parts = [score_rows(archived[archived.arm == arm], method) for arm, method in [
        ("esmfold", "esmfold"), ("esmfold2", "esmfold2"),
        ("protenix_v2_msa", "protenix_msa"), ("protenix_v2_single_seq", "protenix_ss"),
        ("oracle", "oracle")]]
    for method in ("af2", "af3", "boltz2"):
        raw = sources.csv(DATA / f"{method}_structure_metrics.csv", round_trip=True)
        parts.append(score_rows(raw[raw.stem.isin(stems)], method))

    # Match Figure 05's current 248B-token model, fixed top-L cut and confidence
    # selection. Exp250's mf_L arm is an older checkpoint and must not enter.
    validation = sources.csv(REPO / EXP311 / "per_target_selection.csv", round_trip=True)
    validation = validation[(validation.metric == "tm_score") & validation.stem.isin(stems)]
    parts.append(score_rows(validation, "marinfold_helico", "top_l_confidence"))
    test = sources.csv(DATA / "helico_folding_samples.csv", round_trip=True)
    test = test[(test.arm == "top_L") & test.stem.isin(stems)]
    test = test.sort_values(["stem", "ranking_score", "sample_idx"], ascending=[True, False, True]).drop_duplicates("stem")
    parts.append(score_rows(test, "marinfold_helico"))
    baseline = pd.concat(parts, ignore_index=True)
    for method, group in baseline.groupby("method"):
        if set(group.stem) != stems or group.stem.duplicated().any():
            raise ValueError(f"Predictor {method} does not cover exactly the same 305 proteins")

    samples = sources.csv(DATA / "af3_sampling_samples.csv", round_trip=True)
    low_stems = set(targets.loc[targets.tier == "<10", "stem"])
    if len(low_stems) != 5 or set(samples.stem) != low_stems:
        raise ValueError("Extended sampling must cover exactly the five depth <10 proteins")
    for stem, group in samples.groupby("stem"):
        if sorted(group.seed.tolist()) != list(range(10000, 11000)):
            raise ValueError(f"Expected the frozen 1,000-seed pool for {stem}")
    extra = [select_samples(samples, n, selector) for n in (100, 1000)
             for selector in ("ranking_score", "ptm", "tm_score")]
    rows = pd.concat([baseline, *extra], ignore_index=True).merge(targets, on="stem", validate="many_to_one")
    if rows.duplicated(["stem", "method"]).any() or not np.isfinite(rows.tm_score).all():
        raise ValueError("Duplicate or nonfinite context scores")
    rows = rows.sort_values(["method", "stem"]).reset_index(drop=True)
    rows.index.name = "context_row"
    rows = rows.reset_index()
    summaries = []
    for (method, tier), group in rows.groupby(["method", "tier"], sort=True):
        mean, lo, hi = bootstrap(group.tm_score.to_numpy())
        summaries.append(dict(method=method, tier=tier, n=len(group), mean=mean, lo=lo, hi=hi,
                              hits_tm_80=int((group.tm_score >= 0.8).sum()),
                              uses_ground_truth=bool(group.uses_ground_truth.iloc[0]),
                              context_rows=json.dumps(group.context_row.tolist()),
                              stems=json.dumps(group.stem.tolist())))
    summary = pd.DataFrame(summaries)
    low = rows[rows.tier == "<10"]
    base = low[low.method == "af3"][["stem", "tm_score", "context_row"]].rename(
        columns={"tm_score": "baseline_tm", "context_row": "baseline_context_row"})
    deltas = low.merge(base, on="stem", validate="many_to_one")
    deltas["delta_vs_af3_25"] = deltas.tm_score - deltas.baseline_tm
    outputs = {"af3_context_rows.csv": rows, "af3_context_summary.csv": summary,
               "af3_context_low_depth.csv": deltas}
    for name, frame in outputs.items():
        frame.to_csv(DATA / name, index=False)
    manifest = dict(
        preprocessor="prepare_af3_context.py", preprocessor_sha256=sha256(Path(__file__)),
        sources=sources.records, files={name: sha256(DATA / name) for name in outputs},
        matched_natural_proteins=305, msa_bin_counts=targets.groupby("tier").size().to_dict(),
        metric="TM-score; archived reference-normalized protein scores; higher is better",
        uncertainty="95% percentile intervals, 5000 protein bootstraps, seed 325",
        limits=["Different MSA bins contain different proteins; this is not an MSA intervention.",
                "Extended AF3 exists only at depth <10; missing cells mean not run.",
                "AF3 baseline is shared archived MSA/no templates, not the native search pipeline.",
                "Predictor budgets and training data differ; this is not matched compute.",
                "ESMFold2 entries are archived baseline predictions, not the 100-map decoy pool.",
                "Oracle rows use ground truth and are diagnostic, not deployable selectors."],
    )
    (DATA / "af3_context_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.pivot(index="method", columns="tier", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
