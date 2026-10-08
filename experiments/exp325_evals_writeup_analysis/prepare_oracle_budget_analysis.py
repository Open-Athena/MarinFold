"""Reduce the frozen oracle-budget sweep to source-traced plot tables.

Select one diffusion sample by confidence per map, then average the two random
subsets within a protein. Bootstrap proteins, never diffusion samples or maps.
Rendering reads only the resulting small CSVs and does not repeat this analysis.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generation.oracle_budgets import BUDGETS, REPLICATES, map_keys
from prepare import TIERS, bootstrap

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
METRICS = ("gdt_ts", "lddt", "tm_score")
METHODS = {"top_0": "no_contacts", "positive_all": "oracle_positive_all", "oracle": "oracle",
           "random_5": "oracle_5", "random_10": "oracle_10", "random_L5": "oracle_L5", "random_L2": "oracle_L2"}


def select_samples(samples: pd.DataFrame, selector: str) -> pd.DataFrame:
    """Pick confidence argmax per map, using sample index to break exact ties."""
    keys = ["stem", "arm", "map_seed"]
    result = samples.sort_values([*keys, selector, "sample_idx"],
                                ascending=[True, True, True, False, True]).drop_duplicates(keys)
    return result.assign(selector=selector, method=result.arm.map(METHODS))


def main() -> None:
    """Reject incomplete sweeps and write every aggregate's contributing rows."""
    paths = [DATA / name for name in ("oracle_budget_protocol.json", "oracle_budget_targets.csv",
        "oracle_budget_maps.csv", "helico_oracle_budget_samples.csv", "helico_oracle_budget_timings.csv",
        "helico_oracle_budget_run.json", "figure_rows.csv")]
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    protocol = json.loads(paths[0].read_text())
    targets, maps, samples, timings = (pd.read_csv(path) for path in paths[1:5])
    if len(targets) != 5 or targets.stem.duplicated().any() or not (targets.msa_depth < 10).all() or targets.designed.any():
        raise ValueError("Expected all five natural Figure 02 proteins at MSA depth <10")
    samples["source_row"] = np.arange(len(samples))
    expected_maps = {(stem, arm, replicate) for stem in targets.stem for arm, replicate in map_keys()}
    actual_maps = set(samples[["stem", "arm", "map_seed"]].itertuples(index=False, name=None))
    if actual_maps != expected_maps or len(samples) != len(expected_maps) * 3:
        raise ValueError("Missing or unexpected oracle-budget predictions")
    if samples.duplicated(["stem", "arm", "map_seed", "sample_idx"]).any() or set(samples.sample_idx) != {0, 1, 2}:
        raise ValueError("Diffusion sample inventory differs from the frozen protocol")
    if len(timings) != len(expected_maps) or timings.duplicated(["stem", "mode"]).any():
        raise ValueError("Missing direct inference timings")
    numeric = samples[[*METRICS, "ranking_score", "ptm"]].to_numpy()
    if not np.isfinite(numeric).all() or not samples[list(METRICS)].ge(0).all().all() or not samples[list(METRICS)].le(1).all().all():
        raise ValueError("Invalid structural metrics/confidence")
    checked = samples.merge(maps, on=["stem", "arm", "map_seed"], suffixes=("", "_frozen"), validate="many_to_one")
    for column in ("L", "n_present", "n_absent", "n_unknown"):
        if not checked[column].eq(checked[f"{column}_frozen"]).all():
            raise ValueError(f"Runtime conditioning differs from frozen {column}")
    selected = pd.concat([select_samples(samples, selector) for selector in ("ranking_score", "ptm")], ignore_index=True)
    selected.to_csv(DATA / "oracle_budget_selected.csv", index=False)
    records = []
    for (stem, method, selector), group in selected.groupby(["stem", "method", "selector"], sort=True):
        expected = REPLICATES if group.arm.iloc[0] in BUDGETS else 1
        if len(group) != expected:
            raise ValueError(f"{stem}/{method}: wrong number of random subsets")
        for metric in METRICS:
            records.append(dict(stem=stem, method=method, selector=selector, metric=metric,
                value=group[metric].mean(), min_draw=group[metric].min(), max_draw=group[metric].max(),
                n_draws=len(group), n_contacts=group.n_present.mean(),
                source="helico_oracle_budget_samples.csv", source_rows=json.dumps(group.source_row.astype(int).tolist())))
    per_protein = pd.DataFrame(records).merge(targets[["stem", "eval_set", "msa_depth", "tier", "L_exp245"]],
                                            on="stem", validate="many_to_one")
    original = pd.read_csv(DATA / "figure_rows.csv")
    baselines = original[original.stem.isin(targets.stem) &
        ((original.figure == "01_predictors") |
         ((original.figure == "05_folding") & (original.method == "marinfold_helico")))].copy()
    if baselines.duplicated(["stem", "method", "metric"]).any() or len(baselines) != 5 * 8 * 2:
        raise ValueError("Incomplete matched predictor context")
    baselines["source_rows"] = baselines.source_row.map(lambda row: json.dumps([int(row)]))
    baselines["n_draws"] = 1
    baselines["n_contacts"] = np.nan
    baselines["min_draw"] = baselines.value
    baselines["max_draw"] = baselines.value
    baselines = baselines.rename(columns={"L": "L_exp245"})
    combined = pd.concat([per_protein, *[baselines.assign(selector=selector)[per_protein.columns]
        for selector in ("ranking_score", "ptm")]], ignore_index=True)
    combined.to_csv(DATA / "oracle_budget_per_protein.csv", index=False)
    combined[(combined.selector == "ranking_score") & (combined.metric == "gdt_ts")].pivot(
        index="stem", columns="method", values="value").to_csv(DATA / "oracle_budget_gdt_table.csv")
    summaries, deltas = [], []
    for cohort in ("natural", "eval-val", "eval-test"):
        cohort_rows = combined if cohort == "natural" else combined[combined.eval_set == cohort]
        for tier in [*TIERS, "All depths"]:
            subset = cohort_rows if tier == "All depths" else cohort_rows[cohort_rows.tier == tier]
            for (selector, metric, method), group in subset.groupby(["selector", "metric", "method"], sort=True):
                mean, lo, hi = bootstrap(group.value.to_numpy())
                summaries.append(dict(selector=selector, cohort=cohort, tier=tier, metric=metric, method=method,
                    n=len(group), mean=mean, ci_low=lo, ci_high=hi,
                    min_contacts=group.n_contacts.min(), median_contacts=group.n_contacts.median(),
                    max_contacts=group.n_contacts.max()))
            for (selector, metric), group in subset.groupby(["selector", "metric"], sort=True):
                paired = group.pivot(index="stem", columns="method", values="value")
                if paired.isna().any().any():
                    raise ValueError("Unmatched structural comparison")
                for baseline in ("no_contacts", "oracle", "oracle_positive_all"):
                    for method in METHODS.values():
                        if method == baseline:
                            continue
                        values = (paired[method] - paired[baseline]).to_numpy()
                        mean, lo, hi = bootstrap(values)
                        deltas.append(dict(selector=selector, cohort=cohort, tier=tier, metric=metric,
                            method=method, baseline=baseline, n=len(values), delta=mean, ci_low=lo, ci_high=hi))
    pd.DataFrame(summaries).to_csv(DATA / "oracle_budget_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(DATA / "oracle_budget_paired_deltas.csv", index=False)
    # Fresh full-map and zero-contact controls expose stochastic/protocol drift
    # relative to the archived Figure 02 curves instead of hiding it in a join.
    old = original[(original.figure == "02_oracle") & original.stem.isin(targets.stem) &
                   original.method.isin(["oracle", "no_contacts"])][["stem", "method", "metric", "value"]]
    control = per_protein[(per_protein.selector == "ranking_score") & per_protein.method.isin(["oracle", "no_contacts"])]
    control = control.merge(old, on=["stem", "method", "metric"], suffixes=("_fresh", "_archived"), validate="one_to_one")
    control["delta"] = control.value_fresh - control.value_archived
    control.to_csv(DATA / "oracle_budget_control_check.csv", index=False)
    outputs = ["oracle_budget_selected.csv", "oracle_budget_per_protein.csv", "oracle_budget_summary.csv",
               "oracle_budget_paired_deltas.csv", "oracle_budget_control_check.csv", "oracle_budget_gdt_table.csv"]
    manifest = dict(sources=sources, protocol=protocol,
        files={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in outputs},
        n_targets=len(targets), n_maps=len(expected_maps), n_samples=len(samples),
        capped_relative_budgets=maps[maps.requested_contacts != maps.n_present][["stem", "arm", "map_seed", "requested_contacts", "n_present"]].to_dict("records"),
        aggregation="Confidence-select diffusion sample; average the two subsets within protein; equal-weight protein mean and 5000 protein bootstrap replicates",
        baseline_selection="External predictor selections are unchanged. The selector menu changes only Helico sample selection.")
    (DATA / "oracle_budget_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(samples)} samples on {len(targets)} proteins for fast rendering")


if __name__ == "__main__":
    main()
