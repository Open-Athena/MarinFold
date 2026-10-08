"""Cache paired structural outcomes and added-contact precision before rendering."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import bootstrap

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
METRICS = ("gdt_ts", "tm_score", "lddt")


def main() -> None:
    """Validate the matched experiment and bootstrap five proteins, not 35 maps."""
    names = ["seed_completion_design.json", "seed_completion_protocol.json", "seed_completion_cases.csv",
             "seed_completion_maps.csv", "seed_completion_timings.csv", "helico_seed_completion_samples.csv",
             "helico_seed_completion_timings.csv", "helico_seed_completion_run.json",
             "helico_oracle_budget_samples.csv", "oracle_budget_maps.csv", "oracle_budget_targets.csv"]
    sources = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names}
    cases = pd.read_csv(DATA / "seed_completion_cases.csv", dtype={"budget": str})
    maps = pd.read_csv(DATA / "seed_completion_maps.csv", dtype={"budget": str})
    targets = pd.read_csv(DATA / "oracle_budget_targets.csv")
    old_maps = pd.read_csv(DATA / "oracle_budget_maps.csv")
    if len(cases) != 35 or len(maps) != 35 or len(targets) != 5 or not (targets.msa_depth < 10).all():
        raise ValueError("Unexpected cohort or case count")
    new = pd.read_csv(DATA / "helico_seed_completion_samples.csv")
    direct = pd.read_csv(DATA / "helico_oracle_budget_samples.csv")
    for filename in ("seed_completion_timings.csv", "helico_seed_completion_timings.csv"):
        timing = pd.read_csv(DATA / filename)
        if len(timing) != 35 or timing.duplicated(["stem", "mode"]).any() or not (timing.elapsed_seconds > 0).all():
            raise ValueError(f"Incomplete direct timing records: {filename}")
    checked_maps = cases.merge(maps, left_on=["stem", "arm", "replicate"], right_on=["stem", "arm", "map_seed"],
                              suffixes=("", "_map"), validate="one_to_one")
    if len(checked_maps) != 35 or not checked_maps.direct_state_sha256.eq(checked_maps.direct_state_sha256_map).all():
        raise ValueError("Seed identity changed between prompting and completion")
    frames = []
    for method, frame, source in (("complete", new, "helico_seed_completion_samples.csv"),
                                  ("direct", direct, "helico_oracle_budget_samples.csv")):
        frame = frame.assign(source=source, source_row=np.arange(len(frame)))
        join = cases.rename(columns={"replicate": "map_seed", "arm": "completed_arm"})
        join["arm"] = join.completed_arm if method == "complete" else join.direct_arm
        merged = join.merge(frame, on=["stem", "arm", "map_seed"], validate="one_to_many")
        if len(merged) != 105 or merged.duplicated(["stem", "arm", "map_seed", "sample_idx"]).any():
            raise ValueError(f"Incomplete or duplicated {method} samples")
        if set(merged.sample_idx) != {0, 1, 2} or not np.isfinite(merged[[*METRICS, "ranking_score", "ptm"]]).all().all():
            raise ValueError("Invalid sample values")
        expected = merged.final_contacts if method == "complete" else merged.n_seed
        if not merged.n_present.eq(expected).all() or merged.n_absent.any():
            raise ValueError("Runtime contact budget differs from frozen design")
        if method == "direct":
            audited = join.merge(old_maps, on=["stem", "arm", "map_seed"], validate="one_to_one")
            if not audited.direct_state_sha256.eq(audited.state_sha256).all():
                raise ValueError("Reused direct map differs from seed conditioning")
        for selector in ("ranking_score", "ptm"):
            keys = ["stem", "budget", "map_seed"]
            selected = merged.sort_values([*keys, selector, "sample_idx"], ascending=[True, True, True, False, True]).drop_duplicates(keys)
            frames.append(selected.assign(method=method, selector=selector))
    selected = pd.concat(frames, ignore_index=True)
    selected.to_csv(DATA / "seed_completion_selected.csv", index=False)
    protein_rows = []
    for (stem, budget, method, selector), group in selected.groupby(["stem", "budget", "method", "selector"]):
        if len(group) != (1 if budget == "0" else 2):
            raise ValueError("Wrong number of seed subsets")
        for metric in METRICS:
            protein_rows.append(dict(stem=stem, budget=budget, method=method, selector=selector, metric=metric,
                value=group[metric].mean(), min_draw=group[metric].min(), max_draw=group[metric].max(),
                n_seed=int(group.n_seed.iloc[0]), n_contacts=int(group.n_present.iloc[0]), n_draws=len(group),
                source=group.source.iloc[0], source_rows=json.dumps(group.source_row.astype(int).tolist())))
    rows = pd.DataFrame(protein_rows).merge(targets[["stem", "msa_depth", "L_exp245"]], on="stem", validate="many_to_one")
    rows.to_csv(DATA / "seed_completion_per_protein.csv", index=False)
    summaries, deltas = [], []
    for (selector, metric, budget), group in rows.groupby(["selector", "metric", "budget"]):
        paired = group.pivot(index="stem", columns="method", values="value")
        if paired.shape != (5, 2) or paired.isna().any().any():
            raise ValueError("Unmatched protein comparison")
        for method in ("direct", "complete"):
            mean, low, high = bootstrap(paired[method].to_numpy())
            summaries.append(dict(selector=selector, metric=metric, budget=budget, method=method,
                                  n=5, mean=mean, ci_low=low, ci_high=high))
        difference = paired.complete - paired.direct
        mean, low, high = bootstrap(difference.to_numpy())
        deltas.append(dict(selector=selector, metric=metric, budget=budget, n=5, delta=mean,
                          ci_low=low, ci_high=high, improved=int((difference > 0).sum()),
                          worsened=int((difference < 0).sum())))
    pd.DataFrame(summaries).to_csv(DATA / "seed_completion_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(DATA / "seed_completion_deltas.csv", index=False)
    contact_rows = maps.groupby(["stem", "budget"], as_index=False).agg(
        added_precision=("added_precision", "mean"), total_precision=("total_precision", "mean"),
        n_seed=("n_seed", "first"), n_added=("n_added", "first"), n_present=("n_present", "first"))
    contact_rows.to_csv(DATA / "seed_completion_contact_precision.csv", index=False)
    outputs = ["selected", "per_protein", "summary", "deltas", "contact_precision"]
    files = {f"seed_completion_{name}.csv": hashlib.sha256((DATA / f"seed_completion_{name}.csv").read_bytes()).hexdigest() for name in outputs}
    manifest = dict(sources=sources, files=files, n_proteins=5, new_maps=35, new_diffusion_samples=105,
        aggregation="Confidence-select within each map; average two subsets within protein; equal-weight five-protein mean. Paired protein bootstrap with 5000 replicates, seed325.",
        protocol=json.loads((DATA / "seed_completion_protocol.json").read_text()))
    (DATA / "seed_completion_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(deltas).query("selector == 'ranking_score' and metric == 'gdt_ts'").to_string(index=False))


if __name__ == "__main__":
    main()
