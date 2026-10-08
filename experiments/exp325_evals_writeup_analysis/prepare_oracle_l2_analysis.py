"""Validate the full-cohort L/2 sweep and cache auditable protein-level scores.

One confidence-selected diffusion sample per random subset, then an arithmetic
mean across the two subsets. No accuracy-based sample or subset selection.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import FIG250, REPO, TIERS, bootstrap
from prepare_oracle_budget_analysis import METRICS, select_samples

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    """Require all 305 proteins, both subsets, all samples, and direct timings."""
    paths = [DATA / name for name in ("oracle_l2_protocol.json", "oracle_l2_targets.csv",
        "oracle_l2_maps.csv", "helico_oracle_l2_samples.csv", "helico_oracle_l2_timings.csv",
        "helico_oracle_l2_run.json")]
    protocol = json.loads(paths[0].read_text())
    targets, maps, samples, timings = (pd.read_csv(path) for path in paths[1:5])
    if len(targets) != 305 or targets.stem.duplicated().any() or targets.designed.any():
        raise ValueError("Expected the 305 natural comparison proteins")
    if protocol["map_keys"] != ["random_L2-0", "random_L2-1"]:
        raise ValueError("Only the two prespecified L/2 subsets may be run")
    keys = ["stem", "arm", "map_seed", "sample_idx"]
    expected = {(stem, "random_L2", r, s) for stem in targets.stem for r in range(2) for s in range(3)}
    if set(samples[keys].itertuples(index=False, name=None)) != expected or samples.duplicated(keys).any():
        raise ValueError("Missing, duplicate, or unexpected diffusion samples")
    expected_maps = {(stem, f"random_L2-{r}") for stem in targets.stem for r in range(2)}
    if set(timings[["stem", "mode"]].itertuples(index=False, name=None)) != expected_maps or len(timings) != 610:
        raise ValueError("Missing or duplicate inference timings")
    if not (timings.elapsed_seconds > 0).all() or not (timings.n_samples == 3).all():
        raise ValueError("Invalid inference timing or sample budget")
    numeric = samples[[*METRICS, "ptm", "ranking_score"]].to_numpy()
    if not np.isfinite(numeric).all() or not samples[list(METRICS)].ge(0).all().all() or not samples[list(METRICS)].le(1).all().all():
        raise ValueError("Invalid accuracy or confidence")
    if len(maps) != 610 or maps.duplicated(keys[:3]).any() or maps.n_absent.any():
        raise ValueError("Invalid sparse conditioning inventory")
    checked = samples.merge(maps, on=keys[:3], suffixes=("", "_frozen"), validate="many_to_one")
    for field in ("L", "requested_contacts", "n_present", "n_absent", "n_unknown"):
        if not checked[field].eq(checked[f"{field}_frozen"]).all():
            raise ValueError(f"Runtime and frozen {field} differ")
    # Reused low-depth rows must be byte-equivalent as numeric observations.
    prior = pd.read_csv(DATA / "helico_oracle_budget_samples.csv")
    prior = prior[prior.arm == "random_L2"].sort_values(keys).reset_index(drop=True)
    reused = samples[samples.stem.isin(prior.stem)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(reused, prior, check_exact=True)
    samples["source_row"] = np.arange(len(samples))
    selected = pd.concat([select_samples(samples, selector) for selector in ("ranking_score", "ptm")], ignore_index=True)
    selected.to_csv(DATA / "oracle_l2_selected.csv", index=False)
    records = []
    for (stem, selector), group in selected.groupby(["stem", "selector"], sort=True):
        if len(group) != 2 or set(group.map_seed) != {0, 1}:
            raise ValueError(f"{stem}: missing subset")
        for metric in METRICS:
            records.append(dict(stem=stem, method="oracle_L2", selector=selector, metric=metric,
                value=group[metric].mean(), min_draw=group[metric].min(), max_draw=group[metric].max(),
                n_draws=2, n_contacts=group.n_present.mean(),
                source="helico_oracle_l2_samples.csv", source_rows=json.dumps(group.source_row.astype(int).tolist())))
    protein = pd.DataFrame(records).merge(targets[["stem", "eval_set", "msa_depth", "tier", "L_exp245"]],
                                         on="stem", validate="many_to_one")
    protein.to_csv(DATA / "oracle_l2_per_protein.csv", index=False)
    summaries = []
    for cohort in ("natural", "eval-val", "eval-test"):
        subset = protein if cohort == "natural" else protein[protein.eval_set == cohort]
        for tier in [*TIERS, "All depths"]:
            cut = subset if tier == "All depths" else subset[subset.tier == tier]
            for (selector, metric), group in cut.groupby(["selector", "metric"], sort=True):
                mean, lo, hi = bootstrap(group.value.to_numpy())
                summaries.append(dict(cohort=cohort, tier=tier, selector=selector, metric=metric,
                    n=len(group), mean=mean, ci_low=lo, ci_high=hi,
                    stems="|".join(group.sort_values("stem").stem)))
    pd.DataFrame(summaries).to_csv(DATA / "oracle_l2_summary.csv", index=False)
    reference_path = REPO / FIG250 / "3_structure_accuracy/per_target.csv"
    reference = pd.read_csv(reference_path)
    reference = reference[(reference.arm == "oracle") & reference.target_id.isin(targets.stem)]
    reference = reference.rename(columns={"stem": "archived_stem", "target_id": "stem"})
    if len(reference) != 305 or reference.stem.duplicated().any():
        raise ValueError("Incomplete archived full-map oracle comparison")
    full = reference.melt(id_vars=["stem"], value_vars=["gdt_ts", "lddt"],
                          var_name="metric", value_name="full_oracle")
    paired = protein.merge(full, on=["stem", "metric"], validate="many_to_one")
    paired["delta"] = paired.value - paired.full_oracle
    paired.to_csv(DATA / "oracle_l2_vs_full.csv", index=False)
    outputs = ["oracle_l2_selected.csv", "oracle_l2_per_protein.csv", "oracle_l2_summary.csv", "oracle_l2_vs_full.csv"]
    capped = maps[maps.n_present < maps.requested_contacts]
    manifest = dict(protocol=protocol, n_targets=305, n_maps=610, n_samples=1830,
        reused_stems=sorted(prior.stem.unique()),
        sources={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [*paths, reference_path, DATA / "helico_oracle_budget_samples.csv"]},
        files={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in outputs},
        capped_relative_budgets=capped[["stem", "map_seed", "requested_contacts", "n_present"]].to_dict("records"),
        aggregation="Highest ranking_score within each 3-sample map; mean of 2 subsets within protein; equal-weight protein mean and 5000 protein-bootstrap draws. pTM sensitivity saved separately.")
    (DATA / "oracle_l2_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared 305 proteins, 610 maps, 1830 samples; capped L/2 on {capped.stem.nunique()} proteins")


if __name__ == "__main__":
    main()
