"""Preserve the original run and substitute validated full-context rechecks.

Only the three protein/checkpoint units named in capped_rollouts.json are
replaced. The original metric/timing tables and raw samples remain unchanged.
The committed cap_checks.json permits offline reconstruction of this merge.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from fetch_results import ROOT, workstation_filesystem

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
KEYS = ["model", "dataset", "stem"]


def replace_units(original: pd.DataFrame, replacement: pd.DataFrame) -> pd.DataFrame:
    """Replace complete identified units, preserving every unaffected row."""
    keys = pd.MultiIndex.from_frame(replacement[KEYS].drop_duplicates())
    keep = ~pd.MultiIndex.from_frame(original[KEYS]).isin(keys)
    return pd.concat([original.loc[keep], replacement], ignore_index=True)


def fetch() -> None:
    """Fetch the small recheck record and stage its public artifacts."""
    filesystem = workstation_filesystem()
    prefix = ROOT + "/v1-capcheck"
    paths = filesystem.glob(prefix + "/rollout/*/complete/*.json")
    if len(paths) != 3:
        raise ValueError(f"Expected three complete rechecks, found {len(paths)}")
    results = []
    for path in paths:
        with filesystem.open(path, "rt") as handle:
            results.append(json.load(handle))
    (DATA / "cap_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    destination = HERE / "scratch/cap_public"
    for source in filesystem.find(prefix):
        target = destination / source.removeprefix(prefix + "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(source, str(target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    checks = json.loads((DATA / "cap_checks.json").read_text())
    caps = json.loads((DATA / "capped_rollouts.json").read_text())
    expected = {(r["model"], r["dataset"], r["stem"]) for r in caps}
    observed = {(r["model"], r["dataset"], r["stem"]) for r in checks}
    if observed != expected or len(checks) != len(expected):
        raise ValueError("Recheck identities do not match the original capped units")
    metric_rows, sample_rows, timing_rows = [], [], []
    for result in checks:
        base = {key: result[key] for key in (*KEYS, "L", "diagnostic")}
        if any(t["unfinished_rollouts"] for t in result["timings"]):
            raise ValueError(f"A full-context recheck still has capped samples: {base}")
        if not all(t["full_context_budget"] for t in result["timings"]):
            raise ValueError("The requested full-context budget was not used")
        metric_rows.extend({**base, **row} for row in result["metrics"])
        sample_rows.extend({**base, **row} for row in result["single_sample"])
        timing_rows.extend(
            {**row, "model": result["model"]} for row in result["timings"]
        )
    original = pd.read_csv(DATA / "results/per_protein.csv")
    corrected = replace_units(original, pd.DataFrame(metric_rows))
    corrected.to_csv(DATA / "results/per_protein_corrected.csv", index=False)
    single = pd.read_csv(DATA / "results/single_samples.csv")
    replace_units(single, pd.DataFrame(sample_rows)).to_csv(
        DATA / "results/single_samples_corrected.csv", index=False
    )
    pd.DataFrame(timing_rows).to_csv(DATA / "cap_check_timings.csv", index=False)
    original_timing = pd.read_csv(DATA / "timings.csv")
    effective_timing = replace_units(original_timing, pd.DataFrame(timing_rows))
    effective_timing.to_csv(DATA / "timings_corrected.csv", index=False)
    if effective_timing.unfinished_rollouts.sum() != 0:
        raise ValueError("Unfinished rollouts remain after applying rechecks")
    compare_keys = KEYS + ["mode", "budget", "range", "cut"]
    delta = pd.DataFrame(metric_rows).merge(
        original,
        on=compare_keys,
        suffixes=("_recheck", "_original"),
        validate="one_to_one",
    )
    delta["delta"] = delta.precision_recheck - delta.precision_original
    delta[compare_keys + ["precision_original", "precision_recheck", "delta"]].to_csv(
        DATA / "cap_check_deltas.csv", index=False
    )
    print(
        delta[
            (delta.cut == "R")
            & delta["range"].isin(["all", "long"])
            & (delta.budget == 100)
        ].to_string(index=False)
    )
    print(
        "Validated all three rechecks; the final metric table contains no unfinished samples."
    )


if __name__ == "__main__":
    main()
