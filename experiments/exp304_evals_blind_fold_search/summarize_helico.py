#!/usr/bin/env python
"""Copy Helico metrics into the experiment's standard timing table."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "_cache" / "helico" / "results"


def main() -> None:
    manifest = pd.read_csv(DATA / "helico_target_manifest.csv")
    timings = pd.read_csv(RESULTS / "exp304-validation.timings.csv")
    results = pd.read_csv(RESULTS / "exp304-validation.csv")
    merged = manifest.merge(timings, on="target_id", validate="one_to_one")
    if len(merged) != len(manifest) or not (merged.status == "ok").all():
        raise ValueError("Helico validation is incomplete")
    merged["stem"] = merged.pair_id
    merged["n_residues"] = merged.observed_residues
    merged["n_pairs"] = merged.n_residues * (merged.n_residues - 1) // 2
    merged["mode"] = merged.variant + ":input_fold" + merged.input_fold.astype(str)
    merged["total_seconds"] = merged.elapsed_seconds
    merged["elapsed_seconds"] = merged.predict_seconds
    merged["model_nickname"] = "helico-contacts-msafree-01-step-6000"
    merged["runner_tag"] = "modal"
    merged["n_samples"] = 3
    merged["n_cycles"] = 6
    merged.to_csv(DATA / "helico_timings.csv", index=False)
    result = manifest.merge(results, on="target_id", validate="one_to_one")
    result.to_csv(DATA / "helico_results.csv", index=False)
    print(f"wrote {len(merged)} timing and result rows")


if __name__ == "__main__":
    main()
