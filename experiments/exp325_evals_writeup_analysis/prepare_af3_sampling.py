"""Cheap table preprocessing: fixed seed prefixes and confidence/oracle selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def prefix_curves(frame: pd.DataFrame, seed_start: int, threshold: float) -> pd.DataFrame:
    """Select within each prefix, using earliest seed to break exact ties."""
    rows = []
    for stem, group in frame.groupby("stem", sort=True):
        group = group.sort_values("seed")
        records = group.to_dict("records")
        if group.seed.tolist() != list(range(seed_start, seed_start + len(group))):
            raise ValueError(f"{stem}: seeds are not a complete consecutive prefix")
        best = selected = ranked = records[0]
        hits = {cut: 0 for cut in (0.5, 0.7, threshold)}
        first_hit = None
        for n, row in enumerate(records, 1):
            if row["tm_score"] > best["tm_score"]:
                best = row
            if row["ptm"] > selected["ptm"]:
                selected = row
            if row["ranking_score"] > ranked["ranking_score"]:
                ranked = row
            for cut in hits:
                hits[cut] += int(row["tm_score"] >= cut)
            if first_hit is None and row["tm_score"] >= threshold:
                first_hit = n
            rows.append(dict(stem=stem, budget=n, msa_depth=row["msa_depth"],
                             best_tm=best["tm_score"], best_seed=best["seed"],
                             ptm_selected_tm=selected["tm_score"], ptm_selected_seed=selected["seed"],
                             ptm_max=selected["ptm"], ranking_selected_tm=ranked["tm_score"],
                             ranking_selected_seed=ranked["seed"], first_hit_draw=first_hit,
                             **{f"hits_tm_{int(cut * 100)}": value for cut, value in hits.items()}))
    return pd.DataFrame(rows)


def main() -> None:
    """Prepare all numerical plot elements; rendering only reads these CSVs."""
    protocol = json.loads((HERE / "data/af3_sampling_protocol.json").read_text())
    samples = pd.read_csv(HERE / "data/af3_sampling_samples.csv")
    if not np.isfinite(samples[["ptm", "ranking_score", "tm_score"]].to_numpy()).all():
        raise ValueError("Nonfinite confidence or accuracy")
    curves = prefix_curves(samples, protocol["seed_start"], protocol["accuracy_threshold_tm"])
    curves.to_csv(HERE / "data/af3_sampling_curves.csv", index=False)
    budgets = [1, 10, 25, 100, 1000]
    summary = curves[curves.budget.isin(budgets)].copy()
    summary.to_csv(HERE / "data/af3_sampling_summary.csv", index=False)
    names = ["af3_sampling_protocol.json", "af3_sampling_samples.csv", "af3_sampling_original.csv",
             "af3_sampling_scoring.json", "af3_sampling_curves.csv", "af3_sampling_summary.csv"]
    manifest = dict(generator="prepare_af3_sampling.py", args=[],
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    files={name: hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() for name in names})
    (HERE / "data/af3_sampling_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
