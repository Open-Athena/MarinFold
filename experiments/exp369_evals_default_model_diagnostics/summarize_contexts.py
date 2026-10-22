"""Describe the frozen contact budgets and structural coverage of both policies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Measure context composition independently of the model's predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = []
    for target in plan["targets"]:
        for repeat, grid in target["doses_by_repeat"].items():
            grid = grid + [{"axis": "fraction", "dose": 0., "k": 0},
                           {"axis": "sse_round", "dose": 0, "k": 0}]
            for dose in grid:
                for policy in ("random", "sse_balanced"):
                    pairs = target["orders"][f"{policy}:{repeat}"][:dose["k"]]
                    covered = {target["elements"][i] for pair in pairs for i in pair} - {-1}
                    separation = np.array([j - i for i, j in pairs])
                    rows.append({"stem": target["stem"], "repeat": int(repeat), "policy": policy,
                                 "axis": dose["axis"], "dose": dose["dose"], "n_supplied": len(pairs),
                                 "supplied_fraction": len(pairs) / len(target["truth"]),
                                 "covered_elements": len(covered),
                                 "n_coverable_elements": target["n_coverable_elements"],
                                 "covered_element_fraction": len(covered) / target["n_coverable_elements"],
                                 "long_contact_fraction": float((separation >= 24).mean()) if len(pairs) else None,
                                 "mean_sequence_separation": float(separation.mean()) if len(pairs) else None})
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "data/context_composition_per_repeat.csv", index=False)
    proteins = frame.drop(columns="repeat").groupby(["stem", "policy", "axis", "dose"], as_index=False).mean(numeric_only=True)
    summary = proteins.groupby(["policy", "axis", "dose"], as_index=False).mean(numeric_only=True)
    summary["n_proteins"] = proteins.groupby(["policy", "axis", "dose"]).size().to_numpy()
    summary.to_csv(HERE / "data/context_composition_summary.csv", index=False)
    denominators = proteins.loc[proteins.policy == "random"].groupby(["axis", "dose"]).agg(
        n_proteins=("stem", "size"), mean_count=("n_supplied", "mean"), median_count=("n_supplied", "median"),
        mean_fraction=("supplied_fraction", "mean"), mean_coverable_sse=("n_coverable_elements", "mean")).reset_index()
    denominators.to_csv(HERE / "data/intervention_dose_denominators.csv", index=False)
    print(summary.loc[summary.axis == "sse_round"].to_string(index=False))


if __name__ == "__main__":
    main()
