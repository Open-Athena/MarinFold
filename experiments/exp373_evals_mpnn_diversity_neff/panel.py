"""Save compact contact-map source tables for the panel selected before inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import candidate_sets, load

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Extract frequencies and ground truth for the frozen eight-protein panel."""
    plan = json.loads((HERE / "data/plan.json").read_text())
    truth = {
        r["stem"]: r
        for r in json.loads((HERE / "inputs/ground_truth.json").read_text())
    }
    rows = []
    for stem in plan["representative_panel"]:
        gt = truth[stem]
        matrices = {}
        for model in ["default", "mpnn"]:
            raw = load(HERE / "scratch/raw" / model / f"{stem}-t1.0-fixed.json.gz")
            sets, true, _, _ = candidate_sets(raw, gt)
            counts = np.zeros(gt["L"] ** 2)
            for sample in sets:
                if sample:
                    counts[list(sample)] += 1
            matrices[model] = counts / len(sets)
        visible = (
            set(np.flatnonzero(matrices["default"]))
            | set(np.flatnonzero(matrices["mpnn"]))
            | true
        )
        for pair in sorted(visible):
            rows.append(
                {
                    "stem": stem,
                    "i": pair // gt["L"],
                    "j": pair % gt["L"],
                    "truth": int(pair in true),
                    "default": matrices["default"][pair],
                    "mpnn": matrices["mpnn"][pair],
                    "difference": matrices["mpnn"][pair] - matrices["default"][pair],
                }
            )
    pd.DataFrame(rows).to_csv(HERE / "data/panel_contacts.csv.gz", index=False)
    pd.DataFrame({"stem": plan["representative_panel"]}).to_csv(
        HERE / "data/panel_order.csv", index=False
    )


if __name__ == "__main__":
    main()
