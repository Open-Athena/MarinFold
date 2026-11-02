"""Apply the preregistered development gate without inspecting held-out truth."""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def metric_delta(deltas: pd.DataFrame, arm: str, comparator: str, region: str) -> float:
    """Read one primary validity-gated oracle delta."""
    row = deltas[
        (deltas.arm == arm)
        & (deltas.comparator == comparator)
        & (deltas["range"] == region)
        & (deltas.metric == "validity_gated_oracle_r_precision")
    ]
    if len(row) != 1:
        raise ValueError(f"missing unique delta for {arm}/{comparator}/{region}")
    return float(row.mean_delta.iloc[0])


def main() -> None:
    """Freeze the strongest arm only when every preregistered gate passes."""
    deltas = pd.read_csv(HERE / "data" / "dev_paired_deltas.csv")
    candidates = []
    diagnostics = {}
    for size in (8, 16):
        arm = f"medoid_k{size}"
        random_control = f"random_k{size}"
        values = {
            "all_vs_iid": metric_delta(deltas, arm, "iid100", "all"),
            "all_vs_random": metric_delta(deltas, arm, random_control, "all"),
            "long_vs_iid": metric_delta(deltas, arm, "iid100", "long"),
            "long_vs_random": metric_delta(deltas, arm, random_control, "long"),
        }
        passed = (
            values["all_vs_iid"] >= 0.005
            and values["all_vs_random"] >= 0.005
            and values["long_vs_iid"] >= -0.005
            and values["long_vs_random"] >= -0.005
        )
        diagnostics[arm] = {**values, "passed": passed}
        if passed:
            candidates.append((min(values["all_vs_iid"], values["all_vs_random"]), arm))
    choice = max(candidates)[1] if candidates else None
    record = {
        "choice": choice,
        "advance_to_heldout": choice is not None,
        "selection_rule": (
            "all-range oracle >= +0.005 versus iid100 and matched random; "
            "long-range oracle >= -0.005 versus both"
        ),
        "diagnostics": diagnostics,
    }
    (HERE / "data" / "frozen_choice.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
