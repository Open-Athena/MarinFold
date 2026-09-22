#!/usr/bin/env python
"""Compare the predeclared novelty variants on the seven development pairs."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MODES = (
    "b16_e0p0_d0_w10",
    "b16_e0p05_d0_w10",
    "b16_e0p05_d20_w10",
    "b32_e0p05_d20_w10",
    "b16_e0p2_d20_w10",
)


def main() -> None:
    """Write an auditable table with selection and primary-dev denominators."""
    rows = []
    for mode in MODES:
        path = HERE / "data" / f"foldswitch_summary_pilot_{mode}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        report = pd.read_csv(path)
        for record in report.to_dict("records"):
            rows.append(record)
    combined = pd.DataFrame(rows)
    combined.to_csv(HERE / "data" / "pilot_comparison.csv", index=False)
    print(combined[["mode", "cohort", "n", "dual_pool", "dual_blind",
                    "iid_dual_pool", "iid_dual_blind", "beam4_dual_pool",
                    "beam4_dual_blind", "share_unseen_first20",
                    "total_time_ratio_vs_beam4"]].to_string(index=False))


if __name__ == "__main__":
    main()
