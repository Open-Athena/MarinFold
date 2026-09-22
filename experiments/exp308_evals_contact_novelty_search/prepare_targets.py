#!/usr/bin/env python
"""Freeze sequence-only fold-switching targets and the pilot subset."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP306 = HERE.parent / "exp306_evals_contact_block_beam_search"

PILOT = (
    "2hdma_2n54b",  # short secondary development positive
    "3jv6a_1zk9a",  # short secondary development positive
    "4qhfa_4qhha",  # short primary development negative
    "2frha_1fzpd",  # short primary development negative
    "4gqcc_4gqcb",  # primary development negative
    "3hdea_3hdfa",  # primary development negative
    "3t5oa_4a5wb",  # long primary development positive
)


def main() -> None:
    """Save the auditable seven-protein pilot and untouched primary test set."""
    targets = pd.read_csv(EXP306 / "data" / "targets.csv")
    targets = targets[targets.cohort == "foldswitch"].copy()
    if len(targets) != 67 or targets.stem.nunique() != 67:
        raise ValueError("expected 67 non-capped fold-switching targets")
    pilot = set(PILOT)
    if set(targets.stem) & pilot != pilot:
        raise ValueError("missing pilot target")
    if not (targets.loc[targets.stem.isin(pilot), "split"] == "dev").all():
        raise ValueError("pilot target is not in development split")
    targets["pilot"] = targets.stem.isin(pilot)
    targets.to_csv(HERE / "data" / "targets.csv", index=False)
    print(f"saved {len(targets)} targets; pilot={targets.pilot.sum()}; "
          f"primary test={sum((targets.split == 'test') & targets.primary)}")


if __name__ == "__main__":
    main()
