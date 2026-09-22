#!/usr/bin/env python
"""Compare independent root rollouts with exp301's published 500-rollout run."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"


def main() -> None:
    scored = pd.read_parquet(HERE / "_cache" / "scored_all_candidates.parquet")
    root = scored[scored.source_arm == "root"].groupby("pair_id").agg(
        root_phi=("phi", "mean"), root_contacts=("n_pred", "mean"),
        n_root=("candidate_id", "size"),
    )
    prior = pd.read_csv(SOURCE / "fold_preference.csv").set_index("pair_id")
    merged = root.join(prior[["phi", "contacts_per_rollout"]], how="inner")
    if len(merged) != 67 or not (merged.n_root == 100).all():
        raise ValueError("root cohort or rollout count differs from the frozen 67 x 100 design")
    merged["delta_phi"] = merged.root_phi - merged.phi
    merged.to_csv(HERE / "data" / "root_replication.csv")
    print(f"67 roots: phi correlation={merged.root_phi.corr(merged.phi):.3f}, "
          f"mean |delta phi|={merged.delta_phi.abs().mean():.3f}, "
          f"mean contact count exp304={merged.root_contacts.mean():.1f} "
          f"exp301={merged.contacts_per_rollout.mean():.1f}")


if __name__ == "__main__":
    main()
