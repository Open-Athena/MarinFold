#!/usr/bin/env python
"""Post-hoc sensitivity analysis for truly identical PDB-chain sequences."""

import json
from pathlib import Path

import pandas as pd

from analyze import paired_interval

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"


def main() -> None:
    annotations = pd.DataFrame(
        [json.loads(line) for line in (SOURCE / "foldswitch_universe.jsonl").read_text().splitlines()]
    )[["pair_id", "n_seq_mismatch", "pair_identity"]]
    per = pd.read_csv(DATA / "per_protein.csv").merge(
        annotations, on="pair_id", validate="many_to_one"
    )
    subset = per[per.primary & (per.split == "test") & (per.n_seq_mismatch == 0)]
    iid = subset[subset.method == "iid"].set_index("pair_id")
    branch = subset[subset.method == "branch10"].set_index("pair_id")
    common = iid.index.intersection(branch.index)
    if len(common) != 17:
        raise ValueError(f"expected 17 exact-sequence primary test pairs, found {len(common)}")
    details = pd.DataFrame({
        "pair_id": common,
        "iid_enrichment": iid.loc[common, "minority_enrichment"].to_numpy(),
        "branch10_enrichment": branch.loc[common, "minority_enrichment"].to_numpy(),
        "iid_recall": iid.loc[common, "minority_recall"].to_numpy(),
        "branch10_recall": branch.loc[common, "minority_recall"].to_numpy(),
        "iid_dual_hit": iid.loc[common, "dual_contact_hit"].to_numpy(),
        "branch10_dual_hit": branch.loc[common, "dual_contact_hit"].to_numpy(),
    })
    details["paired_delta"] = details.branch10_enrichment - details.iid_enrichment
    details.sort_values("pair_id").to_csv(DATA / "exact_sequence_primary_test.csv", index=False)
    lower, upper = paired_interval(details.paired_delta.to_numpy())
    summary = pd.DataFrame([{
        "n": len(details), "iid_mean_enrichment": details.iid_enrichment.mean(),
        "branch10_mean_enrichment": details.branch10_enrichment.mean(),
        "paired_delta": details.paired_delta.mean(), "paired_delta_lo": lower,
        "paired_delta_hi": upper, "iid_mean_recall": details.iid_recall.mean(),
        "branch10_mean_recall": details.branch10_recall.mean(),
        "iid_dual_hits": int(details.iid_dual_hit.sum()),
        "branch10_dual_hits": int(details.branch10_dual_hit.sum()),
    }])
    summary.to_csv(DATA / "exact_sequence_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
