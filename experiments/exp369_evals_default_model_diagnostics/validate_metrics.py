"""Check intervention scoring against the canonical exp89 evaluator on saved votes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import remaining_metrics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp89_evals_contacts_v1_model_on_eval_set"))
from compute_metrics import metric_rows, resolved_pairs, true_matrix


def main() -> None:
    """Remove supplied candidate pairs before calling the unmodified evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    targets = {target["stem"]: target for target in plan["targets"]}
    rows = []
    for marker in sorted(args.run.rglob("*.complete.json")):
        metadata = json.loads(marker.read_text())
        target = targets[metadata["stem"]]
        supplied = target["orders"][f"{metadata['policy']}:{metadata['repeat']}"][:metadata["n_supplied"]]
        with np.load(str(marker).replace(".complete.json", ".npz")) as archive:
            scores = archive["score"].astype(float)
        pi, pj, separation = resolved_pairs(np.array(target["resolved"]))
        excluded = {tuple(pair) for pair in supplied}
        keep = np.array([(i, j) not in excluded for i, j in zip(pi, pj)])
        truth = true_matrix(target["L"], [(i, j, 1.) for i, j in target["truth"]])
        canonical = metric_rows(scores, truth, pi[keep], pj[keep], separation[keep], target["L"], with_precision=True)
        for contact_range, minimum in (("all", 6), ("long", 24)):
            expected = next(row for row in canonical if row["range"] == contact_range and row["cut"] == "R")
            actual = remaining_metrics(scores, target["truth"], target["resolved"], supplied, minimum)
            actual_precision = actual["r_precision"] if actual["r_precision"] is not None else float("nan")
            if not np.isclose(actual_precision, expected["precision"], rtol=0, atol=0, equal_nan=True):
                raise ValueError(f"R-precision mismatch: {marker} {contact_range}")
            if actual["n_candidates"] != expected["n_candidate"] or actual["n_remaining_true"] != expected["n_true"]:
                raise ValueError(f"Candidate/truth universe mismatch: {marker} {contact_range}")
            rows.append({"stem": target["stem"], "mode": metadata["mode"], "range": contact_range,
                         "canonical_r_precision": expected["precision"], "actual_r_precision": actual_precision,
                         "n_candidates": expected["n_candidate"], "n_remaining_true": expected["n_true"], "exact_match": True})
    if not rows:
        raise ValueError("No completed artifacts to validate")
    pd.DataFrame(rows).to_csv(HERE / "data/canonical_metric_validation.csv", index=False)
    print(f"{len(rows)} exact matches against the unmodified exp89 evaluator")


if __name__ == "__main__":
    main()
