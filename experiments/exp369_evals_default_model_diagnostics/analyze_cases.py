"""Compare unconditioned MarinFold support for AFDB and experimental-only contacts."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

from protocol import remaining_metrics

HERE = Path(__file__).resolve().parent


def exclusive_preference(scores: np.ndarray, af_only: list, pdb_only: list) -> float | None:
    """Return probability that a PDB-only contact outranks an AFDB-only contact.

    This within-model contrast treats the two contradictory label sets equally,
    irrespective of their sizes; ties receive half credit. It is distinct from
    comparing full-map R-precision with different R or from cross-model AUC.
    """
    if not af_only or not pdb_only:
        return None
    af = np.array([scores[i, j] for i, j in af_only])
    pdb = np.array([scores[i, j] for i, j in pdb_only])
    return float(((pdb[:, None] > af[None, :]) + .5 * (pdb[:, None] == af[None, :])).mean())


def main() -> None:
    """Require complete valid predictions for all six cases before comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "data/afdb_inference_plan.json")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    fs, root = fsspec.core.url_to_fs(args.run)
    rows, timings = [], []
    for target in plan["targets"]:
        matrices = []
        for repeat in range(plan["repeats"]):
            base = f"{root}/units/{target['stem']}/random-r{repeat}-k0000"
            with fs.open(base + ".complete.json") as stream:
                marker = json.load(stream)
            if marker["plan_sha256"] != digest or marker["n_rollouts"] != 100:
                raise ValueError(f"Invalid case provenance: {base}")
            if any(marker[key] for key in ("n_unfinished", "n_malformed", "n_out_of_range")):
                raise ValueError(f"Invalid case completion: {base}")
            with fs.open(base + ".npz", "rb") as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != marker["files_sha256"][".npz"]:
                raise ValueError("Case vote checksum mismatch")
            with np.load(io.BytesIO(payload)) as archive:
                matrices.append(archive["score"].astype(float) / 100)
            timings.append({k: v for k, v in marker.items() if k != "files_sha256"})
        for repeat, scores in enumerate(matrices):
            af = remaining_metrics(scores, target["afdb_pairs"], target["resolved"], [])
            pdb = remaining_metrics(scores, target["pdb_pairs"], target["resolved"], [])
            mean_af = float(np.mean([scores[i, j] for i, j in target["afdb_only"]]))
            mean_pdb = float(np.mean([scores[i, j] for i, j in target["pdb_only"]]))
            rows.append({"uniprot": target["uniprot"], "pdb": target["pdb"], "chain": target["chain"], "repeat": repeat,
                         "afdb_r_precision": af["r_precision"], "pdb_r_precision": pdb["r_precision"],
                         "afdb_only_mean_vote": mean_af, "pdb_only_mean_vote": mean_pdb,
                         "exclusive_vote_delta_pdb_minus_afdb": mean_pdb - mean_af,
                         "pdb_only_ranking_preference": exclusive_preference(scores, target["afdb_only"], target["pdb_only"]),
                         "interpretation": target["interpretation"]})
    pd.DataFrame(rows).to_csv(HERE / "data/afdb_marinfold_per_repeat.csv", index=False)
    pd.DataFrame(timings).to_csv(HERE / "data/afdb_timings.csv", index=False)
    summary = pd.DataFrame(rows).groupby(["uniprot", "pdb", "chain", "interpretation"], as_index=False).mean(numeric_only=True).drop(columns="repeat")
    summary.to_csv(HERE / "data/afdb_marinfold_preference.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
