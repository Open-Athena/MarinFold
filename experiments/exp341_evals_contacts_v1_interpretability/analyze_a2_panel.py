# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A2 on the A4 panel: does training on a structure make its contact Jacobian less local?

Reads ``a2_contact_jacobian.py --proteins panel`` outputs for exp199 and
exp232 (``--exp199-dir``, ``--exp232-dir``; fp32 runs) and
``data/a2_panel_proteins.csv``. Per structure and model, locality and the
structural-neighbour share are computed as in ``analyze_a2.py`` on both
effect scales, with precision at L of the same top-L predicted contacts.

Per matched pair the quantity is the difference in differences

    (exp199 - exp232 on the removed structure) - (exp199 - exp232 on its kept control)

exp199 trained on the removed structure and exp232 did not; both trained on
the kept one, so the kept term absorbs how the two models differ overall.
A negative DiD in locality means exposure spreads the Jacobian over the
chain, which is the signature the plan attributes to recall. CIs bootstrap
pairs within bin (``N_BOOT``, seeded); "all" pools the bins.

Locality tracks precision across structures, so a locality DiD could come
from exp199 predicting a different top-L set. ``locality_shared_{scale}``
recomputes locality per model on only the contacts in both models' top L,
which holds the contacts fixed and compares how each model computes them.

Outputs ``data/a2_panel_per_structure.csv`` and ``data/a2_panel_did.csv``.

    uv run python analyze_a2_panel.py --exp199-dir out/a2_panel/exp199 --exp232-dir out/a2_panel/exp232
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import harness
from analyze_a2 import NEIGHBOURHOOD, SCALES, effects, protein_rows
from marinfold.document_structures.contacts_v1.inference import gt_contact_matrix

DATA = Path(__file__).resolve().parent / "data"
METRICS = [*(f"locality_{scale}" for scale in SCALES), *(f"locality_shared_{scale}" for scale in SCALES),
           *(f"structural_{scale}" for scale in SCALES), "precision_at_l"]
N_BOOT = 10_000
SEED = 341


def shared_locality(arrays: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Per model and scale, mean locality over the contacts in every model's top L."""
    index = {model: {tuple(pair): k for k, pair in enumerate(a["top_pairs"])} for model, a in arrays.items()}
    shared = np.array(sorted(set.intersection(*(set(keys) for keys in index.values()))))
    out = {}
    for model, a in arrays.items():
        columns = [index[model][tuple(pair)] for pair in shared]
        positions = np.arange(a["dp_top"].shape[0])[:, None]
        near = np.minimum(np.abs(positions - shared[None, :, 0]), np.abs(positions - shared[None, :, 1])) <= NEIGHBOURHOOD
        out[model] = {}
        for scale, effect in effects(a).items():
            effect = effect[:, columns]
            out[model][f"locality_shared_{scale}"] = np.nanmean((effect * near).sum(axis=0) / effect.sum(axis=0))
    return out


def per_structure(table: pd.DataFrame, directories: dict[str, Path]) -> pd.DataFrame:
    proteins = {p.stem: p for p in harness.load_panel_proteins(DATA / "panel.parquet") if p.stem in set(table["key"])}
    rows = []
    for key in table["key"]:
        paths = {model: directory / "proteins" / f"{key}.npz" for model, directory in directories.items()}
        if not all(path.exists() for path in paths.values()):
            continue
        protein = proteins[key]
        gt = gt_contact_matrix(protein.structure.gt_contacts, protein.length, harness.MIN_SEQ_SEPARATION)
        gt = gt | gt.T
        arrays = {model: dict(np.load(path)) for model, path in paths.items()}
        shared = shared_locality(arrays)
        for model, a in arrays.items():
            is_true = gt[a["top_pairs"][:, 0], a["top_pairs"][:, 1]]
            row, _ = protein_rows(key, a, gt, is_true)
            rows.append({"model": model, **row, **shared[model]})
    return table.merge(pd.DataFrame(rows), on="key", how="inner")


def did_table(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    wide = frame.pivot(index=["bin", "pair"], columns=["model", "set"], values=METRICS)
    out = []
    for metric in METRICS:
        m = wide[metric]
        removed = m[("exp199", "panel-removed")] - m[("exp232", "panel-removed")]
        kept = m[("exp199", "panel-kept")] - m[("exp232", "panel-kept")]
        did = (removed - kept).dropna()
        for name, values in [("all", did), *did.groupby(level="bin")]:
            values = values.to_numpy()
            draws = rng.choice(values, size=(N_BOOT, len(values))).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            out.append({"metric": metric, "bin": name, "n_pairs": len(values),
                        "removed_delta": removed.loc[did.index].loc[name].mean() if name != "all" else removed.loc[did.index].mean(),
                        "kept_delta": kept.loc[did.index].loc[name].mean() if name != "all" else kept.loc[did.index].mean(),
                        "did": values.mean(), "did_lo": low, "did_hi": high})
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp199-dir", type=Path, required=True)
    parser.add_argument("--exp232-dir", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(DATA / "a2_panel_proteins.csv", dtype={"bin": str})
    frame = per_structure(table, {"exp199": args.exp199_dir, "exp232": args.exp232_dir})
    frame.to_csv(DATA / "a2_panel_per_structure.csv", index=False, float_format="%.6g")
    did = did_table(frame, np.random.default_rng(SEED))
    did.to_csv(DATA / "a2_panel_did.csv", index=False, float_format="%.6g")

    print(frame.groupby(["model", "set"])[METRICS].mean().round(3))
    print(did[did["bin"] == "all"].round(4).to_string(index=False))
    print(did[did["metric"] == "locality_raw"].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
