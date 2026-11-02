# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A2 on the A4 panel: which matched pairs get the contact Jacobian.

The A4 Jacobian subset (plan section 5.4) runs the contact Jacobian under
exp199, which trained on the #225-removed structures, and exp232, which
never saw them. If exp199's predictions on removed structures depend on the
chain less locally than exp232's on the same structures, while the matched
kept structures show no such difference, non-locality is the signature of
recall in this model, and A2's locality on the eval proteins can be read
against it.

Cost grows as L^3 per protein, so the subset takes matched pairs whose two
members are both at most ``MAX_LENGTH`` residues: ``PAIRS_PER_BIN`` per
relative-count bin, seeded. Bin 0 has exactly 20 such pairs, and they are
much shorter than bin 0's median (546), so the bins differ in length here
as they do in the full panel.

``est_l40s_seconds`` is the expected fp32 cost: a2_select.py's bf16
estimate times ``FP32_FACTOR``, the fp32/bf16 time ratio measured on nine
proteins of length 70-130 (1.2-1.9, data/a2_precision_check.json).

Writes ``data/a2_panel_proteins.csv``.

    uv run python a2_panel_select.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from a2_select import readout_seconds

DATA = Path(__file__).resolve().parent / "data"
MAX_LENGTH = 250
PAIRS_PER_BIN = 20
FP32_FACTOR = 1.8
SEED = 341


def main() -> int:
    panel = pd.read_parquet(DATA / "panel.parquet", columns=["set", "entry_id", "seq_len", "bin", "pair", "n_relatives"])
    longest = panel.groupby("pair")["seq_len"].max()
    eligible = panel[panel["pair"].isin(longest[longest <= MAX_LENGTH].index)]
    rng = np.random.default_rng(SEED)
    chosen = []
    for _, pairs in eligible.drop_duplicates("pair").groupby("bin", sort=False):
        if len(pairs) < PAIRS_PER_BIN:
            raise ValueError(f"bin {pairs['bin'].iloc[0]}: only {len(pairs)} eligible pairs")
        chosen.extend(pairs["pair"].iloc[np.sort(rng.choice(len(pairs), PAIRS_PER_BIN, replace=False))])
    table = eligible[eligible["pair"].isin(chosen)].rename(columns={"entry_id": "key", "seq_len": "L"})
    table["set"] = "panel-" + table["set"]
    table["est_l40s_seconds"] = FP32_FACTOR * (1 + 4 * table["L"]) * readout_seconds(table["L"].to_numpy())
    table = table[["set", "key", "L", "bin", "pair", "n_relatives", "est_l40s_seconds"]].sort_values(["bin", "pair", "set"])
    table.to_csv(DATA / "a2_panel_proteins.csv", index=False, float_format="%.6g")
    print(table.groupby(["bin", "set"]).agg(n=("key", "size"), median_L=("L", "median")))
    print(f"{len(table)} structures; {table['est_l40s_seconds'].sum() / 3600:.1f} fp32 L40S-hours per model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
