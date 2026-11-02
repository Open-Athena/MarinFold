# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""B1 tuned lens: R-precision of the contact readout decoded at each layer boundary.

Reads ``b1_tuned_lens.py``'s outputs from one or more ``--input-dir``
(protein sets or shards sharing one fitted lens): concatenates
``scores/*.csv`` and ``timings/*.csv`` into ``data/<prefix>_scores.csv``
and ``data/<prefix>_timings.csv``, copies the fitting run's
``lens_fit.csv`` to ``data/<prefix>_fit.csv``, and writes
``data/<prefix>_curve.csv``.

Per lens (logit, tuned), boundary (0 = embeddings, k = after decoder layer
k - 1, 24 = the model), separation band and split (all proteins, each eval
set, each length tercile or ``--length-bins`` bin, as in ``analyze_b1.py``),
over the proteins with true contacts in the band:

- ``rp``: mean R-precision of the seed-averaged P(contact), with a protein
  bootstrap 95% CI (``lo``, ``hi``).
- ``gap``: ``rp`` minus the boundary-24 value on the same proteins, with a
  paired protein bootstrap CI (``gap_lo``, ``gap_hi``). Decision B reads
  the tuned lens's gap at boundary 18 and its rise over boundaries 18 to 24.

No threshold is applied.

    uv run python analyze_b1_lens.py --input-dir out/lens
    uv run python analyze_b1_lens.py --input-dir out/lens_ext out/lens_panel_* --prefix b1_lens_ext \
        --length-bins 0 150 300 500 1000
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_b1 import BANDS, N_BOOT, SEED, bootstrap, load, splits

DATA = Path(__file__).resolve().parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", default="b1_lens", help="output names data/<prefix>_{scores,timings,fit,curve}.csv")
    parser.add_argument("--length-bins", type=int, nargs="+", default=None,
                        help="fixed length-bin edges instead of terciles")
    args = parser.parse_args()

    scores = load(args.input_dir, "scores")
    if scores.duplicated(["lens", "boundary", "stem", "seed"]).any():
        raise ValueError("a lens, boundary, protein and seed appear in more than one input")
    scores.to_csv(DATA / f"{args.prefix}_scores.csv", index=False)
    load(args.input_dir, "timings").to_csv(DATA / f"{args.prefix}_timings.csv", index=False)
    fits = [d / "lens_fit.csv" for d in args.input_dir if (d / "lens_fit.csv").exists()]
    if len(fits) != 1:
        raise ValueError(f"expected exactly one input with lens_fit.csv (the fitting run), found {len(fits)}")
    shutil.copyfile(fits[0], DATA / f"{args.prefix}_fit.csv")

    mean = scores[scores.seed == "mean"]
    final = mean.boundary.max()
    proteins = (mean[(mean.lens == "logit") & (mean.boundary == final)][["stem", "eval_set", "n_residues"]]
                .sort_values("stem").reset_index(drop=True))
    stems = proteins.stem.to_numpy()
    masks = splits(proteins, args.length_bins)
    indexed = mean.set_index(["lens", "boundary", "stem"]).sort_index()

    rng = np.random.default_rng(SEED)
    rows = []
    for band in BANDS:
        column = f"rp_{band}"
        model = indexed.loc[("logit", final), column].loc[stems].to_numpy()
        for lens in ("logit", "tuned"):
            for boundary in range(final + 1):
                rp = indexed.loc[(lens, boundary), column].loc[stems].to_numpy()
                finite = np.isfinite(rp) & np.isfinite(model)
                for split, split_mask in masks.items():
                    mask = split_mask & finite
                    if not mask.any():
                        continue
                    lo, hi = bootstrap(rp[mask], rng)
                    gap_lo, gap_hi = bootstrap(rp[mask] - model[mask], rng)
                    rows.append({
                        "lens": lens, "boundary": boundary, "band": band, "split": split,
                        "n_proteins": int(mask.sum()), "rp": rp[mask].mean(), "lo": lo, "hi": hi,
                        "gap": (rp[mask] - model[mask]).mean(), "gap_lo": gap_lo, "gap_hi": gap_hi,
                    })
    curve = pd.DataFrame(rows)
    curve.to_csv(DATA / f"{args.prefix}_curve.csv", index=False, float_format="%.5g")

    table = curve[(curve.split == "all")].pivot_table(index="boundary", columns=["lens", "band"], values="rp")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table)
    print(f"{N_BOOT} bootstrap draws, seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
