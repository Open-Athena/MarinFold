# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine the exp78 ESM scores with exp74's Protenix scores into one table.

Two jobs:

1. Concatenate exp78's per-dataset (FoldBench-100 + exp65) ESM
   ``contact_precision.csv`` / ``contact_eval_meta.csv`` /
   ``contacts_raw.parquet`` into ``*_all`` files (``--esm-dirs``).

2. Splice in exp74's Protenix ``contact_precision_all.csv``
   (``--protenix-precision``), stamping ``model="protenix-v2"`` on every
   Protenix row, to produce the unified comparison table
   ``contact_precision_all.csv`` keyed by ``(model, mode, predictor)``:

       protenix-v2 / single_seq / {distogram, structure}
       protenix-v2 / msa        / {distogram, structure}
       esmfold     / single_seq / structure
       esmfold2    / single_seq / structure

   The success-criteria 4-bar comparison is the ``predictor == "structure"``
   slice: protenix-v2·single_seq, protenix-v2·msa, esmfold, esmfold2.

FoldBench rows carry no exp65 strata columns (``neff_tier`` etc.); pandas
fills them with NaN on concat, which the plots already handle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine(esm_dirs: list[Path], out_dir: Path, protenix_precision: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _concat_csv(name: str) -> pd.DataFrame:
        frames = [pd.read_csv(d / name) for d in esm_dirs if (d / name).exists()]
        if not frames:
            raise FileNotFoundError(f"no {name} found under {[str(d) for d in esm_dirs]}")
        return pd.concat(frames, ignore_index=True)

    esm_prec = _concat_csv("contact_precision.csv")
    meta = _concat_csv("contact_eval_meta.csv")
    meta.to_csv(out_dir / "contact_eval_meta_all.csv", index=False)

    raw_frames = [pd.read_parquet(d / "contacts_raw.parquet")
                  for d in esm_dirs if (d / "contacts_raw.parquet").exists()]
    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_parquet(
            out_dir / "contacts_raw_all.parquet", index=False)

    # Unified comparison table: ESM rows (already carry `model`) + Protenix.
    frames = [esm_prec]
    if protenix_precision is not None and Path(protenix_precision).exists():
        prot = pd.read_csv(protenix_precision)
        if "model" not in prot.columns:
            prot.insert(0, "model", "protenix-v2")
        frames.append(prot)
        print(f"  + protenix-v2 rows from {protenix_precision}: {len(prot)}")
    else:
        print(f"  WARN: no Protenix precision CSV at {protenix_precision}; "
              "combined table has ESM models only.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "contact_precision_all.csv", index=False)

    print(f"combined {len(esm_dirs)} ESM datasets + protenix:")
    print(f"  contact_precision_all.csv: {len(combined)} rows, "
          f"{combined.stem.nunique()} stems, "
          f"models={sorted(combined.model.dropna().unique())}, "
          f"datasets={sorted(combined.dataset.unique())}")
    print(f"  contact_eval_meta_all.csv: {len(meta)} ESM stems")
    if raw_frames:
        print(f"  contacts_raw_all.parquet: {sum(len(f) for f in raw_frames)} ESM contact rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--esm-dirs", nargs="+", type=Path, required=True,
                    help="exp78 per-dataset ESM score dirs (each with contact_precision.csv etc.).")
    ap.add_argument("--protenix-precision", type=Path, default=None,
                    help="exp74 contact_precision_all.csv to splice in (model=protenix-v2).")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    combine(args.esm_dirs, args.out, args.protenix_precision)
