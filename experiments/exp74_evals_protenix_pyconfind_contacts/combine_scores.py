# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine FoldBench-100 + exp65 contact-eval outputs into unified tables.

Concatenates the per-dataset ``contact_precision.csv`` /
``contact_eval_meta.csv`` / ``contacts_raw.parquet`` into single
``*_all`` files. FoldBench rows carry no exp65 strata columns
(``neff_tier`` etc.); pandas fills them with NaN on concat, which the
plots already handle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine(dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _concat_csv(name: str) -> pd.DataFrame:
        frames = [pd.read_csv(d / name) for d in dirs if (d / name).exists()]
        if not frames:
            raise FileNotFoundError(f"no {name} found under {[str(d) for d in dirs]}")
        return pd.concat(frames, ignore_index=True)

    prec = _concat_csv("contact_precision.csv")
    prec.to_csv(out_dir / "contact_precision_all.csv", index=False)
    meta = _concat_csv("contact_eval_meta.csv")
    meta.to_csv(out_dir / "contact_eval_meta_all.csv", index=False)

    raw_frames = [pd.read_parquet(d / "contacts_raw.parquet")
                  for d in dirs if (d / "contacts_raw.parquet").exists()]
    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_parquet(out_dir / "contacts_raw_all.parquet", index=False)

    print(f"combined {len(dirs)} datasets:")
    print(f"  contact_precision_all.csv: {len(prec)} rows, "
          f"{prec.stem.nunique()} stems, datasets={sorted(prec.dataset.unique())}")
    print(f"  contact_eval_meta_all.csv: {len(meta)} stems")
    if raw_frames:
        print(f"  contacts_raw_all.parquet: {sum(len(f) for f in raw_frames)} contact rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="+", type=Path, required=True,
                    help="Per-dataset score dirs (each with contact_precision.csv etc.).")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    combine(args.dirs, args.out)
