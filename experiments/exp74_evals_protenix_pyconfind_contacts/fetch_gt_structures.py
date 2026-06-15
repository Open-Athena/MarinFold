# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch ground-truth structures for the exp65 eval set into exp65's tree.

Downloads the RCSB mmCIF for each de-novo / CAMEO candidate to the exact
``local_path`` its exp65 manifest specifies (so ``contact_eval.py`` finds
it via ``--gt-root <exp65 dir>``). Idempotent. CASP-FM domains are *not*
handled here — they are clipped from CASP tarballs by exp65's
``fetch_casp_fm.py`` (run separately); this fetcher reports them as
pending so the count is honest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

EXP65 = Path(__file__).resolve().parent.parent / "exp65_evals_low_msa_depth_proteins"
RCSB_CIF = "https://files.rcsb.org/download/{pdb}.cif"


def fetch(eval_manifest: Path) -> None:
    df = pd.read_csv(eval_manifest, dtype=str).fillna("")
    n_ok = n_skip = n_fail = n_casp = 0
    for r in df.itertuples(index=False):
        rec = r._asdict()
        dataset, pdb_local = rec["dataset"], rec["gt_cif"]
        target = EXP65 / pdb_local
        if target.exists() and target.stat().st_size > 0:
            n_skip += 1
            continue
        if dataset == "casp_fm":
            n_casp += 1
            continue
        # pdb_id is the RCSB id for denovo / cameo.
        pdb = rec["gt_cif"].split("/")[-1].removesuffix(".cif")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(RCSB_CIF.format(pdb=pdb), timeout=60)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  fetched {n_ok} ...", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {rec['stem']} ({pdb}): {e}")
            n_fail += 1
    print(f"done: fetched {n_ok}, skipped {n_skip} (already present), "
          f"failed {n_fail}, casp_fm pending {n_casp} (use exp65/fetch_casp_fm.py)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-manifest", type=Path, default=Path("data/eval_manifest_exp65.csv"))
    args = ap.parse_args()
    fetch(args.eval_manifest)
