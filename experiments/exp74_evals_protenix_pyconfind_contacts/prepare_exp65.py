# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Protenix inputs + the eval manifest for the exp65 candidate set.

Everything is derived from exp65's committed CSVs — no structure download
needed to launch the Modal run (Protenix only needs sequences). We use
``candidate_sequences.csv`` as the input sequence (the curated set exp65
already validated its Neff / leakage analysis against), and carry the
3-axis novelty labels (``neff_tier``, ``fold_verdict``, ``seq_leakage``,
``msa_neff``) through onto the eval manifest as strata.

Writes:
  - ``inputs/jobs/<stem>.json``      one Protenix job per candidate
  - ``inputs/manifest.csv``          the run manifest (modal_app reads this)
  - ``eval_manifest_exp65.csv``      what contact_eval.py consumes; ``gt_cif``
                                     points at the exp65 ``structures/`` tree
                                     (populated separately by exp65's fetchers)

The 3 stems present in both the de-novo and CAMEO sources are de-duplicated
(kept once, first occurrence).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXP65 = Path(__file__).resolve().parent.parent / "exp65_evals_low_msa_depth_proteins"
EXP65_DATA = EXP65 / "data"

# Per-source manifests carry the local structure path (relative to the exp65
# dir) and the author chain — needed for ground-truth scoring later.
_SOURCE_MANIFESTS = {
    "denovo_pdb": "denovo_pdb_manifest.csv",
    "casp_fm": "casp_fm_manifest.csv",
    "cameo_hard": "cameo_hard_manifest.csv",
}


def _load_inputs() -> pd.DataFrame:
    label = pd.read_csv(EXP65_DATA / "candidate_2d_label.csv", dtype=str)
    seqs = pd.read_csv(EXP65_DATA / "candidate_sequences.csv", dtype=str)

    # local_path + chain from each source manifest (stem is the join key).
    man_parts = []
    for df_name in _SOURCE_MANIFESTS.values():
        m = pd.read_csv(EXP65_DATA / df_name, dtype=str)
        man_parts.append(m[["stem", "chain", "local_path"]])
    manifests = pd.concat(man_parts, ignore_index=True).drop_duplicates("stem")

    df = label.merge(seqs[["stem", "chain", "sequence"]], on="stem", how="left")
    df = df.merge(manifests[["stem", "local_path"]], on="stem", how="left")
    # Dedup the 3 stems that appear in two sources (identical structure).
    df = df.drop_duplicates("stem", keep="first").reset_index(drop=True)
    return df


def _protenix_job(stem: str, sequence: str) -> dict:
    return {
        "name": stem,
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
        "covalent_bonds": [],
    }


def prepare(out_dir: Path, eval_manifest: Path) -> None:
    df = _load_inputs()

    n_missing_seq = df["sequence"].isna().sum() + (df["sequence"].fillna("").str.len() == 0).sum()
    df = df[df["sequence"].fillna("").str.len() > 0].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jobs").mkdir(exist_ok=True)

    run_rows: list[dict] = []
    eval_rows: list[dict] = []
    n_with_x = 0
    for r in df.itertuples(index=False):
        rec = r._asdict()
        stem = rec["stem"]
        seq = rec["sequence"].strip().upper()
        if "X" in seq:
            n_with_x += 1
        (out_dir / "jobs" / f"{stem}.json").write_text(json.dumps([_protenix_job(stem, seq)], indent=2))
        run_rows.append({
            "stem": stem,
            "pdb_id": rec["pdb_id"],
            "n_residues": len(seq),
            "job_json": f"jobs/{stem}.json",
            "dataset": rec["dataset"],
        })
        eval_rows.append({
            "dataset": rec["dataset"],          # denovo_pdb / casp_fm / cameo_hard
            "stem": stem,
            "gt_cif": rec["local_path"],         # relative to the exp65 dir
            "gt_chain": rec.get("chain") or "",
            "input_seq": seq,
            # strata for downstream stratified plots:
            "neff_tier": rec.get("neff_tier", ""),
            "fold_verdict": rec.get("fold_verdict", ""),
            "seq_leakage": rec.get("seq_leakage", ""),
            "msa_neff": rec.get("msa_neff", ""),
            "length": rec.get("length", ""),
        })

    manifest_csv = out_dir / "manifest.csv"
    pd.DataFrame(run_rows).to_csv(manifest_csv, index=False)
    pd.DataFrame(eval_rows).to_csv(eval_manifest, index=False)

    print(f"candidates: {len(df)} unique (dropped {int(n_missing_seq)} missing-seq); "
          f"{n_with_x} have non-standard 'X' residues")
    print(f"  by dataset: {df['dataset'].value_counts().to_dict()}")
    print(f"  length: min {df['sequence'].str.len().min()} max {df['sequence'].str.len().max()}")
    print(f"wrote {manifest_csv} + {out_dir/'jobs'}/*.json ({len(run_rows)} jobs)")
    print(f"wrote {eval_manifest} ({len(eval_rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("inputs"), help="Input dir for jobs + manifest.")
    ap.add_argument("--eval-manifest", type=Path, default=Path("data/eval_manifest_exp65.csv"))
    args = ap.parse_args()
    prepare(args.out, args.eval_manifest)
