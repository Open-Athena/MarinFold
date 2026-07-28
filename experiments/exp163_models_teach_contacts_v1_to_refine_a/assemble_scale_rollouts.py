# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Merge the exp163 rollout runs into one local table for corpus building (#163).

The 50k-protein sweep corpus is built from **two** generation runs, because the
first 10k batch was already done and there was no reason to pay for it twice:

* ``val10k/runs/rollout_metrics``  — 9,375 proteins x 24 (the original batch)
* ``scale50k/runs/rollout_metrics`` — 40,625 proteins x 24 (this scale-up)

The two draws come from the same ESM-Atlas corpus under identical filters
(``plddt>=80``, ``L<=512``, ``>=5`` contacts) and are statistically
indistinguishable — L median 206, n_gt median 173, pLDDT 84.2 for both — so
concatenating them is unbiased.

Why they had to be two draws at all: ``select_targets_esm_atlas.py`` is NOT
reproducible across runs. It seeds a shuffle of the shard list, but
``HfFileSystem.glob`` does not return a stable order, so the same seed yields a
different draw — a fresh 50k selection had ZERO overlap with the existing 10k.
The 50k target list was therefore pinned explicitly (the 9,375 already-generated
proteins + 40,625 fresh) rather than re-derived. (Fixing the selector to sort
before shuffling would make future draws reproducible.)

Guards against the failure modes this pipeline actually hit:
  * a protein present in targets but with NO rollouts (a shard that died before
    flushing — see the FLUSH_EVERY note in ``gen_rollouts_worker_exp163.py``);
  * duplicate ``(entry_id, r)`` rows, which is what a *concurrent* retry of an
    in-flight shard would produce.

    uv run python assemble_scale_rollouts.py \\
        --targets targets_final50k.parquet --out-rollouts rollouts_50k.parquet
"""
from __future__ import annotations

import argparse
import os

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S3_KW = {"s3": {"addressing_style": "virtual"}}
COLS = ["entry_id", "r", "pred", "all_f1"]
DEFAULT_RUNS = (
    "s3://marin-us-east-02a/MarinFold/exp163/val10k/runs/rollout_metrics",
    "s3://marin-us-east-02a/MarinFold/exp163/scale50k/runs/rollout_metrics",
)


def _fs(url: str):
    if not url.startswith("s3://"):
        return fsspec.filesystem("file")
    return fsspec.filesystem("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                             config_kwargs=S3_KW)


def _strip(url: str) -> str:
    return url.split("://", 1)[1] if "://" in url else url


def read_run(url: str) -> pd.DataFrame:
    fs = _fs(url)
    parts = sorted(fs.glob(f"{_strip(url).rstrip('/')}/*.parquet"))
    if not parts:
        raise SystemExit(f"no part files under {url}")
    frames = []
    for p in parts:
        with fs.open(p, "rb") as fh:
            frames.append(pq.read_table(fh, columns=COLS).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    print(f"  {url}\n    {len(parts)} parts, {len(df):,} rollouts, "
          f"{df.entry_id.nunique():,} proteins", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-rollouts", required=True)
    ap.add_argument("--out-targets", default=None,
                    help="targets restricted to proteins that actually have rollouts")
    a = ap.parse_args()

    print("[exp163] reading rollout runs", flush=True)
    df = pd.concat([read_run(u) for u in a.runs], ignore_index=True)

    dupes = int(df.duplicated(["entry_id", "r"]).sum())
    if dupes:
        print(f"  WARNING: {dupes:,} duplicate (entry_id, r) rows — dropping "
              f"(a concurrent retry of an in-flight shard does this)", flush=True)
        df = df.drop_duplicates(["entry_id", "r"], keep="first").reset_index(drop=True)

    tgt = pd.read_parquet(a.targets)
    have = set(df.entry_id.unique())
    missing = [e for e in tgt.entry_id if e not in have]
    per = df.groupby("entry_id").size()

    print(f"\n[exp163] merged: {len(df):,} rollouts over {len(have):,} proteins")
    print(f"         targets: {len(tgt):,}  with rollouts: {len(tgt)-len(missing):,}  "
          f"MISSING: {len(missing):,}")
    print(f"         rollouts/protein: min={per.min()} median={int(per.median())} max={per.max()}")
    print(f"         mean single-rollout all_f1: {df.all_f1.mean():.4f}")

    df.to_parquet(a.out_rollouts, index=False)
    print(f"\n[exp163] wrote {a.out_rollouts}")
    if a.out_targets:
        kept = tgt[tgt.entry_id.isin(have)].reset_index(drop=True)
        kept.to_parquet(a.out_targets, index=False)
        print(f"[exp163] wrote {a.out_targets} ({len(kept):,} targets)")


if __name__ == "__main__":
    main()
