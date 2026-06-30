# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp98 rollout artifacts to the **public** open-athena/MarinFold HF
bucket so anyone (incl. an auth-free Colab) can read them.

The per-target run lives on GCS (`runs/full/`, auth-required). This consolidates
it into a few parquets and uploads them to the public bucket:

  data/contacts-v1-train-rollouts-exp98/
    per_target_summary.parquet   1000 rows — L, n_gt, best-of-1000 recall/F1, timing
    best_rollouts.parquet        1000 rows — best-recall + best-F1 rollouts (full docs)
    rollout_metrics_all.parquet  1,000,000 rows — every rollout's per-band metrics
    README.md

    HF_TOKEN=<open-athena-scoped> uv run python publish_to_hf.py \
        --run gs://marin-us-east5/protein-structure/MarinFold/exp98_rollouts_contacts_v1_train/runs/full \
        --targets data/targets.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-train-rollouts-exp98"


def find_hf() -> str:
    """An ``hf`` binary that supports ``buckets`` (the one shadowing PATH inside a
    uv venv may be too old). Scan PATH, skipping venv dirs, for one that does."""
    cands, seen = [], set()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "hf")
        if os.path.exists(p) and p not in seen and ".venv" not in p and "/venv/" not in p:
            seen.add(p); cands.append(p)
    w = shutil.which("hf")
    if w:
        cands.append(w)
    for hf in cands:
        try:
            r = subprocess.run([hf, "buckets", "--help"], capture_output=True)
            if r.returncode == 0:
                return hf
        except OSError:
            continue
    raise RuntimeError("no `hf` with `buckets` support found on PATH "
                       "(need huggingface_hub CLI with bucket commands)")


def _read_metrics(fs, p):
    entry = os.path.basename(p)[: -len(".parquet")]
    with fs.open(p, "rb") as fh:
        m = pq.read_table(fh).to_pandas()
    m.insert(0, "entry_id", entry)
    return m


def _read_best(fs, p):
    with fs.open(p, "r") as fh:
        d = json.load(fh)
    row = dict(entry_id=d["entry_id"], L=d["L"], n_gt=d["n_gt"],
               gt_short=d["gt_by_band"]["short"], gt_med=d["gt_by_band"]["med"],
               gt_long=d["gt_by_band"]["long"], n_rollouts=d["n_rollouts"])
    for which in ("best_recall", "best_f1"):
        b = d[which]
        row[f"{which}_r"] = b["r"]
        row[f"{which}_recall"] = b["recall"]
        row[f"{which}_precision"] = b["precision"]
        row[f"{which}_f1"] = b["f1"]
        row[f"{which}_n_gen_tokens"] = b["n_gen_tokens"]
        row[f"{which}_finished"] = b["finished"]
        row[f"{which}_document"] = b["document"]
        row[f"{which}_pred_contacts"] = json.dumps(b["pred_contacts"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GCS run dir (runs/full)")
    ap.add_argument("--summary", default=None,
                    help="per_target_summary.parquet (defaults to <run>/per_target_summary.parquet)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--dest", default=BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="build locally, skip upload")
    a = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(a.run)
    run = a.run.rstrip("/")
    mpaths = [p for p in fs.ls(f"{run}/rollout_metrics", detail=False) if p.endswith(".parquet")]
    bpaths = [p for p in fs.ls(f"{run}/best_rollouts", detail=False) if p.endswith(".json")]
    print(f"{len(mpaths)} rollout_metrics, {len(bpaths)} best_rollouts", flush=True)

    out = tempfile.mkdtemp(prefix="exp98pub_")

    # 1. best_rollouts.parquet
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        best = pd.DataFrame(list(ex.map(lambda p: _read_best(fs, p), bpaths)))
    best = best.sort_values("entry_id").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(best, preserve_index=False), f"{out}/best_rollouts.parquet")
    print(f"best_rollouts.parquet: {len(best)} rows", flush=True)

    # 2. rollout_metrics_all.parquet (sorted by entry_id for filtered reads)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        parts = list(ex.map(lambda p: _read_metrics(fs, p), mpaths))
    allm = pd.concat(parts, ignore_index=True).sort_values(["entry_id", "r"]).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(allm, preserve_index=False),
                   f"{out}/rollout_metrics_all.parquet", row_group_size=50_000)
    print(f"rollout_metrics_all.parquet: {len(allm)} rows", flush=True)

    # 3. per_target_summary.parquet
    summary = a.summary or f"{run}/per_target_summary.parquet"
    with fsspec.open(summary, "rb") as fh:
        pq.write_table(pq.read_table(fh), f"{out}/per_target_summary.parquet")
    print("per_target_summary.parquet copied", flush=True)

    # 4. README
    with open(f"{out}/README.md", "w") as fh:
        fh.write(README)

    for f in ("per_target_summary.parquet", "best_rollouts.parquet",
              "rollout_metrics_all.parquet", "README.md"):
        sz = os.path.getsize(f"{out}/{f}") / 1e6
        print(f"  {f}: {sz:.1f} MB", flush=True)

    if a.dry_run:
        print(f"dry-run: built in {out}")
        return 0

    hf = find_hf()
    for f in ("README.md", "per_target_summary.parquet", "best_rollouts.parquet",
              "rollout_metrics_all.parquet"):
        dest = f"{a.dest.rstrip('/')}/{f}"
        print(f"uploading {f} -> {dest}", flush=True)
        subprocess.run([hf, "buckets", "cp", f"{out}/{f}", dest], check=True)
    print("done", flush=True)
    return 0


README = """\
# contacts-v1 train-set rollouts (MarinFold exp98)

Public artifacts for [Open-Athena/MarinFold#98](https://github.com/Open-Athena/MarinFold/issues/98):
**1,000,000 rollouts** (1000 training targets x 1000 rollouts) from the tuned
contacts-v1 1.5B model (eval loss 2.7566), generated on TPU.

- `per_target_summary.parquet` (1000 rows) — per target: `L`, `n_gt`, best-of-1000
  `best_recall`/`best_f1`, per-rollout means, and timing.
- `best_rollouts.parquet` (1000 rows) — the best-recall and best-F1 rollout per
  target, saved verbatim as complete contacts-v1 documents (`*_document`), with
  metrics and parsed `*_pred_contacts` (JSON list of [i,j] seq-index pairs).
- `rollout_metrics_all.parquet` (1,000,000 rows) — every rollout's per-band
  precision/recall/F1 (`all`/`short`/`med`/`long`), `n_gen_tokens`, `finished`,
  keyed by `entry_id` + `r` (sorted by entry_id for filtered reads).

All readable anonymously, e.g.:

    import pandas as pd
    t = pd.read_parquet("hf://buckets/open-athena/MarinFold/data/"
                        "contacts-v1-train-rollouts-exp98/per_target_summary.parquet")

See the interactive explorer notebook in the experiment dir.
"""

if __name__ == "__main__":
    raise SystemExit(main())
