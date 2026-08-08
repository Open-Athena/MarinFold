# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp163 rollout artifacts to the **public** open-athena/MarinFold HF
bucket so anyone (incl. an auth-free Colab) can read them (issue #163).

The worker (``gen_rollouts_worker_exp163.py``) writes per-shard aggregate parquets
to a GCS run dir (``rollout_metrics/shard-<i>-part-<k>.parquet``). This consolidates
every shard's part files into ONE ``rollout_metrics_all.parquet`` and uploads it,
plus ``targets.parquet`` (GT + sequences), to a NEW public bucket path:

  data/contacts-v1-rollouts-exp163/
    rollout_metrics_all.parquet   every rollout: entry_id, r, n_gen_tokens,
                                  finished, n_pred, pred + per-band
                                  {all,short,med,long}_{npred,tp,prec,rec,f1}
    targets.parquet               per target: sequence, L, gt_contacts
    README.md

Unlike exp98 there is NO best_rollouts / per_target_summary (both dropped in
exp163 — see SCALE_PLAN.md §A). Uploads need an open-athena-scoped ``HF_TOKEN``.

    HF_TOKEN=<open-athena-scoped> uv run python publish_exp163.py \
        --run gs://marin-us-east5/protein-structure/MarinFold/exp163_rollouts_contacts_v1/runs/full \
        --targets gs://.../exp163_rollouts_contacts_v1/targets.parquet
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import fsspec
import pyarrow as pa  # noqa: F401  (kept for parity / schema helpers if extended)
import pyarrow.parquet as pq

BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-rollouts-exp163"
ROW_GROUP_SIZE = 50_000


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


def _default_targets(run: str) -> str:
    """exp98/exp163 layout: targets.parquet sits two levels above ``runs/<name>``."""
    return f"{os.path.dirname(os.path.dirname(run.rstrip('/')))}/targets.parquet"


def concat_parts(fs, parts: list[str], dest: str) -> int:
    """Stream every ``shard-*-part-*.parquet`` into one parquet, appending row
    groups so we never hold the whole (tens-of-millions-of-rows) table in RAM.
    Parts share a schema (one worker wrote them all); cast defensively if a part's
    schema wobbles (e.g. an all-empty ``pred`` column inferred as list<null>)."""
    writer = None
    schema = None
    total = 0
    for p in parts:
        with fs.open(p, "rb") as fh:
            table = pq.read_table(fh)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(dest, schema)
        elif not table.schema.equals(schema):
            table = table.cast(schema)
        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        total += table.num_rows
    if writer is not None:
        writer.close()
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GCS run dir (the worker's --out, e.g. runs/full)")
    ap.add_argument("--targets", default=None,
                    help="targets.parquet with entry_id + gt_contacts + sequence "
                         "(published so the refinement corpus is reproducible; "
                         "defaults to two levels above <run>)")
    ap.add_argument("--dest", default=BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="build locally, skip upload")
    a = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(a.run)
    run = a.run.rstrip("/")
    parts = sorted(p for p in fs.glob(f"{run}/rollout_metrics/shard-*-part-*.parquet")
                   if p.endswith(".parquet"))
    if not parts:
        raise SystemExit(f"no rollout_metrics/shard-*-part-*.parquet under {run}")
    print(f"{len(parts)} part files across all shards", flush=True)

    out = tempfile.mkdtemp(prefix="exp163pub_")

    # 1. rollout_metrics_all.parquet (every rollout, one file, streamed).
    n_rows = concat_parts(fs, parts, f"{out}/rollout_metrics_all.parquet")
    print(f"rollout_metrics_all.parquet: {n_rows} rows", flush=True)

    # 2. targets.parquet (GT + sequences — lets viewers rebuild the corpus).
    targets = a.targets or _default_targets(run)
    with fsspec.open(targets, "rb") as fh:
        pq.write_table(pq.read_table(fh), f"{out}/targets.parquet")
    print(f"targets.parquet copied from {targets}", flush=True)

    # 3. README.
    with open(f"{out}/README.md", "w") as fh:
        fh.write(README)

    files = ("README.md", "rollout_metrics_all.parquet", "targets.parquet")
    for f in files:
        sz = os.path.getsize(f"{out}/{f}") / 1e6
        print(f"  {f}: {sz:.1f} MB", flush=True)

    if a.dry_run:
        print(f"dry-run: built in {out}")
        return 0

    hf = find_hf()
    for f in files:
        dest = f"{a.dest.rstrip('/')}/{f}"
        print(f"uploading {f} -> {dest}", flush=True)
        subprocess.run([hf, "buckets", "cp", f"{out}/{f}", dest], check=True)
    print("done", flush=True)
    return 0


README = """\
# contacts-v1 rollouts for refinement (MarinFold exp163)

Public artifacts for [Open-Athena/MarinFold#163](https://github.com/Open-Athena/MarinFold/issues/163):
base contacts-v1 1.5B (E8, eval loss 2.7566) rollouts over many training targets,
generated on TPU as the **candidate pool** for the rollout-refinement corpus.

Sampling: `T=1.0`, `top_p=0.95`, **top-k disabled** (`-1`; the #142
under-generation fix), ~24 rollouts/target, no logprobs.

- `rollout_metrics_all.parquet` — one row per rollout, keyed by `entry_id` + `r`:
  `n_gen_tokens`, `finished`, `n_pred`, `pred` (the flattened predicted contact
  pairs `[i0,j0,i1,j1,…]` in sequence-index space, for building the candidate
  blocks), and per separation band (`all`/`short`/`med`/`long`) the
  `{npred,tp,prec,rec,f1}`. Written with 50k-row row groups for filtered reads.
- `targets.parquet` — per target: `sequence`, `L`, and `gt_contacts` (ground-truth
  pairs), so the refinement corpus (`build_refinement_corpus.py`) is reproducible.

All readable anonymously, e.g.:

    import pandas as pd
    r = pd.read_parquet("hf://buckets/open-athena/MarinFold/data/"
                        "contacts-v1-rollouts-exp163/rollout_metrics_all.parquet",
                        columns=["entry_id", "r", "pred"])
"""

if __name__ == "__main__":
    raise SystemExit(main())
