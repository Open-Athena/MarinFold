# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp100 only-correct regenerated documents to the **public**
open-athena/MarinFold HF bucket so anyone (incl. an auth-free Colab) can read
them — this is the experiment's success criterion (issue #100).

The per-target run lives on GCS (`runs/full/`, auth-required). This consolidates
it into a few parquets and uploads them to the public bucket:

  data/contacts-v1-train-only-correct-exp100/
    regenerated_documents.parquet  one row/protein — the selected (lowest
                                   structure-NLL) only-correct document, verbatim
    per_target_nll.parquet         one row/protein — NLL spread over the N
                                   rollouts, n_correct, timing
    rollout_nll_all.parquet        N rows/protein — every rollout's NLLs
    README.md

    HF_TOKEN=<open-athena-scoped> uv run python publish_to_hf.py \
        --run gs://marin-us-east5/protein-structure/MarinFold/exp100_only_correct_contacts_v1_train/runs/full
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

BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-train-only-correct-exp100"


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


def _read_document(fs, p):
    """One row: the selected only-correct document for a protein.

    Returns None for proteins skipped at generation time (prefix filled the whole
    context, no room for the structure section) — they are not in the deliverable.
    """
    with fs.open(p, "r") as fh:
        d = json.load(fh)
    if d.get("skipped"):
        return None
    s = d["selected"]
    return dict(entry_id=d["entry_id"], L=d["L"], n_gt=d["n_gt"],
                n_rollouts=d["n_rollouts"], n_correct=d["n_correct"],
                selected_r=s["r"], struct_nll=s["struct_nll"],
                n_contacts=s["n_contacts"], finished=s["finished"],
                precision=s["all_prec"], recall=s["all_rec"],
                document=s["document"],
                pred_contacts=json.dumps(s["pred_contacts"]))


def _read_nll(fs, p):
    """All N rollout-NLL rows for a protein (keyed by entry_id + r)."""
    entry = os.path.basename(p)[: -len(".parquet")]
    with fs.open(p, "rb") as fh:
        m = pq.read_table(fh).to_pandas()
    m = m[m["r"] >= 0]  # drop the r=-1 skip sentinel (prompt_exceeds_context)
    m.insert(0, "entry_id", entry)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GCS run dir (runs/full)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--dest", default=BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="build locally, skip upload")
    a = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(a.run)
    run = a.run.rstrip("/")
    dpaths = [p for p in fs.ls(f"{run}/documents", detail=False) if p.endswith(".json")]
    npaths = [p for p in fs.ls(f"{run}/nll", detail=False) if p.endswith(".parquet")]
    print(f"{len(dpaths)} documents, {len(npaths)} nll", flush=True)

    out = tempfile.mkdtemp(prefix="exp100pub_")

    # 1. regenerated_documents.parquet (the deliverable training set)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(lambda p: _read_document(fs, p), dpaths) if r is not None]
    n_skipped = len(dpaths) - len(rows)
    docs = pd.DataFrame(rows).sort_values("entry_id").reset_index(drop=True)
    print(f"skipped (prompt_exceeds_context): {n_skipped}", flush=True)
    pq.write_table(pa.Table.from_pandas(docs, preserve_index=False),
                   f"{out}/regenerated_documents.parquet")
    print(f"regenerated_documents.parquet: {len(docs)} rows", flush=True)

    # 2. rollout_nll_all.parquet
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        parts = list(ex.map(lambda p: _read_nll(fs, p), npaths))
    alln = pd.concat(parts, ignore_index=True).sort_values(["entry_id", "r"]).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(alln, preserve_index=False),
                   f"{out}/rollout_nll_all.parquet", row_group_size=50_000)
    print(f"rollout_nll_all.parquet: {len(alln)} rows", flush=True)

    # 3. per_target_nll.parquet (NLL spread across the N rollouts per protein)
    alln = alln.assign(ok=(alln["all_prec"] == 1.0) & (alln["all_rec"] == 1.0))
    per = (alln.groupby("entry_id")
           .agg(n_rollouts=("r", "size"),
                n_correct=("ok", "sum"),
                struct_nll_min=("struct_nll", "min"),
                struct_nll_mean=("struct_nll", "mean"),
                struct_nll_max=("struct_nll", "max"),
                struct_nll_per_tok_min=("struct_nll_per_tok", "min"),
                n_contacts=("n_contacts", "max"))
           .reset_index())
    per["n_correct"] = per["n_correct"].astype(int)
    pq.write_table(pa.Table.from_pandas(per, preserve_index=False),
                   f"{out}/per_target_nll.parquet")
    print(f"per_target_nll.parquet: {len(per)} rows", flush=True)

    with open(f"{out}/README.md", "w") as fh:
        fh.write(README)

    files = ("regenerated_documents.parquet", "per_target_nll.parquet",
             "rollout_nll_all.parquet", "README.md")
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
# contacts-v1 train-set only-correct documents (MarinFold exp100)

Public artifacts for [Open-Athena/MarinFold#100](https://github.com/Open-Athena/MarinFold/issues/100):
a **regenerated** contacts-v1 training set. For each protein we sampled N
*constrained* rollouts from the tuned contacts-v1 1.5B model (eval loss 2.7566)
in which the model may only ever emit a **true, not-yet-emitted** contact, and
`<end>` is masked until all true contacts are out — so every document is
100%-correct and full-recall by construction; the only freedom is the order /
orientation. We then re-scored each rollout under the **unmodified** model and
kept the most likely one (lowest structure-section NLL).

- `regenerated_documents.parquet` — one row per protein: the selected only-correct
  contacts-v1 `document` (verbatim, token order preserved), its `struct_nll` /
  `doc_nll`, and parsed `pred_contacts` (JSON list of [i,j] seq-index pairs).
  This is the regenerated training set.
- `per_target_nll.parquet` — one row per protein: NLL spread (min/mean/max) over
  the N rollouts, `n_correct`, `n_contacts`.
- `rollout_nll_all.parquet` — every rollout's NLLs + correctness, keyed by
  `entry_id` + `r`.

All readable anonymously, e.g.:

    import pandas as pd
    d = pd.read_parquet("hf://buckets/open-athena/MarinFold/data/"
                        "contacts-v1-train-only-correct-exp100/regenerated_documents.parquet")
"""

if __name__ == "__main__":
    raise SystemExit(main())
