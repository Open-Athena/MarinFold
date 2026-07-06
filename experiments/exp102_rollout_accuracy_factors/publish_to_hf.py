# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp102 ordered-rollout artifacts to the **public** open-athena/
MarinFold HF bucket so Allen (no GPU) can do the whole issue-#102 analysis on CPU.

Consolidates the per-target worker outputs into one parquet + the target set +
a README, and uploads them via ``hf buckets cp`` (needs an open-athena-scoped
HF_TOKEN; the bucket is anon-readable):

  data/contacts-v1-rollouts-ordered-exp102/
    rollout_metrics_ordered.parquet   ~200k rows — every rollout's ordered `pred`
                                      + per-contact `pred_logprob` + per-band metrics
    targets.parquet                   the ~200 targets (sequence + gt_contacts)
    README.md

    HF_TOKEN=<open-athena-scoped> uv run python publish_to_hf.py --run data/runs/full
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "hf://buckets/open-athena/MarinFold/data/contacts-v1-rollouts-ordered-exp102"


def find_hf() -> str:
    """An ``hf`` binary that supports ``buckets`` (skip venv-shadowed old ones)."""
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
            if subprocess.run([hf, "buckets", "--help"], capture_output=True).returncode == 0:
                return hf
        except OSError:
            continue
    raise RuntimeError("no `hf` with `buckets` support on PATH")


def _read_metrics(p: str) -> pd.DataFrame:
    entry = os.path.basename(p)[: -len(".parquet")]
    m = pq.read_table(p).to_pandas()
    if "entry_id" not in m.columns:
        m.insert(0, "entry_id", entry)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/runs/full", help="worker --out dir")
    ap.add_argument("--targets", default="data/targets.parquet")
    ap.add_argument("--dest", default=BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="build locally, skip upload")
    a = ap.parse_args()

    mdir = f"{a.run.rstrip('/')}/rollout_metrics_ordered"
    mpaths = sorted(p := os.path.join(mdir, f) for f in os.listdir(mdir) if f.endswith(".parquet"))
    print(f"{len(mpaths)} per-target metric files", flush=True)

    out = tempfile.mkdtemp(prefix="exp102pub_")
    allm = (pd.concat([_read_metrics(p) for p in mpaths], ignore_index=True)
            .sort_values(["entry_id", "r"]).reset_index(drop=True))
    pq.write_table(pa.Table.from_pandas(allm, preserve_index=False),
                   f"{out}/rollout_metrics_ordered.parquet", row_group_size=50_000)
    print(f"rollout_metrics_ordered.parquet: {len(allm)} rows "
          f"({allm.entry_id.nunique()} targets)", flush=True)

    pq.write_table(pq.read_table(a.targets), f"{out}/targets.parquet")
    with open(f"{out}/README.md", "w") as fh:
        fh.write(README)

    files = ["README.md", "rollout_metrics_ordered.parquet", "targets.parquet"]
    for f in files:
        print(f"  {f}: {os.path.getsize(f'{out}/{f}')/1e6:.1f} MB", flush=True)
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
# contacts-v1 rollouts with contact ORDER + per-contact logprob (MarinFold exp102)

Public artifacts for [Open-Athena/MarinFold#102](https://github.com/Open-Athena/MarinFold/issues/102):
what differentiates high-accuracy rollouts from average ones? Same tuned
contacts-v1 1.5B model (eval loss 2.7566) and *resample* recipe as
[exp98](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/contacts-v1-train-rollouts-exp98),
but regenerated to keep what exp98 discarded: contact **emission order** and
**per-contact logprob**. ~200 length-stratified train targets x 1000 rollouts.

- `rollout_metrics_ordered.parquet` — one row per rollout, keyed by `entry_id`+`r`
  (joins 1:1 to exp98's `rollout_metrics_all.parquet`):
  - `pred` — flattened predicted contacts `[i0,j0,i1,j1,…]` **in the order the
    model emitted them** (deduped, first occurrence; seq-index space, sep>=6).
    NOTE: unlike exp98's `pred`, this is NOT sorted — position = emission rank.
  - `pred_logprob` — length `n_pred`; `pred_logprob[k]` is contact k's emission
    logprob = sum of the 3 sampled-token logprobs of its `<contact> <pI> <pJ>`
    statement, under the model's raw (un-warped) next-token distribution.
  - `nll`, `nll_per_tok`, `n_gen_tokens`, `finished`, `n_pred`, and per-band
    (`all`/`short`/`med`/`long`) `*_npred/_tp/_prec/_rec/_f1` — as in exp98.
- `targets.parquet` — the ~200 targets: `sequence`, `L`, `gt_contacts`
  (ground-truth pairs), so every derived feature (separation, amino-acid identity
  of a contact's residues, correctness) is computable offline.

All readable anonymously, e.g.:

    import pandas as pd
    B = "hf://buckets/open-athena/MarinFold/data/contacts-v1-rollouts-ordered-exp102"
    m = pd.read_parquet(f"{B}/rollout_metrics_ordered.parquet")
    t = pd.read_parquet(f"{B}/targets.parquet")

See `analysis_starter.py` in the experiment dir for worked examples of the #102
questions (order bias, per-contact confidence, within-target variance).
"""

if __name__ == "__main__":
    raise SystemExit(main())
