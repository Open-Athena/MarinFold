# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 scale target selection: sample ~1M random proteins from the ESM-Atlas
(ESMFold2-distillation) contacts-v1 corpus (exp139), for rollout generation.

The corpus is 3,338 parquet shards (66.8M docs) on the HF bucket
  hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_esm_atlas/train/
Rows are one-per-linclust-cluster (already deduped), NOT pLDDT-ordered, so any
random subset of shards is an unbiased sample. Each row has a contacts-v1
`document` (parse_doc reuses unchanged) plus `global_plddt`/`ptm` (ESMFold2
confidence) for GT-quality filtering.

Emits the exact target schema the rollout worker + build_refinement_corpus.py
consume: entry_id, L, sequence, n_gt, gt_contacts, global_plddt, ptm.

    # local smoke (a downloaded shard):
    uv run --no-sync python select_targets_esm_atlas.py --input shard.parquet --n 500 --out t.parquet
    # full (best on an iris pod near the data, or from a GCS mirror):
    uv run --with 'huggingface_hub>=1.5' python select_targets_esm_atlas.py \
        --input 'hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_esm_atlas/train/shard-*-of-03338.parquet' \
        --n 1000000 --min-plddt 80 --max-len 512 --out targets_esm_atlas_1M.parquet
"""
from __future__ import annotations
import argparse, glob as _glob, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/exp98_data_generate_rollouts_contacts_v1_train"))
from select_targets import parse_doc  # (L, seq, gt_pairs) | None ; pure contacts-v1 vocab

COLS = ["document", "entry_id", "seq_len", "global_plddt", "ptm", "truncated", "num_contacts"]

def list_shards(inp: str) -> list[str]:
    if inp.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        return ["hf://" + p for p in fs.glob(inp[len("hf://"):])]
    return sorted(_glob.glob(inp))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="parquet shard glob (hf:// or local)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-plddt", type=float, default=80.0)
    ap.add_argument("--min-ptm", type=float, default=0.0)
    ap.add_argument("--min-contacts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=163)
    a = ap.parse_args()

    shards = list_shards(a.input)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(shards)
    print(f"{len(shards)} shards; target n={a.n} "
          f"(filters: plddt>={a.min_plddt} ptm>={a.min_ptm} L<={a.max_len} gt>={a.min_contacts})", flush=True)

    pool: list[dict] = []
    seen: set[str] = set()
    t0 = time.time()
    for si, shard in enumerate(shards):
        cols = [c for c in COLS]
        df = pd.read_parquet(shard, columns=cols)
        df = df[(~df.truncated.astype(bool)) & (df.seq_len <= a.max_len)
                & (df.global_plddt >= a.min_plddt) & (df.ptm >= a.min_ptm)
                & (df.num_contacts >= a.min_contacts)]
        for row in df.itertuples(index=False):
            eid = str(row.entry_id)
            if eid in seen:
                continue
            parsed = parse_doc(row.document)
            if parsed is None:
                continue
            L, seq, gt = parsed
            if len(gt) < a.min_contacts:
                continue
            seen.add(eid)
            pool.append(dict(entry_id=eid, L=int(L), sequence=seq, n_gt=len(gt),
                             gt_contacts=[[int(i), int(j)] for i, j in gt],
                             global_plddt=float(row.global_plddt), ptm=float(row.ptm)))
            if len(pool) >= a.n:
                break
        if (si + 1) % 5 == 0 or len(pool) >= a.n:
            print(f"  shard {si+1}/{len(shards)}: pool={len(pool)} ({time.time()-t0:.0f}s)", flush=True)
        if len(pool) >= a.n:
            break

    out = pd.DataFrame(pool[: a.n])
    out.to_parquet(a.out, index=False)
    print(f"\nwrote {len(out)} targets -> {a.out}", flush=True)
    print(f"  L: mean={out.L.mean():.0f} median={int(out.L.median())} max={out.L.max()}", flush=True)
    print(f"  n_gt: mean={out.n_gt.mean():.0f} median={int(out.n_gt.median())}", flush=True)
    print(f"  plddt: mean={out.global_plddt.mean():.1f}  ptm: mean={out.ptm.mean():.3f}", flush=True)
    if len(out) < a.n:
        print(f"  NOTE: only {len(out)} < requested {a.n} (read all {len(shards)} shards)", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
