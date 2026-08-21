# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage 0b — draw exp230's protein pool, decontaminated, from three corpora.

Emits the target schema the rollout worker and ``build_refinement_corpus.py``
both consume — ``entry_id, L, sequence, n_gt, gt_contacts, arm, ...`` — for a
pool that is:

* **one pool for both halves of the corpus.**  The multi-draft documents and
  the plain rehearsal documents are drawn from the same proteins, so the
  token-0 mode marker is the only systematic difference between the halves.
  If the halves came from different protein distributions the model could infer
  its mode from protein statistics instead of from the marker, and making that
  marker a clean switch is what this experiment is for.
* **decontaminated at Tier A / 30 %** against #226's 776 eval queries — the
  drop list built by ``decontam.py``.
* **mixed across predicted and experimental structure.**  AFDB and ESM-Atlas
  are what exp199 was pretrained on, so they are what "rehearsal" has to mean;
  #222's PDB monomers are the new signal and are taken in full.

``entry_id`` is unique within an arm but **not across arms**, so the pool is
keyed on ``(arm, entry_id)`` and a ``target_id`` column carries the flattened
form the rollout worker uses as its primary key.

    uv run python select_targets.py --work /data/exp230_multi \\
        --n-afdb 30000 --n-esm 30000 --n-pdb 0   # 0 = take all survivors
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from corpus_sources import AFDB, ARMS, ESM_ATLAS, PDB_MONOMERS, draw_shards

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SCHEMA = pa.schema(
    [
        ("target_id", pa.string()),
        ("arm", pa.string()),
        ("entry_id", pa.string()),
        ("L", pa.int32()),
        ("sequence", pa.string()),
        ("n_gt", pa.int32()),
        ("gt_contacts", pa.list_(pa.list_(pa.int32(), 2))),
        ("global_plddt", pa.float32()),
        ("resolution", pa.float32()),
        ("shard", pa.int32()),
        ("row", pa.int32()),
    ]
)


def load_droplist(work: Path) -> set[tuple[str, str]]:
    path = work / "droplist_exp230.parquet"
    if not path.exists():
        raise SystemExit(f"no drop list at {path} — run decontam.py first")
    table = pq.read_table(path, columns=["arm", "entry_id"])
    drop = set(zip(table.column("arm").to_pylist(), table.column("entry_id").to_pylist()))
    print(f"[droplist] {len(drop):,} contaminated (arm, entry_id) pairs", flush=True)
    return drop


def collect(spec, *, work: Path, want: int, drop: set, seed: int, log) -> tuple[list[dict], dict]:
    """Stream the arm's staged shards until ``want`` survivors are collected."""
    from corpus_sources import iter_corpus_rows, local_path

    # Only shards actually staged can be read; the draw is over the eligible
    # range, and staging is what bounds it.
    staged = [p for p in draw_shards(spec, work, 10**9, seed) if local_path(work, p).exists()]
    if not staged:
        raise SystemExit(f"no staged shards for arm {spec.arm!r} — run stage.py --arm {spec.arm}")
    rng = random.Random(seed)
    rng.shuffle(staged)

    kept: list[dict] = []
    stats = {"seen": 0, "dropped_contaminated": 0, "shards_read": 0}
    seen_ids: set[str] = set()
    for path in staged:
        if want and len(kept) >= want:
            break
        stats["shards_read"] += 1
        for rec in iter_corpus_rows(spec, work=work, log=lambda *a: None, shards=[path]):
            stats["seen"] += 1
            key = (rec["arm"], rec["entry_id"])
            if key in drop:
                stats["dropped_contaminated"] += 1
                continue
            if rec["entry_id"] in seen_ids:
                continue
            seen_ids.add(rec["entry_id"])
            rec["target_id"] = f"{rec['arm']}:{rec['entry_id']}"
            kept.append(rec)
    if want and len(kept) > want:
        # Trim by a seeded shuffle rather than by shard order, so the pool is
        # not biased toward whichever shards happened to be read first.
        rng.shuffle(kept)
        kept = kept[:want]
    stats["kept"] = len(kept)
    log(f"[{spec.arm}] {stats['shards_read']} shards -> {stats['seen']:,} usable, "
        f"{stats['dropped_contaminated']:,} contaminated, {len(kept):,} kept")
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/data/exp230_multi"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-afdb", type=int, default=30_000)
    ap.add_argument("--n-esm", type=int, default=30_000)
    ap.add_argument("--n-pdb", type=int, default=0, help="0 = every survivor")
    ap.add_argument("--seed", type=int, default=230)
    a = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    out = a.out or (a.work / "targets.parquet")
    drop = load_droplist(a.work)

    def log(*msg):
        print(" ".join(str(m) for m in msg), flush=True)

    t0 = time.time()
    rows: list[dict] = []
    stats: dict[str, dict] = {}
    for spec, want in ((AFDB, a.n_afdb), (ESM_ATLAS, a.n_esm), (PDB_MONOMERS, a.n_pdb)):
        got, st = collect(spec, work=a.work, want=want, drop=drop, seed=a.seed, log=log)
        rows.extend(got)
        stats[spec.arm] = st

    df = pd.DataFrame(rows)
    df = df[[f.name for f in SCHEMA]]
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False), out)
    log(f"[pool] {len(df):,} proteins -> {out} in {time.time() - t0:.0f}s")

    summary = (df.groupby("arm")
                 .agg(n=("target_id", "size"), L_median=("L", "median"),
                      L_p90=("L", lambda s: s.quantile(0.90)),
                      n_gt_median=("n_gt", "median"))
                 .reset_index())
    for arm, st in stats.items():
        summary.loc[summary.arm == arm, "contaminated_dropped"] = st["dropped_contaminated"]
        summary.loc[summary.arm == arm, "usable_seen"] = st["seen"]
    summary.to_csv(DATA / "pool_summary.csv", index=False)
    log("[summary]\n" + summary.to_string(index=False))

    (DATA / "pool.provenance.json").write_text(json.dumps({
        "seed": a.seed,
        "n_total": int(len(df)),
        "by_arm": {k: int(v) for k, v in df.groupby("arm").size().items()},
        "stats": stats,
        "arms": {name: {"prefix": s.prefix, "min_plddt": s.min_plddt,
                        "max_resolution": s.max_resolution,
                        "first_shard": s.first_shard,
                        "first_shard_note": s.first_shard_note}
                 for name, s in ARMS.items()},
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
