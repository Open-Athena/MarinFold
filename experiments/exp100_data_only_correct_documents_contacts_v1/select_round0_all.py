# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Select **all** round-0 proteins of the contacts-v1 train corpus as exp100
targets (the full-scale run; issue #100). Unlike select_targets.py (which samples
1000 with L/contact/truncation filters), this keeps **every** round-0 protein — no
filtering by length or contact count — and reads the ground-truth contacts by
parsing the existing document text (no pyconfind). Because those contacts came
from a document that already fit the 8192-token budget, the regenerated
only-correct document fits too, so nothing needs to be skipped.

Round-0 is one of the 5 exp53 rounds (highest pLDDT, cleanest) → ~1 realization
per unique protein (~827k). Output: ``targets_r0.parquet`` (entry_id, L, sequence,
n_gt, gt_contacts, global_plddt, struct_cluster_id), consumed by gen_prompts.py +
the worker exactly like targets.parquet.

    uv run python select_round0_all.py \
        --gcs-out gs://marin-us-east5/protein-structure/MarinFold/exp100_only_correct_contacts_v1_train/targets_r0.parquet \
        --out data/targets_r0.parquet
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

from select_targets import N_SHARDS, first_round0_shard, parse_doc, shard_path

COLS = ["document", "seq_len", "entry_id", "global_plddt", "struct_cluster_id", "round"]


def process_shard(fs, si: int) -> list[dict]:
    with fs.open(shard_path(si), "rb") as fh:
        tbl = pq.read_table(fh, columns=COLS).to_pylist()
    out = []
    for r in tbl:
        if r["round"] != 0:
            continue
        parsed = parse_doc(r["document"])
        if parsed is None:
            continue
        L, seq, gt = parsed
        out.append(dict(
            entry_id=r["entry_id"], L=L, sequence=seq, n_gt=len(gt),
            gt_contacts=[[i, j] for (i, j) in gt],
            global_plddt=float(r["global_plddt"]) if r["global_plddt"] is not None else None,
            struct_cluster_id=r["struct_cluster_id"], round=0, shard=si))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/targets_r0.parquet")
    ap.add_argument("--gcs-out", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-shards", type=int, default=None, help="first N round-0 shards (testing)")
    a = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    start0 = first_round0_shard(fs)
    shards = list(range(start0, N_SHARDS))
    if a.limit_shards:
        shards = shards[: a.limit_shards]
    print(f"round-0 shards: [{start0}, {N_SHARDS-1}] ({len(shards)} shards)", flush=True)

    rows: list[dict] = []
    seen: set[str] = set()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_shard, fs, si): si for si in shards}
        for fut in as_completed(futs):
            for r in fut.result():
                if r["entry_id"] not in seen:  # paranoia: round-0 is unique already
                    seen.add(r["entry_id"]); rows.append(r)
            done += 1
            if done % 25 == 0 or done == len(shards):
                print(f"  {done}/{len(shards)} shards, {len(rows)} proteins", flush=True)

    rows.sort(key=lambda c: c["entry_id"])
    n0 = sum(1 for r in rows if r["n_gt"] == 0)
    Ls = [r["L"] for r in rows]
    print(f"total proteins: {len(rows)}  (0-contact: {n0})  "
          f"L min/mean/max: {min(Ls)}/{sum(Ls)//len(Ls)}/{max(Ls)}", flush=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, a.out)
    print(f"wrote {a.out}", flush=True)
    if a.gcs_out:
        with fs.open(a.gcs_out, "wb") as fh:
            pq.write_table(table, fh)
        print(f"wrote {a.gcs_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
