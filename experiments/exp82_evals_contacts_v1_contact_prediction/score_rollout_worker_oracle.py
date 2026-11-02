# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""score_rollout_worker.py + a second output: per-rollout ordered contacts.

The stock worker extracts each rollout's contacts, votes them into an [L,L]
matrix, and discards which rollout contributed what -- fine for the standard
"vote" R-precision, but it means "how good is the single best rollout out of
100" can never be recovered after the fact (issue: user wants this as a
second, plottable mean R-precision alongside the standard one).

This is the stock worker, byte-identical except for one addition: alongside
the existing sparse votes table, it also writes a per-rollout DETAIL table --
(dataset, stem, L, rollout, rank, i, j) -- one row per unique, sep-filtered
contact in EMISSION ORDER within that one rollout (rank=0 is first). That
ordering is exactly what "first R generated contacts" (the exp82 per-rollout
R-precision definition) needs; a downstream scorer with ground truth can then
compute, per protein, per rollout, per range: precision of that rollout's
first R contacts (R = true-contact count in that range), and take the max
over the 100 rollouts.

The existing votes output/schema/path is untouched -- this is additive.
Detail parts land under ``<out>/<label>_detail/`` instead of ``<out>/<label>/``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("i", pa.int16()), ("j", pa.int16()), ("votes", pa.int16()),
])
DETAIL_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rollout", pa.int16()), ("rank", pa.int16()), ("i", pa.int16()), ("j", pa.int16()),
])


def stage_model(src: str, dst: Path) -> Path:
    import fsspec
    if "://" not in src:
        return Path(src)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fs, root = fsspec.core.url_to_fs(src)
    files = [f for f in fs.ls(root, detail=True) if f["type"] == "file"]
    assert files, f"no files under {src}"
    for f in files:
        fs.get_file(f["name"], str(dst / os.path.basename(f["name"])))
    size = sum(f["size"] for f in files)
    print(f"[worker] staged model {src} -> {dst} ({len(files)} files, "
          f"{size / 2**30:.2f} GiB, {time.time() - t0:.0f}s)", flush=True)
    return dst


def read_parquet(uri: str, **kw):
    import fsspec
    with fsspec.open(uri, "rb") as fh:
        return pq.read_table(fh, **kw)


def write_parquet(tbl, uri: str) -> None:
    import fsspec
    with fsspec.open(uri, "wb") as fh:
        pq.write_table(tbl, fh, compression="zstd")


def load_targets(path: str):
    recs = read_parquet(path).to_pylist()
    recs.sort(key=lambda r: r["L"])
    return recs


def done_stems(out_dir: str, shard_i: int, num_shards: int) -> tuple[set[str], int]:
    import fsspec
    fs, _ = fsspec.core.url_to_fs(out_dir)
    pat = f"{out_dir.rstrip('/')}/shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet"
    try:
        parts = fs.glob(pat)
    except FileNotFoundError:
        return set(), 0
    seen: set[str] = set()
    for p in parts:
        try:
            t = read_parquet(fs.unstrip_protocol(p), columns=["dataset", "stem"])
            seen |= {f"{d}__{s}" for d, s in zip(t.column("dataset").to_pylist(),
                                                 t.column("stem").to_pylist())}
        except Exception as e:
            print(f"[worker] ignoring unreadable part {p}: {e}", flush=True)
    return seen, len(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True, help="s3/gs prefix; parts land under <out>/<label>/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-request-seed", dest="per_request_seed", action="store_false")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))
    out_dir = f"{a.out.rstrip('/')}/{a.label}"
    detail_dir = f"{a.out.rstrip('/')}/{a.label}_detail"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    recs = load_targets(a.targets)
    mine = [r for k, r in enumerate(recs) if k % num_shards == shard_i]
    skip, n_existing_parts = done_stems(out_dir, shard_i, num_shards)
    todo = [r for r in mine if f"{r['dataset']}__{r['stem']}" not in skip]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[worker] shard {shard_i}/{num_shards}: {len(mine)} assigned, {len(skip)} already done, "
          f"{len(todo)} to do | n_rollouts={a.n_rollouts} top_k={a.top_k} top_p={a.top_p} "
          f"T={a.temperature} per_request_seed={a.per_request_seed} label={a.label}", flush=True)
    if not todo:
        print("[worker] nothing to do")
        return 0

    model_dir = stage_model(a.model, Path("/tmp/marinfold_model"))
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    end_id = tok.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=a.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=a.max_num_seqs, seed=a.seed)

    t0, n_unfinished, n_total, part = time.time(), 0, 0, n_existing_parts
    for s in range(0, len(todo), a.chunk):
        group = todo[s:s + a.chunk]
        prompts, per, sps = [], [], []
        for r in group:
            residues = residues_from_sequence(r["input_seq"])
            first, maps = len(prompts), []
            for k in range(a.n_rollouts):
                doc = build_document(f"{r['stem']}:r{k}", residues, [], config=GenerationConfig())
                prompts.append(doc.document[: doc.document.index(BEGIN) + len(BEGIN)])
                maps.append({(doc.n_term_index + t) % NUM_POS: t for t in range(doc.seq_len)})
            plen = len(tok(prompts[first], add_special_tokens=False).input_ids)
            max_new = min(8192 - plen, a.contact_mult * r["L"] + 128)
            per.append((r, first, maps))
            sps += [SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                   max_tokens=max_new, stop_token_ids=[end_id],
                                   skip_special_tokens=False,
                                   **({"seed": a.seed * 1_000_003 + first + k}
                                      if a.per_request_seed else {}))
                    for k in range(a.n_rollouts)]
        ts = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - ts

        rows = {c: [] for c in ("dataset", "stem", "L", "i", "j", "votes")}
        drows = {c: [] for c in ("dataset", "stem", "L", "rollout", "rank", "i", "j")}
        for r, first, maps in per:
            chunk_outs = outs[first:first + a.n_rollouts]
            n_unfinished += sum(1 for o in chunk_outs if o.outputs[0].finish_reason != "stop")
            n_total += a.n_rollouts
            L = r["L"]
            M = np.zeros((L, L), np.int32)
            for k, (o, seqidx) in enumerate(zip(chunk_outs, maps)):
                seen = set()
                rank = 0
                for x, y in CONTACT_RE.findall(o.outputs[0].text):
                    ia, ib = seqidx.get(int(x)), seqidx.get(int(y))
                    if ia is None or ib is None or ia == ib:
                        continue
                    key = (min(ia, ib), max(ia, ib))
                    if abs(ia - ib) >= MIN_SEP and key not in seen:
                        seen.add(key)
                        M[key] += 1
                        drows["dataset"].append(r["dataset"])
                        drows["stem"].append(r["stem"])
                        drows["L"].append(L)
                        drows["rollout"].append(k)
                        drows["rank"].append(rank)
                        drows["i"].append(key[0])
                        drows["j"].append(key[1])
                        rank += 1
            ii, jj = np.nonzero(np.triu(M, k=1))
            rows["dataset"] += [r["dataset"]] * len(ii)
            rows["stem"] += [r["stem"]] * len(ii)
            rows["L"] += [L] * len(ii)
            rows["i"] += ii.astype(np.int16).tolist()
            rows["j"] += jj.astype(np.int16).tolist()
            rows["votes"] += M[ii, jj].astype(np.int16).tolist()

        dest = (f"{out_dir}/shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet")
        write_parquet(pa.table(rows, schema=SCHEMA), dest)
        ddest = (f"{detail_dir}/shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet")
        dtbl = pa.table({k: pa.array(v, type=DETAIL_SCHEMA.field(k).type)
                         for k, v in drows.items()}, schema=DETAIL_SCHEMA)
        write_parquet(dtbl, ddest)
        part += 1
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[worker] [{s + len(group)}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
              f"{dt:6.1f}s {ntok / dt:7.0f} tok/s unfinished={n_unfinished}/{n_total} "
              f"-> {dest} (+detail {dtbl.num_rows} rows) (elapsed {(time.time() - t0) / 60:.1f}m)",
              flush=True)

    print(f"[worker] DONE shard {shard_i}/{num_shards}: {len(todo)} proteins in "
          f"{(time.time() - t0) / 60:.1f} min | unfinished {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
