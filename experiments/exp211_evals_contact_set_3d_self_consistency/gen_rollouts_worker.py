# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One shard of the issue #211 rollout generation, on one CoreWeave H100.

A fork of exp82's ``score_rollout_worker.py`` with **one** substantive change:
it keeps the rollouts apart.

exp82's worker generates n resampled contacts-v1 rollouts per protein and folds
them immediately into an ``[L, L]`` per-pair vote matrix (``M[key] += 1``), then
writes sparse ``(dataset, stem, L, i, j, votes)`` triplets. That is the right
output for scoring the *marginals*, which is what every contacts-v1 eval does —
and it is exactly the information #211 needs and cannot recover. Which contacts
were emitted *together, in one pass* is destroyed at the moment of the increment.

So this worker writes one row per emitted contact, carrying the rollout index and
the emission order:

    contacts:  (dataset, stem, L, rollout, order, i, j, duplicate)
    rollouts:  (dataset, stem, L, rollout, n_contacts, n_emitted, n_tokens,
                finished, n_out_of_range, n_too_close)

The vote matrix exp82 writes is a strict function of the first table
(``groupby(i, j).size()``), so nothing is lost — ``verify_against_exp82.py``
rebuilds it and checks the R-precision matches, which is what makes the
consistency-vs-accuracy correlation comparable to the published numbers.

**Everything else is held fixed to the settled recipe**, because the point is to
analyse the model we actually report on, not a variant of it: the 554-protein
eval set (#89), one fresh document realization per rollout, ``T=1.0``,
``top_p=0.95``, ``top_k=-1`` (#142 removed the truncating ``top_k=50``),
``max_new = 6L + 128``, ``min_seq_separation = 6``, and the same
``<contact> <pX> <pY>`` regex readout.

**The per-rollout position map is load-bearing.** contacts-v1 picks a random
wrap-around N-terminal index per document, so ``<p742>`` means a different
residue in rollout 3 than in rollout 4. Each rollout's own map is applied before
anything is recorded; without it the rollouts would be silently misaligned to
each other and the whole within-vs-across comparison would be noise.

Run (the dispatcher's bootstrap does this)::

    python gen_rollouts_worker.py --model s3://…/model --targets s3://…/targets.parquet \\
        --out s3://…/rollouts --label exp199 --shard 3/16 --n-rollouts 100
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

CONTACT_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rollout", pa.int16()), ("order", pa.int16()),
    ("i", pa.int16()), ("j", pa.int16()),
    # True when this rollout already emitted the same pair earlier. exp82 drops
    # these; they are kept because a model restating its own contacts is a
    # coherence signal, and dropping them silently would hide it.
    ("duplicate", pa.bool_()),
])

ROLLOUT_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rollout", pa.int16()),
    ("n_contacts", pa.int32()),      # distinct, in-range, sep-respecting
    ("n_emitted", pa.int32()),       # every <contact> statement matched
    ("n_tokens", pa.int32()),
    ("finished", pa.bool_()),        # hit <end> rather than the token budget
    ("n_out_of_range", pa.int32()),  # <pX> outside this realization's map
    ("n_too_close", pa.int32()),     # |i - j| < MIN_SEP, which the format forbids
])


# All S3 access goes through fsspec, never pyarrow's own S3FileSystem and never
# the `aws` CLI: iris injects CoreWeave's endpoint + credentials as an FSSPEC_S3
# blob (plus the virtual-addressing config_kwargs the bootstrap exports), which
# only fsspec/s3fs reads. See exp82's worker and exp163's dispatcher.


def stage_model(src: str, dst: Path) -> Path:
    """Copy a remote model directory to local disk (vLLM needs a real directory)."""
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
    recs.sort(key=lambda r: r["L"])  # short -> long, then interleave across shards
    return recs


def done_stems(out_dir: str, shard_i: int, num_shards: int) -> tuple[set[str], int]:
    """Stems this shard already covered, and how many part files hold them.

    The part count is what the caller must resume *writing* from. Restarting the
    counter at 0 on a resume silently overwrites the shard's own earlier parts —
    their stems are skipped as "done" and then their file is clobbered, so they
    vanish. exp169 lost two proteins to exactly this before exp82's worker
    started returning the count.
    """
    import fsspec

    fs, _ = fsspec.core.url_to_fs(out_dir)
    pat = (f"{out_dir.rstrip('/')}/contacts/"
           f"shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet")
    try:
        parts = fs.glob(pat)
    except FileNotFoundError:
        return set(), 0
    seen: set[str] = set()
    for p in parts:
        try:
            # glob returns bare keys; the protocol has to go back on before
            # fsspec can reopen them. unstrip_protocol keeps this worker
            # backend-agnostic (gs:// on TPU, s3:// on CoreWeave).
            t = read_parquet(fs.unstrip_protocol(p), columns=["dataset", "stem"])
            seen |= {
                f"{d}__{s}"
                for d, s in zip(
                    t.column("dataset").to_pylist(), t.column("stem").to_pylist()
                )
            }
        except Exception as e:  # a half-written part from a kill
            print(f"[worker] ignoring unreadable part {p}: {e}", flush=True)
    return seen, len(parts)


def parse_rollout(text: str, seqidx: dict[int, int]):
    """Read one completion's structure section into ordered contacts.

    ``seqidx`` maps this realization's wrap-around ``<pX>`` index to a 0-based
    sequence position. Returns the ordered in-range contacts plus the reject
    counts, matching exp82's readout except that order and duplicates survive.
    """
    contacts, seen = [], set()
    n_emitted = n_out_of_range = n_too_close = 0
    for x, y in CONTACT_RE.findall(text):
        n_emitted += 1
        ia, ib = seqidx.get(int(x)), seqidx.get(int(y))
        if ia is None or ib is None or ia == ib:
            n_out_of_range += 1
            continue
        if abs(ia - ib) < MIN_SEP:
            n_too_close += 1
            continue
        key = (min(ia, ib), max(ia, ib))
        contacts.append((key[0], key[1], key in seen))
        seen.add(key)
    return contacts, n_emitted, n_out_of_range, n_too_close


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True, help="s3 prefix; parts land under <out>/<label>/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-request-seed", dest="per_request_seed", action="store_false",
                    help="Required on TPU: the JAX backend rejects "
                         "SamplingParams.seed outright. The engine-level --seed "
                         "still applies and the rollouts are independent draws "
                         "either way; only bitwise replay is lost.")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))
    out_dir = f"{a.out.rstrip('/')}/{a.label}"

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
    print(f"[worker] shard {shard_i}/{num_shards}: {len(mine)} assigned, "
          f"{len(skip)} already done, {len(todo)} to do | "
          f"n_rollouts={a.n_rollouts} top_k={a.top_k} top_p={a.top_p} "
          f"T={a.temperature}", flush=True)

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
                # One fresh document realization per rollout — a different
                # wrap-around start index and a different sequence-statement
                # shuffle — which is the test-time augmentation exp82 settled on.
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

        crows = {c: [] for c in CONTACT_SCHEMA.names}
        rrows = {c: [] for c in ROLLOUT_SCHEMA.names}
        for r, first, maps in per:
            chunk_outs = outs[first:first + a.n_rollouts]
            n_unfinished += sum(1 for o in chunk_outs if o.outputs[0].finish_reason != "stop")
            n_total += a.n_rollouts
            for k, (o, seqidx) in enumerate(zip(chunk_outs, maps)):
                contacts, n_em, n_oor, n_close = parse_rollout(o.outputs[0].text, seqidx)
                for order, (i, j, dup) in enumerate(contacts):
                    crows["dataset"].append(r["dataset"])
                    crows["stem"].append(r["stem"])
                    crows["L"].append(r["L"])
                    crows["rollout"].append(k)
                    crows["order"].append(order)
                    crows["i"].append(i)
                    crows["j"].append(j)
                    crows["duplicate"].append(dup)
                rrows["dataset"].append(r["dataset"])
                rrows["stem"].append(r["stem"])
                rrows["L"].append(r["L"])
                rrows["rollout"].append(k)
                rrows["n_contacts"].append(sum(1 for _, _, d in contacts if not d))
                rrows["n_emitted"].append(n_em)
                rrows["n_tokens"].append(len(o.outputs[0].token_ids))
                rrows["finished"].append(o.outputs[0].finish_reason == "stop")
                rrows["n_out_of_range"].append(n_oor)
                rrows["n_too_close"].append(n_close)

        stem = f"shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet"
        # Contacts first: done_stems() reads the contacts directory, so a kill
        # between the two writes must not mark the chunk complete.
        write_parquet(pa.table(rrows, schema=ROLLOUT_SCHEMA), f"{out_dir}/rollouts/{stem}")
        write_parquet(pa.table(crows, schema=CONTACT_SCHEMA), f"{out_dir}/contacts/{stem}")
        part += 1
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[worker] [{s + len(group)}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
              f"{dt:6.1f}s {ntok / dt:7.0f} tok/s unfinished={n_unfinished}/{n_total} "
              f"-> {stem} (elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    print(f"[worker] DONE shard {shard_i}/{num_shards}: {len(todo)} proteins in "
          f"{(time.time() - t0) / 60:.1f} min | unfinished {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
