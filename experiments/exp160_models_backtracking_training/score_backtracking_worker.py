# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One shard of the retraction-aware rollout eval (#160), on one TPU/GPU pod.

Derived from exp82's ``score_rollout_worker.py`` — same settled recipe (n
resampled contacts-v1 rollouts per protein, voted into an ``[L, L]`` per-pair
matrix; ``top_k=-1``, ``top_p=0.95``, ``T=1.0``) — with the two changes a model
that can emit ``<retract>`` forces:

1. **Votes are folded, not counted.** exp82 scans the completion with a bare
   ``<contact>`` regex, which for this model would count a pair the rollout
   later *took back*. Here each rollout is folded through
   ``read.fold_statements`` — the semantic definition of retraction from #158 —
   and only the set live at ``<end>`` votes. For a model that never retracts
   the two are identical, which is what keeps the exp120 control comparable.
2. **The ordered edit list is kept.** The retraction diagnostics (the real
   pass/fail of #160) need *which* pairs were retracted and *when*, not just the
   final set, so each rollout also emits its statement stream. That is written
   as a second parquet family rather than recomputed later: the rollouts are
   sampled, so there is no way to reconstruct them from the votes.

Both outputs are mapped from **position** space (what the model emits, per-
rollout randomized) into **sequence-index** space (where ground truth lives)
using that rollout's own position map. Folding happens *before* the mapping,
because a retract has to match its contact in the space they were written in.

Everything else — sharding by ``idx % n``, the parquet-part resume, fsspec for
all I/O so ``gs://`` and ``s3://`` both work — is exp82's, unchanged.

Run (the dispatcher's bootstrap does this)::

    python score_backtracking_worker.py --model gs://…/model --targets gs://…/targets.parquet \
        --out gs://…/scores --label exp160-bt50 --shard 3/4 --n-rollouts 100
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# The #158 fold. Not a regex in this file: "what a rollout actually claims" is a
# semantic decision that belongs in one place, next to its unit tests. Imported
# at module scope so the readout below can be tested without vLLM present.
from marinfold.document_structures.contacts_v1.read import (
    fold_statements,
    iter_structure_statements,
)

BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6

# Folded live-contact votes: exp82's schema, byte-for-byte, so the downstream
# matrix rebuild and exp89 metric pass are shared with every prior rollout eval.
VOTES_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("i", pa.int16()), ("j", pa.int16()), ("votes", pa.int16()),
])

# One row per rollout: the ordered edit list plus the fold's anomaly counters.
# `kind` is 0=contact, 1=retract. List columns keep this compact — 554 proteins
# x 100 rollouts is ~55k rows, not the ~9M a row-per-statement layout would give.
STREAMS_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rollout", pa.int16()), ("finished", pa.bool_()), ("n_tokens", pa.int32()),
    ("n_unmapped", pa.int16()), ("n_retract_absent", pa.int16()),
    ("n_reemit", pa.int16()), ("n_redundant_contact", pa.int16()),
    ("kind", pa.list_(pa.int8())), ("i", pa.list_(pa.int16())), ("j", pa.list_(pa.int16())),
])


class RolloutReadout(NamedTuple):
    """One rollout, read out of position space into sequence-index space."""

    live: list[tuple[int, int]]              # folded, |i-j| >= MIN_SEP, i < j — the votes
    statements: list[tuple[int, int, int]]   # ordered (kind, i, j); kind 0=contact 1=retract
    n_unmapped: int                          # statements naming a position the protein lacks
    fold: object                             # read.FoldResult, for its anomaly counters


def read_rollout(text: str, seqidx: dict[int, int]) -> RolloutReadout:
    """Fold a completion, then map it into sequence-index space.

    Order matters: the fold runs in **position** space, because that is the
    space the model wrote in and where a ``<retract>`` has to match the
    ``<contact>`` it takes back. Mapping first would be equivalent only when the
    map is total, and it is not — a model may name a position the protein does
    not have.

    The votes are additionally restricted to the scored universe (``i != j``,
    ``|i-j| >= MIN_SEP``); the statement stream is not, because which pairs are
    in the universe is the analysis's decision and dropping statements here
    would silently compress the retraction-distance measurement.
    """
    statements = list(iter_structure_statements(text))
    fold = fold_statements(statements)

    live = []
    for pos_a, pos_b in fold.live:
        ia, ib = seqidx.get(pos_a), seqidx.get(pos_b)
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        live.append((min(ia, ib), max(ia, ib)))

    mapped, n_unmapped = [], 0
    for kind, pos_a, pos_b in statements:
        ia, ib = seqidx.get(pos_a), seqidx.get(pos_b)
        if ia is None or ib is None:
            n_unmapped += 1
            continue
        mapped.append((0 if kind == "contact" else 1, ia, ib))
    return RolloutReadout(live=live, statements=mapped, n_unmapped=n_unmapped, fold=fold)


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
    recs.sort(key=lambda r: r["L"])          # short -> long, then interleave
    return recs


def done_stems(votes_dir: str, shard_i: int, num_shards: int) -> tuple[set[str], int]:
    """Stems this shard already covered, and how many part files hold them.

    The part count is what writing must resume from: restarting the counter at 0
    makes the new part-0000 clobber the old one whose stems are being skipped as
    "done", so those proteins vanish from the output entirely (exp169 lost two
    that way). Only the votes dir is consulted — the two families are written
    together, so it speaks for both.
    """
    import fsspec
    fs, _ = fsspec.core.url_to_fs(votes_dir)
    pat = f"{votes_dir.rstrip('/')}/shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet"
    try:
        parts = fs.glob(pat)
    except FileNotFoundError:
        return set(), 0
    seen: set[str] = set()
    for p in parts:
        try:
            # glob returns bare keys; the protocol has to go back on before
            # fsspec can reopen them. This is what keeps the worker backend-
            # agnostic across gs:// and s3://.
            t = read_parquet(fs.unstrip_protocol(p), columns=["dataset", "stem"])
            seen |= {f"{d}__{s}" for d, s in zip(t.column("dataset").to_pylist(),
                                                 t.column("stem").to_pylist())}
        except Exception as e:                       # a half-written part from a kill
            print(f"[worker] ignoring unreadable part {p}: {e}", flush=True)
    return seen, len(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True, help="prefix; parts land under <out>/<label>/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    # 8, not exp82's 6: retraction makes documents longer (the #159 corpus runs
    # ~38% more statements than contacts alone), and a budget that truncates the
    # backtracking arm but not the control would confound the comparison. Both
    # arms run the same number so neither is advantaged.
    ap.add_argument("--contact-mult", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-request-seed", dest="per_request_seed", action="store_false",
                    help="Required on TPU: the JAX backend rejects SamplingParams.seed "
                         "outright. The engine-level --seed still applies and the rollouts "
                         "are independent draws either way, so only bitwise replay is lost.")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))
    votes_dir = f"{a.out.rstrip('/')}/{a.label}/votes"
    streams_dir = f"{a.out.rstrip('/')}/{a.label}/streams"

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    recs = load_targets(a.targets)
    mine = [r for k, r in enumerate(recs) if k % num_shards == shard_i]
    skip, n_existing_parts = done_stems(votes_dir, shard_i, num_shards)
    todo = [r for r in mine if f"{r['dataset']}__{r['stem']}" not in skip]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[worker] shard {shard_i}/{num_shards}: {len(mine)} assigned, {len(skip)} already done, "
          f"{len(todo)} to do | n_rollouts={a.n_rollouts} top_k={a.top_k} top_p={a.top_p} "
          f"T={a.temperature} contact_mult={a.contact_mult} label={a.label}", flush=True)
    if not todo:
        print("[worker] nothing to do")
        return 0

    model_dir = stage_model(a.model, Path("/tmp/marinfold_model"))
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    end_id = tok.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    # An absent token resolves to the UNK id, not to None, so the control model's
    # pre-retract vocab has to be detected by comparing against unk_token_id.
    # Logged rather than asserted: the exp120 control is *supposed* to lack it.
    retract_id = tok.convert_tokens_to_ids("<retract>")
    has_retract = retract_id is not None and retract_id >= 0 and retract_id != tok.unk_token_id
    print(f"[worker] tokenizer vocab={len(tok)} <end>={end_id} "
          f"<retract>={retract_id if has_retract else 'absent'}", flush=True)

    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=a.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=a.max_num_seqs, seed=a.seed)

    t0, n_unfinished, n_total, part = time.time(), 0, 0, n_existing_parts
    n_retract_stmts = 0
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

        vrows = {c: [] for c in ("dataset", "stem", "L", "i", "j", "votes")}
        srows = {c.name: [] for c in STREAMS_SCHEMA}
        for r, first, maps in per:
            chunk_outs = outs[first:first + a.n_rollouts]
            n_total += a.n_rollouts
            L = r["L"]
            M = np.zeros((L, L), np.int32)
            for k, (o, seqidx) in enumerate(zip(chunk_outs, maps)):
                out0 = o.outputs[0]
                finished = out0.finish_reason == "stop"
                n_unfinished += 0 if finished else 1

                rr = read_rollout(out0.text, seqidx)
                n_retract_stmts += rr.fold.n_retract
                for ia, ib in rr.live:
                    M[ia, ib] += 1

                srows["dataset"].append(r["dataset"])
                srows["stem"].append(r["stem"])
                srows["L"].append(L)
                srows["rollout"].append(k)
                srows["finished"].append(finished)
                srows["n_tokens"].append(len(out0.token_ids))
                srows["n_unmapped"].append(min(rr.n_unmapped, 2**15 - 1))
                srows["n_retract_absent"].append(min(rr.fold.n_retract_absent, 2**15 - 1))
                srows["n_reemit"].append(min(rr.fold.n_reemit, 2**15 - 1))
                srows["n_redundant_contact"].append(min(rr.fold.n_redundant_contact, 2**15 - 1))
                srows["kind"].append([s[0] for s in rr.statements])
                srows["i"].append([s[1] for s in rr.statements])
                srows["j"].append([s[2] for s in rr.statements])

            ii, jj = np.nonzero(np.triu(M, k=1))
            vrows["dataset"] += [r["dataset"]] * len(ii)
            vrows["stem"] += [r["stem"]] * len(ii)
            vrows["L"] += [L] * len(ii)
            vrows["i"] += ii.astype(np.int16).tolist()
            vrows["j"] += jj.astype(np.int16).tolist()
            vrows["votes"] += M[ii, jj].astype(np.int16).tolist()

        stem_part = f"shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet"
        # Streams first: the votes file is what `done_stems` treats as the
        # receipt for a chunk, so it must not exist without its stream twin.
        write_parquet(pa.table(srows, schema=STREAMS_SCHEMA), f"{streams_dir}/{stem_part}")
        write_parquet(pa.table(vrows, schema=VOTES_SCHEMA), f"{votes_dir}/{stem_part}")
        part += 1
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[worker] [{s + len(group)}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
              f"{dt:6.1f}s {ntok / dt:7.0f} tok/s unfinished={n_unfinished}/{n_total} "
              f"retracts={n_retract_stmts} -> {stem_part} "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    print(f"[worker] DONE shard {shard_i}/{num_shards}: {len(todo)} proteins in "
          f"{(time.time() - t0) / 60:.1f} min | unfinished {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%) | {n_retract_stmts} retract statements",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
