# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""One shard of exp230's on-policy rollout generation, on one CoreWeave H100.

Generates ``--n-rollouts`` contacts-v1 rollouts per protein from **exp199**, the
model this experiment fine-tunes.  Those rollouts become the *drafts* of the
multi-draft documents, which is why they have to come from exp199 rather than
be inherited from #98/#163: a draft is supposed to be something the policy
would actually write.

Structure is exp82's ``score_rollout_worker.py`` (the proven CoreWeave fan-out)
with exp163's payload (per-rollout predictions, not a voted matrix).  The one
design change relative to #163 is deliberate and fixes a blocker #163 named but
left open:

    #163's ``gen_prompts_exp163.py`` materialised **one S3 object per target**.
    Its own docstring warns that a 1M-protein run would create ~1M objects and
    that fixing it means changing the generator and the worker's reader
    together.  This worker builds its resampled realizations **in-process** from
    ``marinfold`` — exp82's pattern — so there are no prompt objects at all.
    ``marinfold`` installs into the vLLM image with ``--no-deps`` so the image's
    transformers pin survives.

Sampling is exp82/exp142's settled recipe, not exp98's: ``T=1.0``, ``top_p=0.95``,
**top-k disabled**, budget ``6L+128``.  Top-k 50 is the trap — it rides in from
an export's ``config.json`` when there is no ``generation_config.json``, inflates
``<end>``, and costs ~0.011 R-precision.  ``logprobs`` is not requested: the
corpus needs only ``pred``, and asking for logprobs is ~3.7x slower (#163).

Output: per-shard part parquets, never per-protein objects.

    rollouts/shard-<i>-part-<k>.parquet
      target_id, arm, entry_id, r, L, n_gen_tokens, finished, n_pred, pred,
      n_gt, tp, precision, recall, f1

Resumable: on restart the shard's existing parts are read and their
``target_id``s skipped, so a batch-band preemption resumes instead of redoing
the shard.
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

#: Flush a part file every this many proteins.  #163 lost a shard's work to a
#: preemption between flushes; a smaller number costs more objects but bounds
#: what a preemption can destroy.
FLUSH_EVERY = 250

SCHEMA = pa.schema([
    ("target_id", pa.string()), ("arm", pa.string()), ("entry_id", pa.string()),
    ("r", pa.int16()), ("L", pa.int32()),
    ("n_gen_tokens", pa.int32()), ("finished", pa.bool_()),
    ("n_pred", pa.int32()), ("pred", pa.list_(pa.int16())),
    ("n_gt", pa.int32()), ("tp", pa.int32()),
    ("precision", pa.float32()), ("recall", pa.float32()), ("f1", pa.float32()),
])

# CoreWeave AI Object Storage rejects path-style requests, so every S3 access
# goes through fsspec with virtual-hosted addressing (set by the bootstrap), and
# never through pyarrow's own S3FileSystem or the aws CLI.
os.environ.setdefault(
    "FSSPEC_S3_CONFIG_KWARGS", '{"s3": {"addressing_style": "virtual"}}'
)


def fs_for(uri: str):
    import fsspec

    return fsspec.core.url_to_fs(uri)[0]


def read_parquet(uri: str, columns=None):
    fs = fs_for(uri)
    with fs.open(uri, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def stage_model(src: str, dst: Path) -> str:
    """vLLM wants a local directory; copy the checkpoint down once."""
    if not src.startswith(("s3://", "gs://")):
        return src
    if dst.exists() and (dst / "config.json").exists():
        print(f"[gen] model already staged at {dst}", flush=True)
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    fs = fs_for(src)
    t0 = time.time()
    names = [p for p in fs.ls(src.rstrip("/"), detail=False)]
    for name in names:
        leaf = name.rsplit("/", 1)[-1]
        if not leaf:
            continue
        fs.get(name, str(dst / leaf))
    got = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
    print(f"[gen] staged {got / 1e9:.2f} GB in {time.time() - t0:.0f}s -> {dst}", flush=True)
    return str(dst)


def parse_pred(text: str, nterm: int, L: int) -> list[tuple[int, int]]:
    """Contacts from one completion, mapped ring-index -> sequence index.

    The completion continues a prompt that ends on ``<begin_statements>``, so the
    whole of ``text`` is the statements section.  Pairs outside the sequence, on
    the diagonal, or closer than ``MIN_SEP`` are dropped — the same filter the
    generator applies, so a rollout is scored against ground truth on identical
    terms.
    """
    out: set[tuple[int, int]] = set()
    for a, b in CONTACT_RE.findall(text):
        ia, ib = (int(a) - nterm) % NUM_POS, (int(b) - nterm) % NUM_POS
        if ia >= L or ib >= L or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        out.add((min(ia, ib), max(ia, ib)))
    return sorted(out)


def score(pred: list[tuple[int, int]], gt: set[tuple[int, int]]) -> tuple[int, float, float, float]:
    tp = sum(1 for p in pred if p in gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, precision, recall, f1


def existing_target_ids(out_uri: str, shard_i: int) -> set[str]:
    fs = fs_for(out_uri)
    pattern = f"{out_uri.rstrip('/')}/shard-{shard_i:04d}-part-*.parquet"
    try:
        parts = fs.glob(pattern)
    except FileNotFoundError:
        return set()
    done: set[str] = set()
    for part in parts:
        uri = part if "://" in part else f"s3://{part}"
        try:
            done |= set(read_parquet(uri, columns=["target_id"]).column("target_id").to_pylist())
        except Exception as exc:  # a part truncated by a preemption mid-write
            print(f"[gen] WARN unreadable part {uri}: {exc}", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", required=True, help="i/N")
    ap.add_argument("--n-rollouts", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets of the shard")
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from vllm import LLM, SamplingParams

    targets = read_parquet(a.targets).to_pylist()
    # Interleave the length-sorted order so every shard gets the same mix of
    # short and long proteins; otherwise the long tail decides the job's wall
    # clock (exp82's lesson, and it costs nothing).
    targets.sort(key=lambda r: (r["L"], r["target_id"]))
    mine = [r for k, r in enumerate(targets) if k % num_shards == shard_i]
    done = existing_target_ids(a.out, shard_i)
    if done:
        print(f"[gen] resuming: {len(done):,} of {len(mine):,} targets already written", flush=True)
        mine = [r for r in mine if r["target_id"] not in done]
    if a.limit:
        mine = mine[: a.limit]
    print(f"[gen] shard {shard_i}/{num_shards}: {len(mine):,} targets to generate", flush=True)
    if not mine:
        return 0

    model_dir = stage_model(a.model, Path("/tmp/exp230_model"))
    llm = LLM(
        model=model_dir,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        tensor_parallel_size=a.tensor_parallel_size,
        max_num_seqs=a.max_num_seqs,
        enforce_eager=False,
        trust_remote_code=False,
    )
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids("<end>")
    if end_id is None or end_id < 0:
        raise SystemExit("tokenizer has no <end> token — wrong tokenizer shipped with the model")
    print(f"[gen] <end> id={end_id} vocab={len(tok)}", flush=True)

    fs = fs_for(a.out)
    rows: list[dict] = []
    part = 0
    t_start = time.time()
    n_done = 0

    def flush():
        nonlocal rows, part
        if not rows:
            return
        uri = f"{a.out.rstrip('/')}/shard-{shard_i:04d}-part-{part:04d}.parquet"
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        with fs.open(uri, "wb") as fh:
            pq.write_table(table, fh)
        print(f"[gen] wrote {len(rows):,} rows -> {uri}", flush=True)
        rows = []
        part += 1

    # Start the part counter past anything already on disk so a resume cannot
    # overwrite a completed part.
    try:
        part = 1 + max(
            (int(p.rsplit("-", 1)[-1].split(".")[0])
             for p in fs.glob(f"{a.out.rstrip('/')}/shard-{shard_i:04d}-part-*.parquet")),
            default=-1,
        )
    except FileNotFoundError:
        part = 0

    for rec in mine:
        L = int(rec["L"])
        gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
        residues = residues_from_sequence(rec["sequence"])
        prompts, nterms = [], []
        for k in range(a.n_rollouts):
            # A fresh realization per rollout: resampled N-terminus and
            # statement order. exp82's "resample" half of the recipe — free,
            # and all realizations share a prefix length so they batch.
            built = build_document(f"{rec['target_id']}:r{k}", residues, [],
                                   config=GenerationConfig())
            if built is None:
                continue
            doc = built.document
            prompts.append(doc[: doc.index(BEGIN) + len(BEGIN)])
            nterms.append(built.n_term_index)
        if not prompts:
            print(f"[gen] WARN {rec['target_id']}: no realization built", flush=True)
            continue
        # The budget is exp142's 6L+128, but it must also fit the context: all
        # realizations of one protein share a prompt length, so one measurement
        # covers the batch.
        plen = len(tok(prompts[0], add_special_tokens=False).input_ids)
        max_new = min(a.max_model_len - plen, 6 * L + 128)
        params = SamplingParams(
            temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
            max_tokens=max_new, n=1,
            # <end> terminates a document, and the position/residue/contact
            # tokens are all "special" in this vocab — detokenising with
            # skip_special_tokens would return an EMPTY string and every
            # rollout would silently parse to zero contacts.
            stop_token_ids=[end_id], skip_special_tokens=False,
        )
        outs = llm.generate(prompts, params, use_tqdm=False)
        for r, (out, nterm) in enumerate(zip(outs, nterms)):
            comp = out.outputs[0]
            pred = parse_pred(comp.text, nterm, L)
            tp, precision, recall, f1 = score(pred, gt)
            rows.append({
                "target_id": rec["target_id"], "arm": rec["arm"],
                "entry_id": rec["entry_id"], "r": r, "L": L,
                "n_gen_tokens": len(comp.token_ids),
                "finished": comp.finish_reason == "stop",
                "n_pred": len(pred),
                "pred": [int(v) for pair in pred for v in pair],
                "n_gt": len(gt), "tp": tp,
                "precision": precision, "recall": recall, "f1": f1,
            })
        n_done += 1
        if n_done % FLUSH_EVERY == 0:
            flush()
            rate = n_done / max(time.time() - t_start, 1)
            print(f"[gen] {n_done:,}/{len(mine):,} targets, {rate * 3600:.0f}/h", flush=True)
    flush()
    print(f"[gen] shard {shard_i} done: {n_done:,} targets in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
