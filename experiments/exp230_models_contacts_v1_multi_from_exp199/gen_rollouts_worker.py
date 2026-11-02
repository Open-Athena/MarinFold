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

import pyarrow as pa
import pyarrow.parquet as pq

BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

#: Flush a part file every this many proteins -- the bound on what ONE preemption
#: can destroy, and on a preemptible pool that is the number that matters.
#:
#: Measured on this run: shards see 2-4 preemptions each, and at 250 a flush is
#: 12-30 min apart, so each preemption discarded ~6-15 min of generation. At 50
#: that falls to ~1-3 min. The cost is 5x more part files (48 shards x ~45 =
#: ~2,200 objects), which is nothing next to redoing the work.
FLUSH_EVERY = int(os.environ.get("EXP230_FLUSH_EVERY", "50"))

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


def with_scheme(path: str, like: str) -> str:
    """Re-attach ``like``'s URI scheme to a bare path returned by ``fs.glob``.

    fsspec's ``glob`` strips the protocol, so a GCS listing comes back as
    ``bucket/key`` with no ``gs://``.  Hardcoding a scheme here is the bug this
    function exists to prevent: re-attaching ``s3://`` to a GCS key produced a
    URI that resolved to a different filesystem, every read raised, the failure
    was swallowed by the resume path's ``except``, and resume silently became a
    no-op -- on a *preemptible* pool, where a shard killed at 90 % then redoes
    the entire shard.
    """
    if "://" in path or "://" not in like:
        return path
    return f"{like.split('://', 1)[0]}://{path}"


def existing_target_ids(out_uri: str, shard_i: int) -> set[str]:
    """Target ids already written for this shard, so a restart resumes.

    A part that cannot be read is skipped but LOUD: the only benign cause is a
    write truncated by a preemption, and any other cause means work is about to
    be redone silently.
    """
    fs = fs_for(out_uri)
    pattern = f"{out_uri.rstrip('/')}/shard-{shard_i:04d}-part-*.parquet"
    try:
        parts = fs.glob(pattern)
    except FileNotFoundError:
        return set()
    done: set[str] = set()
    n_bad = 0
    for part in parts:
        uri = with_scheme(part, out_uri)
        try:
            done |= set(read_parquet(uri, columns=["target_id"]).column("target_id").to_pylist())
        except Exception as exc:  # a part truncated by a preemption mid-write
            n_bad += 1
            print(f"[gen] WARN unreadable part {uri}: {type(exc).__name__}: {exc}", flush=True)
    if parts and not done:
        print(f"[gen] WARN {len(parts)} part(s) found but NONE readable -- resume is "
              "not working and this shard is about to redo its work", flush=True)
    elif n_bad:
        print(f"[gen] {n_bad}/{len(parts)} part(s) unreadable (truncated writes)", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", required=True,
                    help="i/N, or a comma list i,j,k/N to run several shards in ONE "
                         "process. Building the engine costs minutes, so a process per "
                         "shard pays that once per SHARD instead of once per device.")
    ap.add_argument("--n-rollouts", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip inductor compilation. Slower per token, but immune to "
                         "the compile-cache races that crash simultaneous engine starts")
    ap.add_argument("--skip-targets", default=None,
                    help="parquet of target_ids already generated ELSEWHERE. Per-shard "
                         "resume only sees this shard's own parts, so it cannot help "
                         "when the target file is re-split and a protein moves shard.")
    ap.add_argument("--chunk", type=int, default=32,
                    help="proteins per generate() call. chunk x n_rollouts should be at "
                         "least max_num_seqs, or the engine runs half empty")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    a = ap.parse_args()

    shard_spec, num_shards_s = a.shard.rsplit("/", 1)
    num_shards = int(num_shards_s)
    shard_list = [int(x) for x in shard_spec.split(",") if x.strip()]

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from vllm import LLM, SamplingParams

    targets = read_parquet(a.targets).to_pylist()
    # Interleave the length-sorted order so every shard gets the same mix of
    # short and long proteins; otherwise the long tail decides the wall clock
    # (exp82's lesson, and it costs nothing). It also means any PREFIX of the
    # fleet is a uniform sample, so a partial run is unbiased.
    targets.sort(key=lambda r: (r["L"], r["target_id"]))
    print(f"[gen] {len(targets):,} targets | shards {shard_list} of {num_shards}", flush=True)

    prior: set[str] = set()
    if a.skip_targets:
        prior = set(read_parquet(a.skip_targets, columns=["target_id"])
                    .column("target_id").to_pylist())
        print(f"[gen] skip-list: {len(prior):,} target_ids generated elsewhere", flush=True)

    # ONE engine for ALL this process's shards. Building it costs minutes (weight
    # load plus inductor compile); a process per shard paid that once per shard,
    # and simultaneous starts raced on the compile cache -- which is what killed
    # 5 of 8 GPUs on the first launch of this fleet.
    model_dir = stage_model(a.model, Path("/tmp/exp230_model"))
    llm = LLM(
        model=model_dir,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        tensor_parallel_size=a.tensor_parallel_size,
        max_num_seqs=a.max_num_seqs,
        enforce_eager=a.enforce_eager,
        trust_remote_code=False,
    )
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids("<end>")
    if end_id is None or end_id < 0:
        raise SystemExit("tokenizer has no <end> token -- wrong tokenizer shipped with the model")
    print(f"[gen] <end> id={end_id} vocab={len(tok)} eager={a.enforce_eager}", flush=True)

    fs = fs_for(a.out)

    def run_shard(shard_i: int) -> None:
        mine = [r for k, r in enumerate(targets) if k % num_shards == shard_i]
        n0 = len(mine)
        if prior:
            mine = [r for r in mine if r["target_id"] not in prior]
        done = existing_target_ids(a.out, shard_i)
        if done:
            mine = [r for r in mine if r["target_id"] not in done]
        if a.limit:
            mine = mine[: a.limit]
        print(f"[gen] shard {shard_i}/{num_shards}: {len(mine):,} of {n0:,} to generate",
              flush=True)
        if not mine:
            return

        rows: list[dict] = []
        # Start the part counter past anything on disk so a resume cannot
        # overwrite a completed part.
        try:
            part = 1 + max(
                (int(q.rsplit("-", 1)[-1].split(".")[0])
                 for q in fs.glob(f"{a.out.rstrip('/')}/shard-{shard_i:04d}-part-*.parquet")),
                default=-1,
            )
        except FileNotFoundError:
            part = 0

        def flush() -> None:
            nonlocal rows, part
            if not rows:
                return
            uri = f"{a.out.rstrip('/')}/shard-{shard_i:04d}-part-{part:04d}.parquet"
            fs.makedirs(a.out.rstrip("/"), exist_ok=True)
            with fs.open(uri, "wb") as fh:
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), fh)
            print(f"[gen] wrote {len(rows):,} rows -> {uri}", flush=True)
            rows = []
            part += 1

        t0 = time.time()
        n_done = 0
        # Generate CHUNK proteins at a time. One protein is only n_rollouts
        # prompts, against an engine configured for hundreds of concurrent
        # sequences, so a per-protein call fills a few percent of the batch and
        # the device idles between calls. max_tokens is per-protein, so
        # SamplingParams is a LIST aligned with prompts.
        for s0 in range(0, len(mine), a.chunk):
            group = mine[s0: s0 + a.chunk]
            prompts: list[str] = []
            params: list = []
            per: list[tuple[dict, int, list[int]]] = []
            for rec in group:
                L = int(rec["L"])
                residues = residues_from_sequence(rec["sequence"])
                first, nterms = len(prompts), []
                for k in range(a.n_rollouts):
                    # Fresh realization per rollout: resampled N-terminus and
                    # statement order -- exp82's "resample" half of the recipe.
                    built = build_document(f"{rec['target_id']}:r{k}", residues, [],
                                           config=GenerationConfig())
                    if built is None:
                        continue
                    doc = built.document
                    prompts.append(doc[: doc.index(BEGIN) + len(BEGIN)])
                    nterms.append(built.n_term_index)
                if not nterms:
                    continue
                plen = len(tok(prompts[first], add_special_tokens=False).input_ids)
                max_new = min(a.max_model_len - plen, 6 * L + 128)
                params += [SamplingParams(
                    temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                    max_tokens=max_new, n=1,
                    # <end> terminates a document, and the position/residue/contact
                    # tokens are all "special" in this vocab -- detokenising with
                    # skip_special_tokens would return an EMPTY string and every
                    # rollout would silently parse to zero contacts.
                    stop_token_ids=[end_id], skip_special_tokens=False,
                )] * len(nterms)
                per.append((rec, first, nterms))
            if not prompts:
                continue
            outs = llm.generate(prompts, params, use_tqdm=False)
            for rec, first, nterms in per:
                L = int(rec["L"])
                gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
                for r, nterm in enumerate(nterms):
                    comp = outs[first + r].outputs[0]
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
            if n_done % FLUSH_EVERY < a.chunk:
                flush()
                rate = n_done / max(time.time() - t0, 1)
                print(f"[gen] shard {shard_i}: {n_done:,}/{len(mine):,}, {rate * 3600:.0f}/h "
                      f"({len(prompts)} prompts/call)", flush=True)
        flush()
        print(f"[gen] shard {shard_i} done: {n_done:,} targets in "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    for shard_i in shard_list:
        try:
            run_shard(shard_i)
        except Exception as exc:  # noqa: BLE001 -- one bad shard must not end the process
            print(f"[gen] ERROR shard {shard_i}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
