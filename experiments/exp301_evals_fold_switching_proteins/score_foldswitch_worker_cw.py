#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp301 worker — which fold does MarinFold predict, and is the alternative reachable?

One shard of the fold-switching universe on one H100. Two passes over the same
loaded engine, because the model load dominates the work:

**Pass 1 (M1, rollouts).** exp82's fixed recipe — a fresh document realization per
rollout, T=1.0, top-p 0.95, top-k off, budget ``6L+128`` capped by the context —
repeated under several seeds. Per rollout we record hits against the three sets
``prepare_inputs.py`` built: ``A`` (fold1-unique), ``B`` (fold2-unique) and ``S``
(shared). The fold score

    phi = |rollout n A| / |A|  -  |rollout n B| / |B|

is computed in analysis from those counts. Predictions are restricted to the
common universe first — a contact on a residue only one structure resolves is
not scorable against either fold.

**Pass 2 (M2, teacher-forced NLL).** Sampling only sees what the model emits
often; scoring sees the whole distribution. For each pair we build fold1's and
fold2's ground-truth documents **under the same ``entry_id``**, which makes the
sequence-section realization byte-identical (verified: same start index, same
statement order), so the contact set is the only difference between the two
documents. Reported per statement, not per document, and in a ``matched``
variant where both folds are cut to the same number of contacts — otherwise the
comparison is confounded by |A| and |B| differing.

Outputs three parquet families under ``<out>/<label>/``: ``rollouts/``,
``votes/`` and ``nll/``. Resumable — a preempted shard skips pairs already
written.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEGIN = "<begin_statements>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NUM_POS = 2000
MIN_SEP = 6
CONTEXT = 8192

ROLLOUT_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("role", pa.string()),
    ("seed", pa.int16()), ("rollout", pa.int16()),
    ("L", pa.int32()),
    ("n_pred_raw", pa.int32()), ("n_pred", pa.int32()), ("n_pred_fs", pa.int32()),
    ("n_hit_a", pa.int32()), ("n_hit_b", pa.int32()), ("n_hit_s", pa.int32()),
    ("n_hit_a_fs", pa.int32()), ("n_hit_b_fs", pa.int32()),
    ("n_a", pa.int32()), ("n_b", pa.int32()), ("n_s", pa.int32()),
    ("n_a_fs", pa.int32()), ("n_b_fs", pa.int32()),
    ("finished", pa.bool_()), ("n_tokens", pa.int32()),
    ("max_new", pa.int32()), ("budget_capped", pa.bool_()),
])

VOTE_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("seed", pa.int16()),
    ("i", pa.int16()), ("j", pa.int16()), ("votes", pa.int16()),
])

NLL_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("realization", pa.int16()), ("fold", pa.int8()),
    ("variant", pa.string()), ("n_statements", pa.int32()),
    ("n_statement_tokens", pa.int32()), ("sum_logprob", pa.float64()),
    ("prefix_tokens", pa.int32()), ("doc_tokens", pa.int32()),
])


def stage_model(src: str, dst: Path) -> Path:
    """Copy a remote model directory to local disk (vLLM needs a real directory).

    Any fsspec URL works — ``hf://buckets/...`` reads the public MarinFold bucket
    anonymously, which is how this worker gets the checkpoint without a CoreWeave
    S3 staging step.
    """
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


def read_parquet(uri: str):
    import fsspec

    with fsspec.open(uri, "rb") as fh:
        return pq.read_table(fh)


def write_parquet(tbl, uri: str) -> None:
    import fsspec

    with fsspec.open(uri, "wb") as fh:
        pq.write_table(tbl, fh, compression="zstd")


def done_pairs(out_dir: str, shard_i: int, num_shards: int) -> tuple[set[str], int]:
    """Pairs this shard already wrote rollouts for, and the next part number."""
    import fsspec

    pattern = f"{out_dir}/rollouts/shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet"
    try:
        fs, _ = fsspec.core.url_to_fs(out_dir)
        paths = fs.glob(pattern)
    except FileNotFoundError:
        return set(), 0
    seen: set[str] = set()
    for path in paths:
        protocol = out_dir.split("://", 1)[0] if "://" in out_dir else ""
        uri = f"{protocol}://{path}" if protocol and "://" not in path else path
        seen |= set(read_parquet(uri).column("pair_id").to_pylist())
    return seen, len(paths)


def canonical(pairs) -> set[tuple[int, int]]:
    """``[[i, j], ...]`` from the universe file as a set of ordered tuples."""
    return {(int(i), int(j)) for i, j in pairs}


def in_region(pair: tuple[int, int], region) -> bool:
    if region is None:
        return False
    lo, hi = region
    return lo <= pair[0] < hi or lo <= pair[1] < hi


def parse_rollout(text: str, seq_index: dict[int, int]) -> set[tuple[int, int]]:
    """Contacts emitted by one rollout, in reference coordinates."""
    out: set[tuple[int, int]] = set()
    for x, y in CONTACT_RE.findall(text):
        a, b = seq_index.get(int(x)), seq_index.get(int(y))
        if a is None or b is None or a == b or abs(a - b) < MIN_SEP:
            continue
        out.add((min(a, b), max(a, b)))
    return out


def main() -> int:  # noqa: C901 — one long linear pipeline reads better than five hops
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True, help="s3 prefix; parts land under <out>/<label>/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-nll-realizations", type=int, default=4)
    ap.add_argument("--skip-nll", action="store_true")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))
    out_dir = f"{a.out.rstrip('/')}/{a.label}"

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
        residues_from_sequence,
    )
    from marinfold.document_structures.contacts_v1.parse import RawContact
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = GenerationConfig()
    recs = read_parquet(a.targets).to_pylist()
    recs.sort(key=lambda r: r["L"])
    # Interleave by length: a shard that collected all the long proteins would
    # set the wall clock for the whole fan-out.
    mine = [r for k, r in enumerate(recs) if k % num_shards == shard_i]
    skip, n_parts = done_pairs(out_dir, shard_i, num_shards)
    todo = [r for r in mine if r["pair_id"] not in skip]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[worker] shard {shard_i}/{num_shards}: {len(mine)} assigned, {len(skip)} done, "
          f"{len(todo)} to do | rollouts={a.n_rollouts} seeds={a.n_seeds} "
          f"T={a.temperature} top_p={a.top_p} top_k={a.top_k}", flush=True)
    if not todo:
        print("[worker] nothing to do")
        return 0

    model_dir = stage_model(a.model, Path("/tmp/marinfold_model"))
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    end_id = tok.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=CONTEXT,
              gpu_memory_utilization=a.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=a.max_num_seqs, seed=a.seed)

    t0 = time.time()
    for n, rec in enumerate(todo, 1):
        pair_id, L, role = rec["pair_id"], int(rec["L"]), rec["role"]
        residues = residues_from_sequence(rec["sequence"])
        set_a = canonical(rec["contacts_fold1"])
        set_b = canonical(rec["contacts_fold2"])
        common = {int(p) for p in rec["common_positions"]}
        region = None if rec["fs_lo"] < 0 else (int(rec["fs_lo"]), int(rec["fs_hi"]))
        only_a, only_b, shared = set_a - set_b, set_b - set_a, set_a & set_b
        a_fs = {p for p in only_a if in_region(p, region)}
        b_fs = {p for p in only_b if in_region(p, region)}

        # ---- Pass 1: rollouts -------------------------------------------------
        prompts: list[str] = []
        index_maps: list[dict[int, int]] = []
        for seed in range(a.n_seeds):
            for k in range(a.n_rollouts):
                doc = build_document(f"{pair_id}:s{seed}:r{k}", residues, [], config=cfg)
                prompts.append(doc.document[: doc.document.index(BEGIN) + len(BEGIN)])
                index_maps.append({(doc.n_term_index + t) % NUM_POS: t for t in range(doc.seq_len)})
        plen = len(tok(prompts[0], add_special_tokens=False).input_ids)
        wanted = a.contact_mult * L + 128
        max_new = min(CONTEXT - plen, wanted)
        sps = [
            SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                           max_tokens=max_new, stop_token_ids=[end_id],
                           skip_special_tokens=False, seed=a.seed * 1_000_003 + idx)
            for idx in range(len(prompts))
        ]
        ts = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        gen_dt = time.time() - ts

        rollout_rows = {c.name: [] for c in ROLLOUT_SCHEMA}
        vote_rows = {c.name: [] for c in VOTE_SCHEMA}
        votes = np.zeros((a.n_seeds, L, L), np.int16)
        for idx, (out, seq_index) in enumerate(zip(outs, index_maps)):
            seed, k = divmod(idx, a.n_rollouts)
            raw = parse_rollout(out.outputs[0].text, seq_index)
            pred = {p for p in raw if p[0] in common and p[1] in common}
            pred_fs = {p for p in pred if in_region(p, region)}
            for i, j in pred:
                votes[seed, i, j] += 1
            rollout_rows["pair_id"].append(pair_id)
            rollout_rows["role"].append(role)
            rollout_rows["seed"].append(seed)
            rollout_rows["rollout"].append(k)
            rollout_rows["L"].append(L)
            rollout_rows["n_pred_raw"].append(len(raw))
            rollout_rows["n_pred"].append(len(pred))
            rollout_rows["n_pred_fs"].append(len(pred_fs))
            rollout_rows["n_hit_a"].append(len(pred & only_a))
            rollout_rows["n_hit_b"].append(len(pred & only_b))
            rollout_rows["n_hit_s"].append(len(pred & shared))
            rollout_rows["n_hit_a_fs"].append(len(pred & a_fs))
            rollout_rows["n_hit_b_fs"].append(len(pred & b_fs))
            rollout_rows["n_a"].append(len(only_a))
            rollout_rows["n_b"].append(len(only_b))
            rollout_rows["n_s"].append(len(shared))
            rollout_rows["n_a_fs"].append(len(a_fs))
            rollout_rows["n_b_fs"].append(len(b_fs))
            rollout_rows["finished"].append(out.outputs[0].finish_reason == "stop")
            rollout_rows["n_tokens"].append(len(out.outputs[0].token_ids))
            rollout_rows["max_new"].append(max_new)
            rollout_rows["budget_capped"].append(max_new < wanted)

        for seed in range(a.n_seeds):
            ii, jj = np.nonzero(np.triu(votes[seed], k=1))
            vote_rows["pair_id"] += [pair_id] * len(ii)
            vote_rows["seed"] += [seed] * len(ii)
            vote_rows["i"] += ii.astype(np.int16).tolist()
            vote_rows["j"] += jj.astype(np.int16).tolist()
            vote_rows["votes"] += votes[seed][ii, jj].tolist()

        # ---- Pass 2: teacher-forced NLL of both folds -------------------------
        nll_rows = {c.name: [] for c in NLL_SCHEMA}
        if not a.skip_nll and role == "foldswitch":
            docs, meta = [], []
            for m in range(a.n_nll_realizations):
                rng = random.Random(f"{pair_id}:{m}")
                matched_n = min(len(set_a), len(set_b))
                variants = {
                    "full": (sorted(set_a), sorted(set_b)),
                    # Random subsets rather than the strongest-N that contacts_v1
                    # uses: we do not carry degrees, and a random cut also avoids
                    # making the comparison depend on the degree distribution.
                    "matched": (rng.sample(sorted(set_a), matched_n),
                                rng.sample(sorted(set_b), matched_n)),
                }
                for variant, (ca, cb) in variants.items():
                    for fold, contacts in ((1, ca), (2, cb)):
                        # Same entry_id for both folds => byte-identical sequence
                        # section, so the contact set is the only difference.
                        doc = build_document(
                            f"{pair_id}:nll{m}", residues,
                            [RawContact(i, j, 1.0) for i, j in contacts], config=cfg)
                        docs.append(doc.document)
                        meta.append((m, fold, variant, len(contacts), doc))
            prefix_lens = []
            for text in docs:
                prefix_lens.append(len(tok(text[: text.index(BEGIN) + len(BEGIN)],
                                           add_special_tokens=False).input_ids))
            scored = llm.generate(
                docs,
                [SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)] * len(docs),
                use_tqdm=False)
            for (m, fold, variant, n_contacts, doc), out, pre in zip(meta, scored, prefix_lens):
                lps = out.prompt_logprobs
                # prompt_logprobs[0] is None (no distribution before token 0);
                # entry t is a {token_id: Logprob} for the realized token t.
                total = 0.0
                counted = 0
                for t in range(pre, len(lps)):
                    entry = lps[t]
                    if not entry:
                        continue
                    total += float(next(iter(entry.values())).logprob)
                    counted += 1
                nll_rows["pair_id"].append(pair_id)
                nll_rows["realization"].append(m)
                nll_rows["fold"].append(fold)
                nll_rows["variant"].append(variant)
                nll_rows["n_statements"].append(n_contacts)
                nll_rows["n_statement_tokens"].append(counted)
                nll_rows["sum_logprob"].append(total)
                nll_rows["prefix_tokens"].append(pre)
                nll_rows["doc_tokens"].append(len(lps))

        stem = f"shard-{shard_i:03d}-of-{num_shards:03d}-part-{n_parts:04d}.parquet"
        write_parquet(pa.table(rollout_rows, schema=ROLLOUT_SCHEMA), f"{out_dir}/rollouts/{stem}")
        write_parquet(pa.table(vote_rows, schema=VOTE_SCHEMA), f"{out_dir}/votes/{stem}")
        if nll_rows["pair_id"]:
            write_parquet(pa.table(nll_rows, schema=NLL_SCHEMA), f"{out_dir}/nll/{stem}")
        n_parts += 1

        hits_a = sum(rollout_rows["n_hit_a"]) / max(len(only_a) * len(outs), 1)
        hits_b = sum(rollout_rows["n_hit_b"]) / max(len(only_b) * len(outs), 1)
        # A calibration row carries no fold2, so its "phi" is just recall on
        # its own ground truth — the published-number gate, not a fold score.
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[worker] [{n}/{len(todo)}] {pair_id:16s} L={L:4d} |A|={len(only_a):4d} "
              f"|B|={len(only_b):4d} recall_A={hits_a:.3f} recall_B={hits_b:.3f} "
              f"phi={hits_a - hits_b:+.3f} unfinished="
              f"{sum(1 for x in rollout_rows['finished'] if not x)}/{len(outs)} "
              f"{gen_dt:5.1f}s {ntok / max(gen_dt, 1e-9):7.0f} tok/s "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    print(f"[worker] DONE shard {shard_i}/{num_shards}: {len(todo)} pairs in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
