# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Generate `<contacts-v1.multi>` rollouts and emit **one row per section**.

The five aggregation modes (#230 follow-up) all read the same generations:

1. plain `<contacts-v1>` — NOT this worker; that is Gate A's exp82 scorer.
2. consensus vote across the sections of a single rollout
3. the best section of a single rollout (**oracle** — it selects using ground truth)
4. the last section
5. the second-to-last section

Modes 2–5 differ only in how the sections are *combined*, never in how they are
generated, so this worker deliberately stops at the sections and writes them out.
Scoring happens offline in ``score_agg_modes.py``. That split is the point: a new
aggregation rule costs a re-score of a parquet, not another pass over 8 GPUs, and
every mode is guaranteed to be reading byte-identical generations rather than
independently sampled ones — which is what makes the comparison between them
paired.

Section splitting and contact parsing are **imported** from
``eval_modes_worker``, not re-implemented, so Gate B and these modes cannot drift
apart in what they call a section.

    python eval_agg_worker.py --model <hf> --targets <parquet> --out <dir> --shard 0/8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eval_modes_worker import (BEGIN, MULTI_TOKEN, PLAIN_TOKEN, END, contacts_of,
                               fs_for, read_parquet, split_sections, stage_model)

SCHEMA = pa.schema([
    ("target_id", pa.string()), ("dataset", pa.string()), ("stem", pa.string()),
    ("L", pa.int32()), ("r", pa.int16()),
    ("n_sections", pa.int32()),          # sections in THIS rollout (uncapped)
    ("sec_idx", pa.int32()),             # 0-based position of this section
    ("n_contacts", pa.int32()),
    ("contacts", pa.list_(pa.list_(pa.int32(), 2))),
    ("finished", pa.bool_()),            # did the rollout emit <end>
    ("n_gen_tokens", pa.int32()),
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", required=True, help="i/N")
    ap.add_argument("--n-rollouts", type=int, default=8)
    # Generous on purpose: these modes need the model's NATURAL section count.
    # Clipping here would silently cap "last" and "second-to-last" at the cap.
    ap.add_argument("--max-sections", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from vllm import LLM, SamplingParams

    targets = read_parquet(a.targets).to_pylist()
    targets.sort(key=lambda r: (r["L"], r["target_id"]))
    mine = [r for k, r in enumerate(targets) if k % num_shards == shard_i]
    if a.limit:
        mine = mine[: a.limit]
    print(f"[agg] shard {shard_i}/{num_shards}: {len(mine)} proteins", flush=True)

    model_dir = stage_model(a.model, Path("/tmp/exp230_agg_model"))
    llm = LLM(model=model_dir, max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size,
              max_num_seqs=a.max_num_seqs, trust_remote_code=False)
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids(END)
    mode_id = tok.convert_tokens_to_ids(MULTI_TOKEN)
    unk = getattr(tok, "unk_token_id", None)
    if mode_id is None or mode_id < 0 or (unk is not None and mode_id == unk):
        raise SystemExit(f"tokenizer does not know {MULTI_TOKEN!r} -- the RENAMED "
                         "tokenizer (id 7) must ship with these weights")
    print(f"[agg] {MULTI_TOKEN} id={mode_id} <end> id={end_id} vocab={len(tok)}", flush=True)

    fs = fs_for(a.out)
    out_dir = a.out.rstrip("/")
    rows: list[dict] = []
    part = 0

    def flush():
        nonlocal rows, part
        if not rows:
            return
        uri = f"{out_dir}/shard-{shard_i:04d}-part-{part:04d}.parquet"
        fs.makedirs(out_dir, exist_ok=True)
        with fs.open(uri, "wb") as fh:
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), fh)
        print(f"[agg] wrote {len(rows)} rows -> {uri}", flush=True)
        rows = []
        part += 1

    t0 = time.time()
    for n_done, rec in enumerate(mine, 1):
        L = int(rec["L"])
        residues = residues_from_sequence(rec["input_seq"])
        prompts, nterms = [], []
        for k in range(a.n_rollouts):
            built = build_document(f"{rec['target_id']}:a{k}", residues, [],
                                   config=GenerationConfig())
            if built is None:
                continue
            doc = built.document
            head = doc[: doc.index(BEGIN) + len(BEGIN)]
            assert head.startswith(PLAIN_TOKEN)
            prompts.append(MULTI_TOKEN + head[len(PLAIN_TOKEN):])
            nterms.append(built.n_term_index)
        if not prompts:
            continue

        plen = len(tok(prompts[0], add_special_tokens=False).input_ids)
        params = SamplingParams(
            temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
            max_tokens=max(16, a.max_model_len - plen), n=1,
            stop_token_ids=[end_id], skip_special_tokens=False,
        )
        outs = llm.generate(prompts, params, use_tqdm=False)
        for r, (out, nterm) in enumerate(zip(outs, nterms)):
            comp = out.outputs[0]
            raw = split_sections(comp.text)
            secs = [contacts_of(s, nterm, L) for s in raw]
            fin = comp.finish_reason == "stop"
            ntok = len(comp.token_ids)
            if not secs:
                # A rollout with no parsable section still happened; record it so
                # the denominator stays the number of ROLLOUTS, not of sections.
                rows.append({
                    "target_id": rec["target_id"], "dataset": rec["dataset"],
                    "stem": rec["stem"], "L": L, "r": r, "n_sections": 0,
                    "sec_idx": -1, "n_contacts": 0, "contacts": [],
                    "finished": fin, "n_gen_tokens": ntok})
                continue
            for si, s in enumerate(secs):
                rows.append({
                    "target_id": rec["target_id"], "dataset": rec["dataset"],
                    "stem": rec["stem"], "L": L, "r": r, "n_sections": len(secs),
                    "sec_idx": si, "n_contacts": len(s),
                    "contacts": [[int(i), int(j)] for i, j in sorted(s)],
                    "finished": fin, "n_gen_tokens": ntok})
        if n_done % 25 == 0:
            el = time.time() - t0
            print(f"[agg] [{n_done}/{len(mine)}] L={L} {el:.0f}s "
                  f"{len(rows)} rows buffered", flush=True)
            flush()
    flush()
    print(f"[agg] shard {shard_i} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
