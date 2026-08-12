# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1, local variant (issue #211) — generate the rollouts on this workstation.

``dispatch_rollouts_cw.py`` fans ``gen_rollouts_worker.py`` over 16 CoreWeave
H100s. That path is **blocked**: the workstation's CoreWeave credentials are
rejected ("the access key ID you provided does not exist in our records") and the
fresh iris checkout no longer declares a ``coreweave`` store. Both are
environment problems, not code problems, and neither is fixable from here.

It also turns out not to matter. exp82 measured the full 554-protein eval set at
n=100 rollouts taking **~80 min on one A5000** — the cluster fan-out existed to
turn 80 minutes into 5, not to make the job possible. So this runs the identical
measurement locally against local paths.

**The recipe is byte-identical to the worker's**, deliberately: same
``build_document`` prompt realization per rollout, same ``T=1.0`` /
``top_p=0.95`` / ``top_k=-1`` (#142), same ``max_new = 6L + 128``, same
``parse_rollout`` readout, same two output tables. Only the I/O layer differs, so
the rollouts are comparable to every published contacts-v1 number.

Run in the dedicated vLLM venv (the analysis venv pins torch 2.5.1+cu121, which
vLLM would fight over)::

    _scratch/vllmenv/bin/python run_rollouts_local.py \\
        --model ~/.cache/marinfold/<...> --targets data/eval_targets.parquet \\
        --out _scratch/rollouts --n-rollouts 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rollouts_worker import (  # noqa: E402
    BEGIN, CONTACT_SCHEMA, NUM_POS, ROLLOUT_SCHEMA, parse_rollout,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="data/eval_targets.parquet")
    ap.add_argument("--out", type=Path, default=Path("_scratch/rollouts"))
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip torch.compile; needed on hosts whose CUDA "
                         "driver the compiled path will not build against")
    a = ap.parse_args()

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    recs = pq.read_table(a.targets).to_pylist()
    recs.sort(key=lambda r: r["L"])
    if a.limit:
        recs = recs[: a.limit]

    (a.out / "contacts").mkdir(parents=True, exist_ok=True)
    (a.out / "rollouts").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (a.out / "contacts").glob("*.parquet")}

    tok = AutoTokenizer.from_pretrained(a.model)
    end_id = tok.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=a.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=a.max_num_seqs, seed=a.seed,
              enforce_eager=a.enforce_eager)

    todo = [r for r in recs if f"{r['dataset']}__{r['stem']}" not in done]
    print(f"[local] {len(todo)}/{len(recs)} proteins to do, "
          f"n_rollouts={a.n_rollouts}", flush=True)

    t0, n_unfinished, n_total = time.time(), 0, 0
    for s in range(0, len(todo), a.chunk):
        group = todo[s:s + a.chunk]
        prompts, per, sps = [], [], []
        for r in group:
            residues = residues_from_sequence(r["input_seq"])
            first, maps = len(prompts), []
            for k in range(a.n_rollouts):
                doc = build_document(f"{r['stem']}:r{k}", residues, [],
                                     config=GenerationConfig())
                prompts.append(doc.document[: doc.document.index(BEGIN) + len(BEGIN)])
                maps.append({(doc.n_term_index + t) % NUM_POS: t
                             for t in range(doc.seq_len)})
            plen = len(tok(prompts[first], add_special_tokens=False).input_ids)
            max_new = min(8192 - plen, a.contact_mult * r["L"] + 128)
            per.append((r, first, maps))
            sps += [SamplingParams(temperature=a.temperature, top_p=a.top_p,
                                   top_k=a.top_k, max_tokens=max_new,
                                   stop_token_ids=[end_id], skip_special_tokens=False,
                                   seed=a.seed * 1_000_003 + first + k)
                    for k in range(a.n_rollouts)]
        ts = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - ts

        for r, first, maps in per:
            crows = {c: [] for c in CONTACT_SCHEMA.names}
            rrows = {c: [] for c in ROLLOUT_SCHEMA.names}
            chunk_outs = outs[first:first + a.n_rollouts]
            n_unfinished += sum(1 for o in chunk_outs
                                if o.outputs[0].finish_reason != "stop")
            n_total += a.n_rollouts
            for k, (o, seqidx) in enumerate(zip(chunk_outs, maps)):
                contacts, n_em, n_oor, n_close = parse_rollout(o.outputs[0].text, seqidx)
                for order, (i, j, dup) in enumerate(contacts):
                    crows["dataset"].append(r["dataset"]); crows["stem"].append(r["stem"])
                    crows["L"].append(r["L"]); crows["rollout"].append(k)
                    crows["order"].append(order); crows["i"].append(i)
                    crows["j"].append(j); crows["duplicate"].append(dup)
                rrows["dataset"].append(r["dataset"]); rrows["stem"].append(r["stem"])
                rrows["L"].append(r["L"]); rrows["rollout"].append(k)
                rrows["n_contacts"].append(sum(1 for _, _, d in contacts if not d))
                rrows["n_emitted"].append(n_em)
                rrows["n_tokens"].append(len(o.outputs[0].token_ids))
                rrows["finished"].append(o.outputs[0].finish_reason == "stop")
                rrows["n_out_of_range"].append(n_oor)
                rrows["n_too_close"].append(n_close)
            name = f"{r['dataset']}__{r['stem']}.parquet"
            # Rollouts first: the resume scan reads the contacts dir, so a kill
            # between the two writes must not mark the protein complete.
            pq.write_table(pa.table(rrows, schema=ROLLOUT_SCHEMA),
                           a.out / "rollouts" / name, compression="zstd")
            pq.write_table(pa.table(crows, schema=CONTACT_SCHEMA),
                           a.out / "contacts" / name, compression="zstd")

        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[local] [{s + len(group)}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
              f"{dt:6.1f}s {ntok / dt:7.0f} tok/s unfinished={n_unfinished}/{n_total} "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    print(f"[local] DONE {len(todo)} proteins in {(time.time() - t0) / 60:.1f} min | "
          f"unfinished {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
