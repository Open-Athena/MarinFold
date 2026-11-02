# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does the prepared checkpoint still emit ~22 sections in SkyRL's own vLLM? — issue #237.

This is the pre-flight that separates "the RL harness did something" from "the
model was already broken when the harness loaded it". Three things can quietly
break between #230's evaluation and this experiment's first training step, and
all three produce fluent-looking rollouts rather than an error:

* **rope.** #230 measured its checkpoint under vLLM 0.27 / transformers 5.15.
  SkyRL's venv is vLLM 0.23 / transformers 5.8, a different reader of the same
  config. A reader that misses the llama3 scaling loses 0.76 nats/token silently
  (#163's retraction).
* **the mode marker.** Id 7 must still be ``<contacts-v1.multi>``; if the wrong
  tokenizer travelled with the weights, generation falls back to single-document
  and every section-level reward in this experiment is computed over one section.
* **the sections themselves.** #237's kill criterion is "mean sections per
  rollout below 12". That threshold is only meaningful if the run STARTS near
  #230's 22.0, and the training prompts are #208's pool rather than #230's eval
  set — a different length mix, and a multi rollout's section count is bounded by
  the context left after the sequence header.

Reports the numbers the gates are set from, on the actual training prompts.

    python probe_multi_generation.py --model <dir> --data <parquet> --n 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contact_rewards as cr  # noqa: E402
import section_rewards as sr  # noqa: E402

MULTI_TOKEN, MULTI_ID = "<contacts-v1.multi>", 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16, help="prompts")
    ap.add_argument("--samples", type=int, default=4, help="rollouts per prompt")
    ap.add_argument("--max-gen", type=int, default=7000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = pq.read_table(a.data).to_pylist()[: a.n]
    llm = LLM(model=a.model, max_model_len=a.max_model_len, tensor_parallel_size=1,
              max_num_seqs=64, trust_remote_code=False, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    got = tok.convert_ids_to_tokens([MULTI_ID])[0]
    if got != MULTI_TOKEN:
        raise SystemExit(f"FATAL: id {MULTI_ID} is {got!r}, not {MULTI_TOKEN!r}")
    print(f"[probe] tokenizer ok: id {MULTI_ID} = {got}, vocab {len(tok)}")

    prompts = [r["prompt"][0]["content"] for r in rows]
    extras = [json.loads(r["extras"]) for r in rows]
    params = SamplingParams(temperature=1.0, top_p=0.95, top_k=-1,
                            max_tokens=a.max_gen, n=a.samples, skip_special_tokens=False)
    outs = llm.generate(prompts, params, use_tqdm=False)

    stats = []
    for out, ex in zip(outs, extras):
        gt = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in ex["gt_contacts"]}
        gt = {p for p in gt if cr.in_band(p)}
        pos_to_seq = {int(p): i for i, p in enumerate(ex["seq_positions"])}
        for comp in out.outputs:
            ids = list(comp.token_ids)
            walk = sr.walk_rollout(ids, pos_to_seq, gt)
            consensus, marg = sr.consensus_and_marginals(walk.sections, gt, int(ex["L"]))
            f1s = sr.section_f1s(walk.sections, gt)
            stats.append(dict(
                L=int(ex["L"]), n_tok=len(ids), oov=int(max(ids, default=0) >= 2845),
                sections=walk.n_sections, union=walk.diagnostics["union_pairs"],
                votes=walk.diagnostics["total_votes"],
                jaccard=walk.diagnostics["mean_jaccard"],
                finished=int(walk.finished), consensus=consensus,
                best_f1=max(f1s) if f1s else 0.0, last_f1=f1s[-1] if f1s else 0.0,
                marg_std=float(marg.std()), dead=int(marg.std() == 0.0),
                precision=(walk.n_correct / walk.n_scored) if walk.n_scored else 0.0,
            ))

    keys = ["L", "n_tok", "sections", "union", "votes", "votes_per_pair", "jaccard",
            "finished", "consensus", "best_f1", "last_f1", "marg_std", "dead",
            "precision", "oov"]
    agg = {}
    for k in keys:
        if k == "votes_per_pair":
            agg[k] = float(np.sum([s["votes"] for s in stats]) / max(np.sum([s["union"] for s in stats]), 1))
            continue
        v = np.array([s[k] for s in stats], dtype=np.float64)
        agg[k] = float(np.nanmean(v))
    print(f"\n=== {len(stats)} rollouts from {len(rows)} prompts ===")
    for k in keys:
        print(f"  {k:<16} {agg[k]:.4f}")
    print()
    if agg["oov"] > 0:
        raise SystemExit("FATAL: sampled ids outside the 2845 vocabulary -- the vLLM padding trap")
    if agg["sections"] < 12:
        print("WARNING: fewer than 12 sections per rollout BEFORE any RL. #237's kill "
              "criterion is written against #230's eval-time 22.0; set min_sections from "
              "this number instead, or the gate fires on the harness rather than on the reward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
