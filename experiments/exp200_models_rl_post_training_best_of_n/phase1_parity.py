# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 gate: does exp200's RL sampling path reproduce exp163's numbers?

Everything downstream of this is worthless if the answer is no. exp200 generates
through ``ContactsV1RLEnv`` -> ``inference_ctx.llm.generate``, and scores by
walking TOKEN IDS; exp163 generated through its own worker and scored by regexing
DECODED TEXT. Those are two independent implementations of the same measurement,
so running both over the same rollouts is a real check rather than a smoke test.

Three things are compared:

1. **Per-rollout agreement.** For every generation, exp200's ``dense_rewards``
   section F1s vs exp163's ``parse_sections`` + ``score_rollout``. These should
   agree to floating point, not "closely".
2. **Headline numbers** against #163 §4.1, measured on the same eval set: arm F
   best 0.3025 / last 0.2493 / first 0.1840, ~15 sections uncapped, Jaccard
   0.071. Note the published figures are UNCAPPED; run with ``--max-sections 0``
   to disable the cap when comparing to them directly.
3. **Prompt invariants.** Sentinel is id 7 at index 0, the rest of the prompt is
   untouched, and the prompt ends on ``<begin_statements>``.

Usage on a v5p-8 (see ``dispatch_parity.py``)::

    python phase1_parity.py --model gs://.../tpuF-bf16/step-404 \\
        --targets gs://.../eval554/targets.parquet \\
        --prompts gs://.../eval554/prompts \\
        --out gs://.../exp200/phase1 --limit 100 --n-generations 4
"""

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import time

import fsspec
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.rl.decoding import DecodingConfig
from marin.rl.environments.inference_ctx import (
    VLLMEngineConfig,
    VLLMFallbackSamplingConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)

import _exp163_rollout_metrics as rm
import contact_rewards as cr
from contacts_env import ContactsV1RLEnv

logger = logging.getLogger(__name__)

# exp163 WRITEUP §4.1, arm F, 553 proteins x 4 rollouts, UNCAPPED sections.
PUBLISHED = {"best_f1": 0.3025, "last_f1": 0.2493, "first_f1": 0.1840,
             "n_sections": 14.99, "mean_jaccard": 0.071}


def stage(uri: str) -> str:
    """Copy a remote model dir to local disk; vLLM needs local files."""
    if "://" not in uri:
        return uri
    dest = tempfile.mkdtemp(prefix="exp200-model-")
    fs, _ = fsspec.core.url_to_fs(uri)
    t0 = time.time()
    for path in fs.ls(uri.rstrip("/"), detail=False):
        name = path.rsplit("/", 1)[-1]
        if not name:
            continue
        with fs.open(path, "rb") as src, open(os.path.join(dest, name), "wb") as dst:
            shutil.copyfileobj(src, dst, length=32 << 20)
    logger.info("[parity] staged %s -> %s in %.1fs", uri, dest, time.time() - t0)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100, help="proteins to sample")
    ap.add_argument("--n-generations", type=int, default=4)
    ap.add_argument("--max-sections", type=int, default=0,
                    help="0 disables the cap, matching the published uncapped numbers")
    ap.add_argument("--section-contacts", type=int, default=220)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 0 means "no cap"; the env wants a number, so use a bound nothing reaches.
    max_sections = a.max_sections if a.max_sections > 0 else 64
    local_model = stage(a.model)

    ctx = vLLMInferenceContext(
        inference_config=vLLMInferenceContextConfig(
            engine=VLLMEngineConfig(
                model_name=local_model,
                # Substring-matched to pick a renderer this path never reaches.
                canonical_model_name="qwen3-1_5b-contacts-v1-multi",
                max_model_len=a.max_model_len,
                tensor_parallel_size=a.tensor_parallel_size,
                gpu_memory_utilization=0.90,
                seed=a.seed,
            ),
            fallback_sampling=VLLMFallbackSamplingConfig(top_k=None, stop_strings=None),
        )
    )

    env = ContactsV1RLEnv(
        targets_path=a.targets,
        prompts_path=a.prompts,
        mode="multi",
        max_sections=max_sections,
        section_contacts=a.section_contacts,
        max_model_len=a.max_model_len,
        # Every protein is fair game: this is a measurement, not training.
        eval_fraction=0.0,
        initial_precision=0.30,
        limit=a.limit,
        seed=a.seed,
    )

    decoding = DecodingConfig(
        temperature=a.temperature,
        top_p=a.top_p,
        # None, not -1: DecodingConfig rejects a non-positive top_k and the env
        # translates None to vLLM's -1, which is exp163's sampling regime.
        top_k=None,
        max_output_tokens=(3 * a.section_contacts + 8) * max_sections,
        stop_token_ids=[cr.END_ID],
        seed=None,
    )

    t0 = time.time()
    groups, metrics = env.sample(
        ctx,
        n_examples=a.limit,
        n_generations=a.n_generations,
        decoding=decoding,
        prng_key=jax.random.PRNGKey(a.seed),
        mode="train",
    )
    elapsed = time.time() - t0
    rollouts = [r for g in groups for r in g.rollouts]
    logger.info("[parity] %d rollouts from %d groups in %.1fs", len(rollouts), len(groups), elapsed)

    tokenizer = ctx.tokenizer
    rows = []
    n_prompt_bad = 0
    for rollout in rollouts:
        entry_id, _, realization = rollout.env_example_id.rpartition(":")
        target = env._targets[entry_id]
        prompts = {p["r"]: p for p in env._prompts_for(entry_id)}
        row_prompt = prompts[int(realization.lstrip("r"))]
        pos_to_seq = {int(p): i for i, p in enumerate(row_prompt["seq_positions"])}

        # exp200's measurement, recomputed from the stored response ids.
        mine = cr.dense_rewards(
            list(rollout.response_tokens), pos_to_seq, target["gt"],
            mode="multi", precision_baseline=0.30, max_sections=max_sections,
        )

        # exp163's measurement, independently, from decoded text.
        text = tokenizer.decode(list(rollout.response_tokens), skip_special_tokens=False)
        gtb = rm.gt_by_band(target["gt"])
        theirs = [rm.score_rollout(p, gtb)["all_f1"]
                  for p in rm.parse_sections(text, pos_to_seq)][:max_sections]

        prompt_ids = list(rollout.prompt_tokens)
        prompt_ok = prompt_ids[0] == cr.MULTI_DOC_ID and prompt_ids[-1] == cr.BEGIN_STATEMENTS_ID
        n_prompt_bad += not prompt_ok

        mine_f1 = [v for v in mine.section_f1 if not math.isnan(v)]
        theirs_f1 = [v for v in theirs if not math.isnan(v)]
        n_common = min(len(mine_f1), len(theirs_f1))
        max_delta = max((abs(x - y) for x, y in zip(mine_f1[:n_common], theirs_f1[:n_common])), default=0.0)

        rows.append(
            dict(
                entry_id=entry_id, r=int(realization.lstrip("r")),
                L=target["L"], n_gt=len(target["gt"]),
                n_gen_tokens=len(rollout.response_tokens),
                finished=not rollout.is_truncated,
                prompt_ok=prompt_ok,
                n_sections_mine=len(mine_f1), n_sections_theirs=len(theirs_f1),
                best_f1_mine=mine.episode_reward,
                best_f1_theirs=max(theirs_f1) if theirs_f1 else 0.0,
                last_f1_mine=mine.diagnostics["last_f1"],
                last_f1_theirs=theirs_f1[-1] if theirs_f1 else float("nan"),
                first_f1_mine=mine.diagnostics["first_f1"],
                first_f1_theirs=theirs_f1[0] if theirs_f1 else float("nan"),
                mean_jaccard=mine.diagnostics["mean_jaccard"],
                precision=mine.diagnostics["precision"],
                n_pred=mine.diagnostics["n_pred"],
                max_section_f1_delta=max_delta,
            )
        )

    def mean(key):
        values = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(values)) if values else float("nan")

    best_delta = max(abs(r["best_f1_mine"] - r["best_f1_theirs"]) for r in rows)
    section_delta = max(r["max_section_f1_delta"] for r in rows)
    n_section_mismatch = sum(r["n_sections_mine"] != r["n_sections_theirs"] for r in rows)

    print("\n" + "=" * 72)
    print("PARITY vs exp163 (independent implementations of the same measurement)")
    print("=" * 72)
    print(f"  rollouts                     {len(rows)}")
    print(f"  max |best_f1 delta|          {best_delta:.3e}")
    print(f"  max |section_f1 delta|       {section_delta:.3e}")
    print(f"  section-count mismatches     {n_section_mismatch}/{len(rows)}")
    print(f"  malformed prompts            {n_prompt_bad}/{len(rows)}")
    print("\n" + "=" * 72)
    print(f"HEADLINE vs #163 §4.1 (published on 553x4, uncapped; here {a.limit}x{a.n_generations})")
    print("=" * 72)
    print(f"  {'metric':16s} {'exp200':>10s} {'exp163':>10s} {'delta':>10s}")
    for key, published in PUBLISHED.items():
        observed = mean(f"{key}_mine") if f"{key}_mine" in rows[0] else mean(key)
        print(f"  {key:16s} {observed:10.4f} {published:10.4f} {observed - published:+10.4f}")
    print(f"  {'frac_finished':16s} {mean('finished'):10.4f}")
    print(f"  {'n_gen_tokens':16s} {mean('n_gen_tokens'):10.1f}")
    print(f"  {'precision':16s} {mean('precision'):10.4f}")
    print(f"  {'n_pred':16s} {mean('n_pred'):10.1f}")
    print("=" * 72 + "\n")

    out = a.out.rstrip("/")
    with fsspec.open(f"{out}/parity_rollouts.parquet", "wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows), fh)
    summary = {
        "n_rollouts": len(rows), "n_groups": len(groups), "elapsed_s": elapsed,
        "max_best_f1_delta": best_delta, "max_section_f1_delta": section_delta,
        "n_section_mismatch": n_section_mismatch, "n_prompt_bad": n_prompt_bad,
        "observed": {k: mean(f"{k}_mine") if f"{k}_mine" in rows[0] else mean(k) for k in PUBLISHED},
        "published": PUBLISHED,
        "env_metrics": metrics,
        "args": vars(a),
    }
    with fsspec.open(f"{out}/parity_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info("[parity] wrote %s/parity_{rollouts.parquet,summary.json}", out)

    # Agreement is the gate. The headline numbers are informative but noisy at
    # this sample size (#163: a 40-protein probe read +0.048 where 553 read +0.065).
    if best_delta > 1e-6 or section_delta > 1e-6 or n_prompt_bad:
        logger.error("[parity] FAILED: the two implementations disagree")
        return 1
    logger.info("[parity] PASSED: implementations agree to floating point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
