#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""M4 — how many contacts does it take to move MarinFold to the other fold?

M1 asks which fold the model reaches on its own. This asks what it costs to make
it reach the other one. contacts-v1 is promptable: conditioning is just extending
the prefix past ``<begin_statements>`` with ``<contact> <pi> <pj>`` statements, so
"how much evidence moves the model" is a number, in units of contacts. AlphaFold
has no analogue of it.

**The dose.** ``k`` contacts drawn from ``B`` (fold2-unique), swept over
``--doses``. The symmetric control draws the same ``k`` from ``A`` instead. The
asymmetry between the two arms at matched ``k`` is the quantity of interest: it
measures how much harder it is to push the model off its preferred fold than to
push it further onto it.

**The reduced task.** Given contacts are removed from the scored universe --
recovery is measured on ``A\\G`` and ``B\\G``, never on ``G`` itself. Scoring the
given set would just read back the prompt (exp163's protocol; without it the
dose-response is circular and rises trivially with ``k``).

**Preregistered, from prior results.** exp254 measured a single true seed at
+0.0124 within protein, so ``k=1`` is expected to do nothing -- that is a known
baseline, not a null result. exp163 moved R-precision 0.145 -> 0.556 conditioning
on true partial *sets*, so the live region is ``k >= 10``. The reported headline
is **k\\***: the smallest dose at which the B-seeded arm's mean fold score turns
negative.

Reuses the sampling, parsing and IO helpers from ``score_foldswitch_worker_cw``
rather than restating them, so the recipe cannot drift between M1 and M4.

    .venv-vllm/bin/python score_conditioning.py \\
        --model /data/exp301/model --targets data/eval_targets.parquet \\
        --out /data/exp301/conditioning --label exp277 --shard 0/1
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pyarrow as pa

from score_foldswitch_worker_cw import (
    BEGIN,
    CONTEXT,
    NUM_POS,
    canonical,
    parse_rollout,
    read_parquet,
    stage_model,
    write_parquet,
)

SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("arm", pa.string()), ("k", pa.int32()),
    ("replicate", pa.int16()), ("rollout", pa.int16()), ("L", pa.int32()),
    ("n_pred", pa.int32()),
    ("n_hit_a_rem", pa.int32()), ("n_hit_b_rem", pa.int32()),
    ("n_a_rem", pa.int32()), ("n_b_rem", pa.int32()),
    ("n_given_echoed", pa.int32()),
    ("finished", pa.bool_()), ("n_tokens", pa.int32()),
])

DEFAULT_DOSES = (0, 1, 2, 5, 10, 20, 40)


def done_pairs(out_dir: str) -> set[str]:
    """Pairs already written under ``out_dir`` (resume support)."""
    import fsspec

    try:
        fs, _ = fsspec.core.url_to_fs(out_dir)
        paths = fs.glob(f"{out_dir}/*.parquet")
    except FileNotFoundError:
        return set()
    seen: set[str] = set()
    for path in paths:
        protocol = out_dir.split("://", 1)[0] if "://" in out_dir else ""
        uri = f"{protocol}://{path}" if protocol and "://" not in path else path
        seen |= set(read_parquet(uri).column("pair_id").to_pylist())
    return seen


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--doses", default=",".join(str(d) for d in DEFAULT_DOSES))
    ap.add_argument("--n-rollouts", type=int, default=50)
    ap.add_argument("--n-replicates", type=int, default=2,
                    help="independent draws of the given set per (pair, arm, k)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))
    out_dir = f"{a.out.rstrip('/')}/{a.label}"
    doses = [int(d) for d in a.doses.split(",")]

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
        residues_from_sequence,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = GenerationConfig()
    recs = [r for r in read_parquet(a.targets).to_pylist() if r["role"] == "foldswitch"]
    recs.sort(key=lambda r: r["L"])
    mine = [r for k, r in enumerate(recs) if k % num_shards == shard_i]
    # Resume: the full dose grid is a multi-hour run, so a pair already written
    # is skipped rather than recomputed. Keyed on pair_id, since each pair's
    # whole grid lands in one file.
    done = done_pairs(out_dir)
    mine = [r for r in mine if r["pair_id"] not in done]
    if a.limit:
        mine = mine[: a.limit]
    print(f"[cond] shard {shard_i}/{num_shards}: {len(mine)} pairs to do "
          f"({len(done)} already written) | doses={doses} "
          f"rollouts={a.n_rollouts} replicates={a.n_replicates}", flush=True)
    if not mine:
        print("[cond] nothing to do")
        return 0

    model_dir = stage_model(a.model, Path("/tmp/marinfold_model"))
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    end_id = tok.convert_tokens_to_ids("<end>")
    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=CONTEXT,
              gpu_memory_utilization=a.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=a.max_num_seqs, seed=a.seed)

    t0 = time.time()
    for n, rec in enumerate(mine, 1):
        pair_id, L = rec["pair_id"], int(rec["L"])
        residues = residues_from_sequence(rec["sequence"])
        set_a, set_b = canonical(rec["contacts_fold1"]), canonical(rec["contacts_fold2"])
        common = {int(p) for p in rec["common_positions"]}
        only_a, only_b = set_a - set_b, set_b - set_a

        prompts, meta, index_maps = [], [], []
        for arm, source in (("seed_b", only_b), ("seed_a", only_a)):
            for k in doses:
                if k > len(source):
                    continue
                for rep in range(1 if k == 0 else a.n_replicates):
                    rng = random.Random(f"{pair_id}:{arm}:{k}:{rep}")
                    given = set(rng.sample(sorted(source), k)) if k else set()
                    a_rem, b_rem = only_a - given, only_b - given
                    for r in range(a.n_rollouts):
                        doc = build_document(f"{pair_id}:{arm}:k{k}:rep{rep}:r{r}",
                                             residues, [], config=cfg)
                        prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
                        pos_of_seq = {t: (doc.n_term_index + t) % NUM_POS for t in range(doc.seq_len)}
                        # Emit the given set the way training documents do:
                        # shuffled, with a random orientation per statement.
                        order = sorted(given)
                        rng.shuffle(order)
                        for i, j in order:
                            x, y = (i, j) if rng.random() < 0.5 else (j, i)
                            prefix += f" <contact> <p{pos_of_seq[x]}> <p{pos_of_seq[y]}>"
                        prompts.append(prefix)
                        index_maps.append({pos_of_seq[t]: t for t in range(doc.seq_len)})
                        meta.append((arm, k, rep, r, given, a_rem, b_rem))

        plen = len(tok(prompts[0], add_special_tokens=False).input_ids)
        max_new = min(CONTEXT - plen, a.contact_mult * L + 128)
        sps = [SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                              max_tokens=max_new, stop_token_ids=[end_id],
                              skip_special_tokens=False, seed=a.seed * 7919 + idx)
               for idx in range(len(prompts))]
        ts = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - ts

        rows = {c.name: [] for c in SCHEMA}
        for (arm, k, rep, r, given, a_rem, b_rem), out, seq_index in zip(meta, outs, index_maps):
            raw = parse_rollout(out.outputs[0].text, seq_index)
            pred = {p for p in raw if p[0] in common and p[1] in common}
            rows["pair_id"].append(pair_id)
            rows["arm"].append(arm)
            rows["k"].append(k)
            rows["replicate"].append(rep)
            rows["rollout"].append(r)
            rows["L"].append(L)
            rows["n_pred"].append(len(pred))
            rows["n_hit_a_rem"].append(len(pred & a_rem))
            rows["n_hit_b_rem"].append(len(pred & b_rem))
            rows["n_a_rem"].append(len(a_rem))
            rows["n_b_rem"].append(len(b_rem))
            # How often the model simply repeats what it was handed. A high
            # echo rate with no movement on the remainder means the prompt was
            # copied, not used.
            rows["n_given_echoed"].append(len(pred & given))
            rows["finished"].append(out.outputs[0].finish_reason == "stop")
            rows["n_tokens"].append(len(out.outputs[0].token_ids))

        stem = f"shard-{shard_i:03d}-of-{num_shards:03d}-{pair_id}.parquet"
        write_parquet(pa.table(rows, schema=SCHEMA), f"{out_dir}/{stem}")

        def arm_phi(arm: str, k: int) -> float:
            sel = [t for t in range(len(rows["arm"])) if rows["arm"][t] == arm and rows["k"][t] == k]
            if not sel:
                return float("nan")
            ra = sum(rows["n_hit_a_rem"][t] / max(rows["n_a_rem"][t], 1) for t in sel) / len(sel)
            rb = sum(rows["n_hit_b_rem"][t] / max(rows["n_b_rem"][t], 1) for t in sel) / len(sel)
            return ra - rb

        trail = "  ".join(f"k{k}:{arm_phi('seed_b', k):+.3f}" for k in doses if k <= len(only_b))
        print(f"[cond] [{n}/{len(mine)}] {pair_id:16s} L={L:4d} |A|={len(only_a):4d} "
              f"|B|={len(only_b):4d} seed_b phi {trail}  {dt:6.1f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    print(f"[cond] DONE shard {shard_i}/{num_shards} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
