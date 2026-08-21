# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 -- N rollouts per eval protein, in one of two arms.

``--arm iid``
    exp82's published recipe, unchanged: N sampled contacts-v1 completions per
    protein, each continuing a *fresh* document realization (resampled
    N-terminus and ``<pX> <AA>`` statement order). This is the control, and its
    consensus number is the gate: it has to reproduce #245's published m2-p06
    eval-val R-precision.

``--arm seeded``
    the same N realizations -- realization *r* is byte-identical between the two
    arms, so the arms are paired and the seeding is the only difference -- but
    the structure section of rollout *r* is pre-filled with the *r*-th ranked
    pairwise contact from phase 1, written in that realization's position tokens
    with a coin-flipped orientation.

Output is the **per-rollout, order-preserving contact list**, not the aggregated
vote matrix: the votes are a sum over it (``build_metrics.py`` rebuilds them),
but the oracle best-of-N readout needs the individual rollouts and cannot be
recovered from the sum. ``is_seed`` marks the pre-filled statement, so consensus
can be scored both with and without the seed voting for itself.

Sampling knobs are exp82's and are not meant to be moved: temperature 1.0,
top-p 0.95, top-k disabled, budget ``6L + 128`` tokens.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import EXPECTED_UNITS, load_targets, parse_rollout, realization, seed_statement

DETAIL_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rollout", pa.int16()), ("rank", pa.int16()),
    ("i", pa.int16()), ("j", pa.int16()), ("is_seed", pa.bool_()),
])


def done_stems(out_dir: Path) -> tuple[set[str], int]:
    """Stems already written, and the part number to resume writing from.

    Restarting the part counter at zero on a resume would clobber the earlier
    parts whose stems are being skipped, silently dropping those proteins from
    the output -- exp169 lost two proteins to exactly that.
    """
    parts = sorted(out_dir.glob("detail-part-*.parquet"))
    seen: set[str] = set()
    for path in parts:
        table = pq.read_table(path, columns=["dataset", "stem"])
        seen |= {
            f"{d}__{s}" for d, s in zip(table.column("dataset").to_pylist(),
                                        table.column("stem").to_pylist())
        }
    return seen, len(parts)


def load_seeds(path: Path) -> dict[str, list[tuple[int, int]]]:
    """``{dataset__stem: [(i, j), ...]}`` in pairwise-rank order."""
    df = pd.read_parquet(path).sort_values(["dataset", "stem", "rank"])
    seeds: dict[str, list[tuple[int, int]]] = {}
    for (dataset, stem), group in df.groupby(["dataset", "stem"], sort=False):
        seeds[f"{dataset}__{stem}"] = list(
            zip(group["i"].astype(int), group["j"].astype(int))
        )
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=("iid", "seeded"), required=True)
    ap.add_argument("--seeds", type=Path, default=None,
                    help="phase 1 seeds.parquet (required for --arm seeded)")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.arm == "seeded" and args.seeds is None:
        ap.error("--arm seeded requires --seeds")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from marinfold.document_structures.contacts_v1 import residues_from_sequence

    targets = load_targets()
    assert len(targets) == EXPECTED_UNITS, (
        f"expected {EXPECTED_UNITS} eval-val units, got {len(targets)}"
    )
    if args.limit:
        targets = targets[: args.limit]

    arm_dir = args.out / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    skip, part = done_stems(arm_dir)
    todo = [t for t in targets if t.key not in skip]

    seeds = load_seeds(args.seeds) if args.arm == "seeded" else {}
    if args.arm == "seeded":
        for target in todo:
            available = len(seeds.get(target.key, ()))
            assert available >= args.n_rollouts, (
                f"{target.key}: only {available} pairwise seeds for "
                f"{args.n_rollouts} rollouts"
            )

    print(f"[{args.arm}] {len(targets)} targets, {len(skip)} already done, "
          f"{len(todo)} to do | n_rollouts={args.n_rollouts} T={args.temperature} "
          f"top_p={args.top_p} top_k={args.top_k} budget={args.contact_mult}L+128",
          flush=True)
    if not todo:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    # generation_config="vllm": ignore any generation defaults baked into the
    # export (some carry top_k=50), so the recipe comes only from SamplingParams.
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=args.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=args.max_num_seqs, seed=args.seed)

    unfinished_rows: list[dict] = []
    t0, n_unfinished, n_total = time.time(), 0, 0
    for start in range(0, len(todo), args.chunk):
        group = todo[start:start + args.chunk]
        prompts, params, per = [], [], []
        for target in group:
            residues = residues_from_sequence(target.input_seq)
            first, maps, seeded_pairs = len(prompts), [], []
            for r in range(args.n_rollouts):
                prefix, seq_positions = realization(target.stem, residues, f"r{r}")
                pair = None
                if args.arm == "seeded":
                    i, j = seeds[target.key][r]
                    # A per-(protein, rollout) RNG so the orientation coin flip
                    # is reproducible and independent of iteration order.
                    rng = random.Random(f"{target.key}:{r}:{args.seed}")
                    prefix += seed_statement(seq_positions[i], seq_positions[j], rng)
                    pair = (i, j)
                prompts.append(prefix)
                maps.append({pos: k for k, pos in enumerate(seq_positions)})
                seeded_pairs.append(pair)
            prompt_tokens = len(
                tokenizer(prompts[first], add_special_tokens=False).input_ids
            )
            max_new = min(8192 - prompt_tokens, args.contact_mult * target.L + 128)
            per.append((target, first, maps, seeded_pairs))
            params += [
                SamplingParams(temperature=args.temperature, top_p=args.top_p,
                               top_k=args.top_k, max_tokens=max_new,
                               stop_token_ids=[end_id], skip_special_tokens=False,
                               seed=args.seed * 1_000_003 + first + r)
                for r in range(args.n_rollouts)
            ]

        gen_start = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        gen_seconds = time.time() - gen_start

        rows = {c: [] for c in
                ("dataset", "stem", "L", "rollout", "rank", "i", "j", "is_seed")}
        for target, first, maps, seeded_pairs in per:
            mine = outputs[first:first + args.n_rollouts]
            unfinished = sum(1 for o in mine if o.outputs[0].finish_reason != "stop")
            n_unfinished += unfinished
            n_total += args.n_rollouts
            if unfinished:
                unfinished_rows.append(dict(dataset=target.dataset, stem=target.stem,
                                            L=target.L, unfinished=unfinished,
                                            n_rollouts=args.n_rollouts))
            for r, (output, pos_to_seq, pair) in enumerate(zip(mine, maps, seeded_pairs)):
                emitted = parse_rollout(output.outputs[0].text, pos_to_seq)
                # The seed is part of the document the model wrote, so it leads
                # the rollout's own ranking; `is_seed` keeps it separable.
                pairs = emitted if pair is None else [pair] + [p for p in emitted if p != pair]
                for rank, (i, j) in enumerate(pairs):
                    rows["dataset"].append(target.dataset)
                    rows["stem"].append(target.stem)
                    rows["L"].append(target.L)
                    rows["rollout"].append(r)
                    rows["rank"].append(rank)
                    rows["i"].append(i)
                    rows["j"].append(j)
                    rows["is_seed"].append(pair is not None and (i, j) == pair)

        dest = arm_dir / f"detail-part-{part:04d}.parquet"
        pq.write_table(pa.table(rows, schema=DETAIL_SCHEMA), dest, compression="zstd")
        part += 1
        n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"[{args.arm}] [{start + len(group)}/{len(todo)}] "
              f"L={group[0].L}-{group[-1].L} {gen_seconds:6.1f}s "
              f"{n_tokens / gen_seconds:7.0f} tok/s "
              f"unfinished={n_unfinished}/{n_total} -> {dest.name} "
              f"(elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    report = arm_dir / "unfinished.csv"
    with report.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "stem", "L",
                                                "unfinished", "n_rollouts"])
        writer.writeheader()
        writer.writerows(unfinished_rows)
    print(f"[{args.arm}] DONE {len(todo)} proteins in {(time.time() - t0) / 60:.1f} min "
          f"| unfinished {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%) -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
