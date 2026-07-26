# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a modest backtracking corpus for a first #160 training signal.

Runs the engine + exp120 adapter over many proteins (one document each) from
exp98's ``targets.parquet``, writing one JSON per document so the run is
**resumable** (re-invoking skips entries already done). ``--aggregate`` folds
the per-doc JSONs into a single parquet + prints corpus-level statistics — the
across-many-proteins version of the pilot's go/no-go (FP-enrichment,
retraction rate, delay spread), plus the hard invariant (every doc folds→GT).

This is deliberately small-scale (the unbatched loop is ~13 s/doc); it exists
to validate the full pipeline end-to-end, produce real corpus statistics, and
give #160 something to train on — not to be the final 4.2M-doc corpus.

Run from the exp dir::

    uv run python gen_corpus.py --n 1000 --min-l 30 --max-l 130
    uv run python gen_corpus.py --aggregate
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import pandas as pd

from run_pilot import (
    _fix_tokenizer_config,
    _read_targets,
    _TARGETS,
    gt_from_row,
    run_one,
)

from backtrack_engine import RetractionPolicy
from marinfold import load_backend
from marinfold.registry import resolve_model


def _corpus_dir(out: str) -> str:
    d = os.path.join(out, "corpus")
    os.makedirs(d, exist_ok=True)
    return d


def aggregate(out: str) -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(_corpus_dir(out), "*.json"))):
        with open(p) as fh:
            rows.append(json.load(fh))
    if not rows:
        print("no documents to aggregate")
        return
    df = pd.DataFrame(rows)
    parquet = os.path.join(out, "backtracking_corpus.parquet")
    df.to_parquet(parquet, index=False)
    n = len(df)
    fp_total = int(df["n_fp_emitted"].sum())
    fp_trig = int(df["fp_retracted_by_trigger"].sum())
    tp_trig = int(df["tp_retracted_by_trigger"].sum())
    print(f"wrote {parquet}  ({n} docs, {df['doc_tokens'].sum():,} tokens)")
    print("=== corpus stats ===")
    print(f"all fold_to_gt: {bool(df['folds_to_gt'].all())}   "
          f"truncated: {int(df['truncated'].sum())}/{n}")
    print(f"mean retract/doc: {df['n_retract_stmts'].mean():.1f}   "
          f"mean contacts/doc: {df['n_contact_stmts'].mean():.1f}")
    print(f"FP emitted: {fp_total}   caught by trigger: {fp_trig} "
          f"({fp_trig / fp_total:.1%})   trigger false alarms (TP): {tp_trig}")
    print(f"mean trigger delay: {df['mean_trigger_delay'].mean():.1f} statements")
    print(f"wall/doc: mean {df['wall_s'].mean():.1f}s  total {df['wall_s'].sum()/3600:.1f}h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", action="store_true",
                    help="fold per-doc JSONs into a parquet + print stats, then exit")
    ap.add_argument("--model", default="contacts-v1-exp120-1.5B")
    ap.add_argument("--targets", default=_TARGETS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-l", type=int, default=30)
    ap.add_argument("--max-l", type=int, default=130)
    ap.add_argument("--min-sep", type=int, default=6)
    ap.add_argument("--eval-cadence", type=int, default=3)
    ap.add_argument("--min-delay", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--s-floor", type=float, default=1e-3)
    ap.add_argument("--noise-prob", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.out)
        return

    policy = RetractionPolicy(
        min_delay=args.min_delay, eval_cadence=args.eval_cadence,
        tau=args.tau, s_floor=args.s_floor, noise_retract_prob=args.noise_prob,
    )
    corpus = _corpus_dir(args.out)

    print(f"resolving model {args.model} ...", flush=True)
    model_path = resolve_model(args.model)
    _fix_tokenizer_config(model_path)
    backend = load_backend("transformers", model=str(model_path), dtype="bfloat16")

    df = _read_targets(args.targets)
    df = df[(df["L"] >= args.min_l) & (df["L"] <= args.max_l)].head(args.n)
    print(f"{len(df)} candidate targets in L∈[{args.min_l},{args.max_l}]", flush=True)

    done = 0
    for _, row in df.iterrows():
        entry_id = str(row["entry_id"])
        out_json = os.path.join(corpus, f"{entry_id}.json")
        if os.path.exists(out_json):        # resume
            done += 1
            continue
        gt = gt_from_row(row, args.min_sep)
        if not gt:
            continue
        t0 = time.time()
        doc, m = run_one(backend, entry_id, row["sequence"], gt, policy, args.min_sep)
        m["document"] = doc
        m["sha1"] = hashlib.sha1(doc.encode()).hexdigest()
        # atomic-ish write (temp then rename) so a kill mid-write can't corrupt.
        tmp = out_json + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(m, fh)
        os.replace(tmp, out_json)
        done += 1
        print(f"[{done}] {entry_id} L={m['L']} retract={m['n_retract_stmts']} "
              f"fp_trig={m['fp_retracted_by_trigger']}/{m['n_fp_emitted']} "
              f"folds={m['folds_to_gt']} {time.time() - t0:.1f}s", flush=True)

    print(f"done: {done} documents in {corpus}", flush=True)


if __name__ == "__main__":
    main()
