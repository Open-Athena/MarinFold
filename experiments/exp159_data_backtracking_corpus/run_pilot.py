# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pilot: run the backtracking engine with the real exp120 model on a few proteins.

Two gating numbers come out of this (issue #159):

1. **Per-document wall-clock** — is model-in-the-loop synthesis affordable?
   (Re-conditioning re-prefills the growing prompt each step; no KV reuse yet.)
2. **FP-enrichment of trigger retractions** — does the posterior-collapse
   trigger actually flag the model's *false positives* (go/no-go), or does it
   fire indiscriminately / leave everything to the correctness flush?

GT contacts + sequences come from exp98's published ``targets.parquet`` (so no
pyconfind needed). Every synthesised document is validated: its rendered form
must fold (``read.live_contacts``) to exactly GT.

Run from the marinfold/ dir (for the installed env + model registry)::

    uv run python ../experiments/exp159_data_backtracking_corpus/run_pilot.py \
        --n 3 --max-l 90 --out ../experiments/exp159_data_backtracking_corpus/data
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd  # noqa: E402

from backtrack_adapter import ModelAdapter  # noqa: E402
from backtrack_engine import (  # noqa: E402
    RetractionPolicy,
    build_backtracking_structure,
    canon,
)

from marinfold import load_backend  # noqa: E402
from marinfold.document_structures.contacts_v1 import inference as inf  # noqa: E402
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH  # noqa: E402
from marinfold.registry import resolve_model  # noqa: E402

_TARGETS = (
    "hf://buckets/open-athena/MarinFold/"
    "data/contacts-v1-train-rollouts-exp98/targets.parquet"
)
_TRIGGER = {"collapse", "floor", "rank"}


def _fix_tokenizer_config(model_path) -> None:
    """Relabel the checkpoint's marinfold-custom tokenizer_class for AutoTokenizer.

    exp120's export writes ``tokenizer_class: "TokenizersBackend"`` (a marinfold
    class ``AutoTokenizer`` can't resolve), though its ``tokenizer.json`` is a
    plain WordLevel tokenizer. Relabel it to ``PreTrainedTokenizerFast`` so the
    transformers backend loads it. Done after every ``resolve_model`` because
    that call re-mirrors the bucket and would overwrite an earlier edit.
    (Flagged upstream — the marinfold transformers backend should load
    tokenizer.json directly; #160's eval path will hit the same thing.)
    """
    import json

    p = os.path.join(str(model_path), "tokenizer_config.json")
    if not os.path.exists(p):
        return
    with open(p) as fh:
        cfg = json.load(fh)
    if cfg.get("tokenizer_class") not in (None, "PreTrainedTokenizerFast"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(p, "w") as fh:
            json.dump(cfg, fh, indent=2)


def _read_targets(url: str):
    """Read the targets parquet. Bucket ``hf://buckets/...`` paths (which
    pyarrow/fsspec don't route) are fetched via their HF resolve URL."""
    if not url.startswith("hf://buckets/"):
        return pd.read_parquet(url)
    import tempfile

    import requests
    from huggingface_hub import get_token

    org, repo, path = url[len("hf://buckets/"):].split("/", 2)
    resolve = f"https://huggingface.co/buckets/{org}/{repo}/resolve/{path}"
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(resolve, headers=headers, timeout=180)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(r.content)
    tmp.close()
    return pd.read_parquet(tmp.name)


def gt_from_row(row, min_sep: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        canon(int(i), int(j))
        for i, j in row["gt_contacts"]
        if abs(int(j) - int(i)) >= min_sep
    )


def run_one(backend, entry_id, sequence, gt_seq, policy, min_sep):
    structure = inf.structure_from_sequence(sequence, entry_id=entry_id)
    adapter = ModelAdapter(
        backend, structure, entry_id=entry_id, temperature=1.0, top_p=0.95, top_k=50
    )
    # Statement budget from the token budget: prefix + 3·stmts + <end> <= ctx.
    avail = CONTEXT_LENGTH - len(adapter.prefix_ids) - 1
    max_statements = max(len(gt_seq) + 2, avail // 3)

    t0 = time.time()
    res = build_backtracking_structure(
        gt_seq, adapter, adapter, policy,
        max_statements=max_statements, rng=random.Random(0),
    )
    wall = time.time() - t0
    doc = adapter.assemble_document(res.statements)

    # FP / trigger accounting.
    fp_trigger = sum(1 for _, _, was_true, t in res.retractions
                     if not was_true and t in _TRIGGER)
    fp_flush = sum(1 for _, _, was_true, t in res.retractions
                   if not was_true and t == "flush")
    tp_trigger = sum(1 for _, _, was_true, t in res.retractions
                     if was_true and t in _TRIGGER)
    n_fp = fp_trigger + fp_flush
    trigger_delays = [d for _, d, was_true, t in res.retractions
                      if not was_true and t in _TRIGGER]

    return doc, {
        "entry_id": entry_id,
        "L": adapter.L,
        "n_gt": len(gt_seq),
        "wall_s": round(wall, 2),
        "propose_calls": adapter.n_propose_calls,
        "score_calls": adapter.n_score_calls,
        "n_contact_stmts": res.n_contact_statements,
        "n_retract_stmts": res.n_retract_statements,
        "n_reemit": res.n_reemit,
        "n_fp_emitted": n_fp,
        "fp_retracted_by_trigger": fp_trigger,
        "fp_retracted_at_flush": fp_flush,
        "tp_retracted_by_trigger": tp_trigger,   # trigger false alarms (want ~0)
        "trigger_recall": round(fp_trigger / n_fp, 3) if n_fp else float("nan"),
        "mean_trigger_delay": (round(sum(trigger_delays) / len(trigger_delays), 1)
                               if trigger_delays else float("nan")),
        "doc_tokens": len(doc.split()),
        "correct": res.correct,
        "folds_to_gt": adapter.document_folds_to_gt(doc, gt_seq),
        "truncated": res.truncated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="contacts-v1-exp120-1.5B")
    ap.add_argument("--targets", default=_TARGETS)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min-l", type=int, default=40)
    ap.add_argument("--max-l", type=int, default=90)
    ap.add_argument("--min-sep", type=int, default=6)
    ap.add_argument("--eval-cadence", type=int, default=3)
    ap.add_argument("--min-delay", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--s-floor", type=float, default=1e-3)
    ap.add_argument("--noise-prob", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()

    policy = RetractionPolicy(
        min_delay=args.min_delay, eval_cadence=args.eval_cadence,
        tau=args.tau, s_floor=args.s_floor, noise_retract_prob=args.noise_prob,
    )

    print(f"resolving model {args.model} ...", flush=True)
    model_path = resolve_model(args.model)
    _fix_tokenizer_config(model_path)
    print(f"loading backend from {model_path}", flush=True)
    backend = load_backend("transformers", model=str(model_path), dtype="bfloat16")

    print(f"reading targets {args.targets}", flush=True)
    df = _read_targets(args.targets)
    df = df[(df["L"] >= args.min_l) & (df["L"] <= args.max_l)].head(args.n)
    print(f"{len(df)} targets in L∈[{args.min_l},{args.max_l}]", flush=True)

    os.makedirs(args.out, exist_ok=True)
    docs_dir = os.path.join(args.out, "pilot_docs")
    os.makedirs(docs_dir, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        gt = gt_from_row(row, args.min_sep)
        if not gt:
            continue
        print(f"  {row['entry_id']} (L={row['L']}, n_gt={len(gt)}) ...", flush=True)
        doc, m = run_one(backend, str(row["entry_id"]), row["sequence"], gt, policy,
                         args.min_sep)
        rows.append(m)
        with open(os.path.join(docs_dir, f"{row['entry_id']}.txt"), "w") as fh:
            fh.write(doc + "\n")
        print(f"    wall={m['wall_s']}s  fp={m['n_fp_emitted']} "
              f"trigger_recall={m['trigger_recall']} folds_to_gt={m['folds_to_gt']}",
              flush=True)

    out_csv = os.path.join(args.out, "pilot_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)
    if rows:
        mdf = pd.DataFrame(rows)
        print("\n=== pilot summary ===")
        print(f"docs: {len(mdf)}   all fold_to_gt: {bool(mdf['folds_to_gt'].all())}"
              f"   any truncated: {bool(mdf['truncated'].any())}")
        print(f"wall/doc: mean {mdf['wall_s'].mean():.1f}s  max {mdf['wall_s'].max():.1f}s")
        print(f"FP emitted total: {int(mdf['n_fp_emitted'].sum())}  "
              f"caught by trigger: {int(mdf['fp_retracted_by_trigger'].sum())}  "
              f"trigger false alarms (TP): {int(mdf['tp_retracted_by_trigger'].sum())}")


if __name__ == "__main__":
    main()
