# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 -- tokenize + greedily pack exp230's corpus, carrying the loss weights.

Turns ``build_corpus.py``'s parquet (one ``document`` text column plus ``kind``)
into the **prebuilt** cache format levanter needs to train on per-token weights:

    input_ids     int32[seq_len]
    loss_weights  float32[seq_len]

**Why this step exists.** levanter 1.2 removed ``DatasetComponent.loss_weight_fn``,
so the only surviving way to train on per-token weights is
``PrebuiltLmDatasetFormat(input_ids_key=..., loss_weights_key=...)``, which reads
them out of the cache.  The mask has to be materialised here.  (#163 established
this; ``_loss_mask.py`` is its profile code, vendored verbatim.)

**Packing is ours too.**  ``PrebuiltLmDataset`` maps one cache row to one training
example and has no packing mode, so this does the greedy packing
``PackedTokenDataset`` would otherwise do: fill a ``--seq-len`` row with whole
documents -- never split one, a partial protein document is nonsense -- and pad
the tail at weight 0.  Cross-document attention is still blocked at train time:
``PrebuiltLmDataset`` derives segment ids from the ``<eos>`` appended after each
document.

Two weight decisions, both #163's and both load-bearing:

* **Profile F on multi documents** -- ``w_header 0.1 / w_draft 1.0 / w_final 1.0``.
  ``weight[i]`` supervises predicting ``token[i+1]``, so the slot that teaches
  *"emit another ``<begin_statements>``"* is the last token of a **draft**, and the
  slot that teaches *"emit ``<end>``"* is the last token of the **final**.  With
  ``w_draft`` at 0 or 0.1 the continue transition is supervised 10x more weakly
  than stopping and the model emits exactly one section; F is where continuing
  competes with stopping.  #163 swept this and E (2.0 final) bought nothing.
* **Plain documents weighted 1.0 throughout** -- the rehearsal half trains the
  ordinary contacts-v1 objective, header included, exactly as exp199 was
  pretrained.

And one that is easy to get wrong: **the ``<eos>`` slot is explicitly zeroed.**
The weight *on* a document's ``<eos>`` supervises the first token of the *next*
document in the packed row -- cross-document leakage that is invisible whenever
``w_header`` is 0, and ``w_header`` is 0.1 here.

    uv run python tokenize_corpus.py --in /data/exp230_multi/corpus \\
        --out /data/exp230_multi/tokenized --tokenizer /data/exp230_multi/tokenizer_multi
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from _loss_mask import resolve_span_ids, span_weights

SCHEMA = pa.schema([
    ("input_ids", pa.list_(pa.int32())),
    ("loss_weights", pa.list_(pa.float32())),
])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--w-header", type=float, default=0.1)
    ap.add_argument("--w-draft", type=float, default=1.0)
    ap.add_argument("--w-final", type=float, default=1.0)
    ap.add_argument("--rows-per-file", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="only used to print steps/epoch")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    begin_id, end_id = resolve_span_ids(tok)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    vocab = len(tok)
    if eos_id is None or pad_id is None:
        raise SystemExit("tokenizer must define both <eos> and <pad>")
    print(f"[tok] vocab={vocab} begin={begin_id} end={end_id} eos={eos_id} pad={pad_id}",
          flush=True)

    files = sorted(a.src.glob("*.parquet")) if a.src.is_dir() else [a.src]
    if not files:
        raise SystemExit(f"no parquet under {a.src}")
    a.out.mkdir(parents=True, exist_ok=True)

    buf_ids: list[int] = []
    buf_w: list[float] = []
    rows: list[dict] = []
    n_file = 0
    stats = {"docs": 0, "multi": 0, "plain": 0, "tokens": 0, "weighted": 0.0,
             "sequences": 0, "too_long": 0, "max_token_id": 0}
    t0 = time.time()

    def flush_rows():
        nonlocal rows, n_file
        if not rows:
            return
        out = a.out / f"tok-{n_file:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out)
        print(f"[tok] wrote {len(rows):,} sequences -> {out}", flush=True)
        rows = []
        n_file += 1

    def close_sequence():
        """Pad the current buffer out to seq_len and emit it."""
        nonlocal buf_ids, buf_w
        if not buf_ids:
            return
        pad = a.seq_len - len(buf_ids)
        rows.append({
            "input_ids": buf_ids + [pad_id] * pad,
            "loss_weights": buf_w + [0.0] * pad,
        })
        stats["sequences"] += 1
        buf_ids, buf_w = [], []

    for path in files:
        table = pq.read_table(path, columns=["document", "kind"])
        for doc, kind in zip(table.column("document").to_pylist(),
                             table.column("kind").to_pylist()):
            ids = tok(doc, add_special_tokens=False).input_ids
            if kind == "plain":
                # The rehearsal half is the ordinary objective: every token,
                # header included, exactly as exp199 was pretrained.
                w = np.ones(len(ids), dtype=np.float32)
            else:
                w = span_weights(np.asarray(ids), w_header=a.w_header,
                                 w_draft=a.w_draft, w_final=a.w_final,
                                 begin_id=begin_id, end_id=end_id)
            # <eos> terminates the document for segment-id purposes. Its weight
            # slot would supervise the FIRST token of the next document in the
            # packed row, so it is zeroed rather than inheriting w_header.
            ids = list(ids) + [eos_id]
            w = np.concatenate([w, np.zeros(1, dtype=np.float32)])

            if len(ids) > a.seq_len:
                stats["too_long"] += 1
                continue
            if len(buf_ids) + len(ids) > a.seq_len:
                close_sequence()
            buf_ids.extend(int(t) for t in ids)
            buf_w.extend(float(v) for v in w)
            stats["docs"] += 1
            stats[kind] += 1
            stats["tokens"] += len(ids)
            stats["weighted"] += float(w.sum())
            stats["max_token_id"] = max(stats["max_token_id"], max(ids))
            if len(rows) >= a.rows_per_file:
                flush_rows()
    close_sequence()
    flush_rows()

    # exp199's embedding table is 2845 rows. The corpus tokenizers published
    # alongside #222's PDB documents are 2848 (they carry the three trailing
    # tokens), and packing an id >= 2845 would silently index off the end of a
    # warm-started embedding.
    if stats["max_token_id"] >= vocab:
        raise SystemExit(f"token id {stats['max_token_id']} >= vocab {vocab}")

    steps = int(np.ceil(stats["sequences"] / a.batch_size))
    density = stats["tokens"] / max(stats["sequences"] * a.seq_len, 1)
    print(f"\n[tok] {stats['docs']:,} documents "
          f"({stats['multi']:,} multi + {stats['plain']:,} plain) "
          f"-> {stats['sequences']:,} sequences of {a.seq_len}")
    print(f"[tok] {stats['tokens']:,} tokens, packing density {density:.1%}, "
          f"{stats['weighted'] / max(stats['tokens'], 1):.1%} of token-weight armed")
    print(f"[tok] max token id {stats['max_token_id']} (vocab {vocab})")
    print(f"[tok] skipped {stats['too_long']:,} documents longer than seq_len")
    print(f"[tok] STEPS_PER_EPOCH at batch {a.batch_size} = {steps:,}")
    print(f"[tok] {(time.time() - t0) / 60:.1f} min")

    (a.out / "tokenized.provenance.json").write_text(json.dumps({
        "tokenizer": str(a.tokenizer), "vocab": vocab, "seq_len": a.seq_len,
        "w_header": a.w_header, "w_draft": a.w_draft, "w_final": a.w_final,
        "plain_weight": "uniform 1.0",
        "batch_size": a.batch_size, "steps_per_epoch": steps,
        "stats": {k: (float(v) if isinstance(v, float) else int(v))
                  for k, v in stats.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
