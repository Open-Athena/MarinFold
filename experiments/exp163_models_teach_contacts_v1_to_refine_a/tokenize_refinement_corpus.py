# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize + pack the exp163 refinement corpus, carrying the answer-span loss mask.

Turns the raw refinement corpus (parquet, one ``document`` text column — the
output of ``build_refinement_corpus.py``) into the **prebuilt** cache format
current levanter needs in order to train on a masked span:

    input_ids     int32[seq_len]    packed refinement documents
    loss_weights  float32[seq_len]  1.0 on each doc's <begin_statements>..<end>

WHY THIS STEP EXISTS. exp108's training path tokenizes on the fly
(``auto_build_caches=True`` + ``TextLmDatasetFormat``), and exp163 would happily
do the same — except that levanter 1.2 removed ``DatasetComponent.loss_weight_fn``
(the executor-era hook exp120 used). The only surviving way to train on per-token
loss weights is ``PrebuiltLmDatasetFormat(input_ids_key=…, loss_weights_key=…)``,
which reads the weights **out of the cache**. So the mask has to be materialized
here. See ``loss_mask.py`` for the mask itself and why offline is the better home
for it anyway (no cloudpickle across the fray boundary; ids resolved, not assumed).

PACKING IS OURS TOO. ``PrebuiltLmDataset`` has no packing mode — it maps one cache
row to one training example — so this script does the greedy packing levanter's
``PackedTokenDataset`` would otherwise do: fill a ``--seq-len`` sequence with whole
documents (never split one; a partial protein document is nonsense), pad the tail
with ``<pad>`` at loss weight 0. Cross-document attention is still blocked at train
time: ``PrebuiltLmDataset`` passes ``eos_id`` to ``GrugLmExample.causal``, which
derives segment ids from the ``<eos>`` this script appends after each document —
exactly the levanter tokenizer's own ``enforce_eos`` behaviour.

Run (workstation; reads/writes CoreWeave S3 — the corpus is ~100 MB):

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run python tokenize_refinement_corpus.py \\
        --in  s3://marin-us-east-02a/MarinFold/exp163/val10k/refinement_corpus/'*.parquet' \\
        --out s3://marin-us-east-02a/MarinFold/exp163/val10k/refinement_tokenized

Print the token total it reports: it is what ``EXP163_STEPS_PER_EPOCH`` is derived
from (steps/epoch = ceil(train_tokens / (batch * seq_len)), and with prebuilt
packing that is simply ``ceil(n_sequences / batch)`` — the script prints both).
"""

from __future__ import annotations

import argparse
import os

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from loss_mask import (
    answer_span_loss_weights,
    check_document_spans,
    find_sections,
    resolve_span_ids,
    span_weights,
)

# CoreWeave AI Object Storage rejects path-style requests; s3fs must address the
# bucket virtual-hosted. Same knob the rollout bootstrap exports on the pods.
S3_CONFIG_KWARGS = {"s3": {"addressing_style": "virtual"}}

DEFAULT_TOKENIZER = "timodonnell/contacts-v1-tokenizer"


def _fs_for(url: str):
    """fsspec filesystem for ``url``, with the CoreWeave S3 quirks applied."""
    protocol = url.split("://", 1)[0] if "://" in url else "file"
    if protocol != "s3":
        return fsspec.filesystem(protocol)
    return fsspec.filesystem(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        config_kwargs=S3_CONFIG_KWARGS,
    )


def _strip_protocol(url: str) -> str:
    return url.split("://", 1)[1] if "://" in url else url


def read_documents(in_url: str, text_key: str) -> list[str]:
    """Read the ``text_key`` column from every parquet shard matched by ``in_url``."""
    fs = _fs_for(in_url)
    paths = sorted(fs.glob(_strip_protocol(in_url)))
    if not paths:
        raise SystemExit(f"no parquet shards matched {in_url}")
    docs: list[str] = []
    for path in paths:
        with fs.open(path, "rb") as handle:
            table = pq.read_table(handle, columns=[text_key])
        docs.extend(table.column(text_key).to_pylist())
        print(f"  read {path} ({table.num_rows} docs)", flush=True)
    return docs


def encode_documents(docs: list[str], tokenizer, seq_len: int, *, fmt: str = "candidate",
                     w_header: float = 0.0, w_draft: float = 0.0,
                     w_final: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tokenize each document and compute its per-token loss weights.

    Mirrors levanter's ``BatchTokenizer``: ``encode(text, add_special_tokens=False)``
    then append ``<eos>`` (the contacts-v1 tokenizer has no bos, so nothing is
    prepended).

    ``fmt="candidate"`` (v1) arms the single answer span and nothing else.
    ``fmt="multi-draft"`` (v2) applies the three-way profile: sequence header at
    ``w_header``, every draft section at ``w_draft``, the LAST section at
    ``w_final``. Because the weights are applied here rather than baked into the
    text, one text corpus can be tokenized several times to sweep the profile.
    """
    begin_id, end_id = resolve_span_ids(tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise SystemExit("tokenizer has no eos token; packing needs it for segment ids")

    encoded: list[tuple[np.ndarray, np.ndarray]] = []
    n_too_long = 0
    for i, text in enumerate(docs):
        ids = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
        if fmt == "candidate":
            check_document_spans(ids, begin_id=begin_id, end_id=end_id)
        else:
            find_sections(ids, begin_id=begin_id, end_id=end_id)   # validates the grammar
        # The eos rides INSIDE the document so the packer's segment boundaries land
        # correctly. In v1 it is never trained (the mask closes at <end>); in v2 it
        # IS trained -- it is the document terminator now that <end> only closes a
        # section, so the final section's <end> position carries w_final.
        ids = np.append(ids, np.int32(eos_id))
        if len(ids) > seq_len:
            # The corpus builder budgets documents to fit ctx=8192 including
            # <end>; +1 for eos can only bite at the exact boundary.
            n_too_long += 1
            continue
        if fmt == "candidate":
            weights = answer_span_loss_weights(ids, begin_id=begin_id, end_id=end_id)
        else:
            weights = span_weights(ids, w_header=w_header, w_draft=w_draft,
                                   w_final=w_final, begin_id=begin_id, end_id=end_id)
        encoded.append((ids, weights))
        if (i + 1) % 5000 == 0:
            print(f"  tokenized {i + 1}/{len(docs)}", flush=True)
    if n_too_long:
        print(f"  WARNING: dropped {n_too_long} document(s) longer than seq_len={seq_len}", flush=True)
    return encoded


def pack(
    encoded: list[tuple[np.ndarray, np.ndarray]],
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily pack whole documents into fixed ``seq_len`` sequences.

    Returns ``(input_ids[n, seq_len] int32, loss_weights[n, seq_len] float32)``.
    A document is never split; the tail of each sequence is ``<pad>`` at weight
    0.0, so padding contributes nothing to the loss.
    """
    sequences: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    cur_ids: list[np.ndarray] = []
    cur_w: list[np.ndarray] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_ids, cur_w, cur_len
        if not cur_ids:
            return
        ids = np.concatenate(cur_ids)
        w = np.concatenate(cur_w)
        padding = seq_len - len(ids)
        if padding:
            ids = np.append(ids, np.full(padding, pad_id, dtype=np.int32))
            w = np.append(w, np.zeros(padding, dtype=np.float32))
        sequences.append(ids)
        weights.append(w)
        cur_ids, cur_w, cur_len = [], [], 0

    for ids, w in encoded:
        if cur_len + len(ids) > seq_len:
            flush()
        cur_ids.append(ids)
        cur_w.append(w)
        cur_len += len(ids)
    flush()

    return np.stack(sequences), np.stack(weights)


def write_shards(
    input_ids: np.ndarray,
    loss_weights: np.ndarray,
    *,
    out_url: str,
    rows_per_shard: int,
) -> list[str]:
    """Write ``(input_ids, loss_weights)`` list-columns as parquet shards."""
    fs = _fs_for(out_url)
    base = _strip_protocol(out_url).rstrip("/")
    n_shards = max(1, -(-len(input_ids) // rows_per_shard))
    written: list[str] = []
    for shard in range(n_shards):
        lo, hi = shard * rows_per_shard, min((shard + 1) * rows_per_shard, len(input_ids))
        table = pa.table(
            {
                "input_ids": pa.array(list(input_ids[lo:hi]), type=pa.list_(pa.int32())),
                "loss_weights": pa.array(list(loss_weights[lo:hi]), type=pa.list_(pa.float32())),
            }
        )
        path = f"{base}/tokenized-{shard:05d}-of-{n_shards:05d}.parquet"
        with fs.open(path, "wb") as handle:
            pq.write_table(table, handle, compression="zstd")
        written.append(path)
        print(f"  wrote {path} ({hi - lo} sequences)", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_url", required=True, help="raw corpus parquet glob (text column)")
    ap.add_argument("--out", dest="out_url", required=True, help="output prefix for tokenized shards")
    ap.add_argument("--text-key", default="document")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument("--train-batch", type=int, default=128, help="only used to print steps/epoch")
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N documents")
    ap.add_argument("--format", choices=["candidate", "multi-draft"], default="candidate")
    ap.add_argument("--w-header", type=float, default=0.0,
                    help="multi-draft: weight on the sequence header (the original "
                         "contacts-v1 LM signal -- the anti-forgetting lever)")
    ap.add_argument("--w-draft", type=float, default=0.0,
                    help="multi-draft: weight on draft sections. NOTE these hold ~13%%-"
                         "precision contacts, so this trains the model to emit wrong "
                         "contacts -- sweep it, do not assume it is safe")
    ap.add_argument("--w-final", type=float, default=1.0,
                    help="multi-draft: weight on the final (ground-truth) section")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print(f"[exp163] reading {a.in_url}", flush=True)
    docs = read_documents(a.in_url, a.text_key)
    if a.limit:
        docs = docs[: a.limit]
    print(f"[exp163] {len(docs)} documents; tokenizing with {a.tokenizer}", flush=True)

    encoded = encode_documents(docs, tokenizer, a.seq_len, fmt=a.format,
                               w_header=a.w_header, w_draft=a.w_draft, w_final=a.w_final)
    doc_tokens = sum(len(ids) for ids, _ in encoded)
    answer_tokens = sum(float(w.sum()) for _, w in encoded)     # weighted, not a count

    input_ids, loss_weights = pack(encoded, seq_len=a.seq_len, pad_id=pad_id)
    n_seq = len(input_ids)
    steps_per_epoch = -(-n_seq // a.train_batch)

    profile = (
        f"(header {a.w_header} / draft {a.w_draft} / final {a.w_final})"
        if a.format == "multi-draft"
        else "(answer span only)"
    )
    print(
        f"\n[exp163] packed {len(encoded)} documents -> {n_seq} sequences of {a.seq_len}\n"
        f"         document tokens : {doc_tokens:,} (incl. eos)\n"
        f"         weighted tokens : {answer_tokens:,.0f} "
        f"({answer_tokens / max(1, doc_tokens):.1%} of document tokens, weight-summed)\n"
        f"         profile         : {a.format} {profile}\n"
        f"         packing density : {doc_tokens / (n_seq * a.seq_len):.1%}\n"
        f"         EXP163_STEPS_PER_EPOCH = {steps_per_epoch}  (batch {a.train_batch})",
        flush=True,
    )

    print(f"\n[exp163] writing {a.out_url}", flush=True)
    write_shards(input_ids, loss_weights, out_url=a.out_url, rows_per_shard=a.rows_per_shard)
    print("[exp163] done.", flush=True)


if __name__ == "__main__":
    main()
