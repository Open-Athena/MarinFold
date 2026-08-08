# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Recompute ``loss_weights`` on an already-tokenized exp163 corpus.

A weight-profile arm changes only ``loss_weights``; ``input_ids`` are identical. So a
new arm needs no re-tokenization -- the section structure is fully recoverable from the
ids themselves (``<begin_statements>``=9 opens a section, ``<end>``=10 terminates a
document, and the document-type sentinel at position 0 is ``<contacts-v1>``=2 for a
plain rehearsal document or ``<contacts-v1.multi>``=7 for a multi-draft one).

The one wrinkle is packing: a row holds SEVERAL whole documents concatenated, then
``<pad>``. ``find_sections`` assumes exactly one ``<end>``, so the row must be split
back into documents at each ``<end>`` before weights are computed per document.

Why this arm exists: measured on the real corpus, a draft section averages 53.7
contacts against the final's 200.6, so the final dominates the weighted-token budget
unless drafts carry a comparable per-token weight. Profiles with w_draft 0 / 0.1 / 0.3
(<=39% of the signal on drafts) were all tested on TPU and all generate exactly ONE
section -- the model emits a final-shaped answer and stops. Teaching multi-draft
*generation* needs drafts to carry roughly half the signal or more.
"""
from __future__ import annotations

import argparse

import numpy as np

from loss_mask import BEGIN_STATEMENTS_ID, END_ID, span_weights

PAD_ID = 0
EOS_ID = 1       # appended after <end>; the real document boundary when packed
PLAIN_SENTINEL = 2   # <contacts-v1>        -- E8's own objective, weight 1.0 throughout
MULTI_SENTINEL = 7   # <contacts-v1.multi>  -- header / draft / final profile applies


def row_weights(ids: np.ndarray, *, w_header: float, w_draft: float,
                w_final: float) -> np.ndarray:
    """Per-token loss weights for one PACKED row (several whole docs + padding).

    Documents in the packed stream terminate ``... <end> <eos>``: the tokenizer
    appends ``<eos>`` after ``<end>`` so levanter's ``PrebuiltLmDataset`` can derive
    segment ids. So the split boundary is ``<eos>``, NOT ``<end>`` -- splitting on
    ``<end>`` leaves the next slice starting on the stray ``<eos>``.

    ``span_weights`` is applied to the document *including* its trailing ``<eos>``,
    which is exactly what ``tokenize_refinement_corpus`` did when the corpus was
    built, so a ``(0, 0, 1)`` profile reproduces the original mask bit-for-bit.
    """
    w = np.zeros(len(ids), dtype=np.float32)
    ends = [int(i) for i in np.flatnonzero(ids == EOS_ID)]
    start = 0
    for e in ends:
        doc = ids[start:e + 1]
        if len(doc) and doc[0] == PLAIN_SENTINEL:
            w[start:e + 1] = 1.0                      # unmasked, as E8 was trained
        elif len(doc) and doc[0] == MULTI_SENTINEL:
            w[start:e + 1] = span_weights(doc, w_header=w_header, w_draft=w_draft,
                                          w_final=w_final,
                                          begin_id=BEGIN_STATEMENTS_ID, end_id=END_ID)
        else:
            raise ValueError(f"document at {start} opens with id {doc[0] if len(doc) else None}, "
                             f"expected {PLAIN_SENTINEL} or {MULTI_SENTINEL}")
        # weight[i] supervises predicting token[i+1], so the slot ON a document's
        # terminating <eos> would supervise the FIRST token of whatever document
        # packing placed next -- a cross-document target that is an artifact of
        # packing, not of the task. Emitting <eos> is already supervised by the
        # <end> slot before it. (With w_header=0 this was zero anyway, so the
        # existing v1/v3 corpora are unaffected.)
        w[e] = 0.0
        start = e + 1
    if start < len(ids) and not np.all(ids[start:] == PAD_ID):
        raise ValueError(f"non-pad tokens after the final <eos> at {start}")
    return w


def _selftest() -> None:
    """Synthetic packed row: one multi doc (2 drafts + final) then one plain doc.

    Documents end ``<end> <eos>``, mirroring the real packed corpus.
    """
    def sec(n):     return [BEGIN_STATEMENTS_ID] + [11, 12, 13] * n
    multi = [MULTI_SENTINEL, 8, 20, 21] + sec(2) + sec(2) + sec(3) + [END_ID, EOS_ID]
    plain = [PLAIN_SENTINEL, 8, 20, 21] + sec(4) + [END_ID, EOS_ID]
    ids = np.array(multi + plain + [PAD_ID] * 5)
    w = row_weights(ids, w_header=0.1, w_draft=1.0, w_final=2.0)
    m = len(multi)
    assert w[0] == 0.1, w[0]                              # multi header
    assert w[4] == 1.0 and w[11] == 1.0                   # the two drafts
    assert w[18] == 2.0, w[18]                            # final section
    assert w[m - 3] == 2.0, w[m - 3]                      # last contact tok -> emits <end>
    assert w[m - 1] == 0.0, w[m - 1]                      # <eos> slot: no cross-doc target
    assert np.all(w[m:m + len(plain) - 1] == 1.0)         # plain doc, unmasked ...
    assert w[m + len(plain) - 1] == 0.0                   # ... except its own <eos> slot
    assert np.all(w[m + len(plain):] == 0.0)              # padding
    # (0,0,1) must still reproduce the v1/v3 mask: only the final section carries loss
    w2 = row_weights(ids, w_header=0.0, w_draft=0.0, w_final=1.0)
    assert w2[0] == 0.0 and w2[4] == 0.0 and w2[11] == 0.0
    assert w2[18] == 1.0 and w2[m - 3] == 1.0
    print("[reweight] selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", help="glob of tokenized parquet (input_ids, loss_weights)")
    ap.add_argument("--dst", help="output dir (gs:// or s3://)")
    ap.add_argument("--w-header", type=float, default=0.1)
    ap.add_argument("--w-draft", type=float, default=1.0)
    ap.add_argument("--w-final", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.src:
        _selftest()
        if not a.src:
            return

    import pyarrow as pa
    import pyarrow.parquet as pq

    def _fs_for(url: str):
        """Build the filesystem EXPLICITLY rather than via ``url_to_fs``.

        ``url_to_fs`` picks config out of the environment, and fsspec on the marin
        pods does not JSON-decode ``FSSPEC_S3_CONFIG_KWARGS`` -- s3fs then gets a
        ``str`` where it expects a dict and dies in ``_prepare_config_kwargs`` with
        ``'str' object has no attribute 'copy'``. Constructing it here also keeps
        the CoreWeave virtual-addressing requirement in one obvious place.
        """
        if url.startswith("s3://"):
            import s3fs
            return s3fs.S3FileSystem(config_kwargs={"s3": {"addressing_style": "virtual"}})
        if url.startswith("gs://"):
            import gcsfs
            return gcsfs.GCSFileSystem()
        import fsspec
        return fsspec.core.url_to_fs(url)[0]

    _selftest()
    src_fs = _fs_for(a.src)
    dst_fs = _fs_for(a.dst)
    dst_root = a.dst.split("://", 1)[1].rstrip("/") if "://" in a.dst else a.dst.rstrip("/")
    files = sorted(src_fs.glob(a.src.split("://", 1)[1] if "://" in a.src else a.src))
    if not files:
        raise SystemExit(f"no shards matched {a.src}")
    proto = a.src.split("://", 1)[0] if "://" in a.src else ""
    print(f"[reweight] {len(files)} shard(s) | header={a.w_header} draft={a.w_draft} "
          f"final={a.w_final}", flush=True)

    tot_slots = tot_w = 0
    for n, f in enumerate(files):
        with src_fs.open(f, "rb") as fh:
            t = pq.read_table(fh)
        ids_col = t.column("input_ids").to_pylist()
        new_w = []
        for ids in ids_col:
            arr = np.asarray(ids)
            w = row_weights(arr, w_header=a.w_header, w_draft=a.w_draft, w_final=a.w_final)
            tot_slots += w.size
            tot_w += float(w.sum())
            new_w.append(w.tolist())
        out = pa.table({"input_ids": t.column("input_ids"),
                        "loss_weights": pa.array(new_w, type=pa.list_(pa.float32()))})
        dest = f"{dst_root}/{f.rsplit('/', 1)[-1]}"
        with dst_fs.open(dest, "wb") as fh:
            pq.write_table(out, fh)
        print(f"[reweight]   {n+1}/{len(files)} {f.rsplit('/', 1)[-1]}", flush=True)
    print(f"[reweight] DONE weighted={tot_w:,.0f}/{tot_slots:,} slots "
          f"({100*tot_w/tot_slots:.1f}%) -> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
