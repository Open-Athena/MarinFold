# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Publish the exp230 fine-tuning corpus to the public ``open-athena/MarinFold`` bucket.

Layout follows the convention every other corpus on the bucket uses
(``contacts_v1_think``, ``contacts_v1_backtracking``): a ``README.md``, a
``tokenizer/`` directory, and the data underneath.  The prefix carries the
**experiment number** so the corpus is traceable to the run that produced it,
which the older corpus names on this bucket do not do::

    data/document_structures/contacts_v1_multi_exp230/
        README.md
        provenance.json
        tokenizer/      tokenizer.json, tokenizer_config.json
        train/          519,998 documents, 6 parquets
        tokenized/      input_ids + loss_weights, 13 parquets

**The tokenized half is not redundant.** The documents are plain text; the thing
that makes this corpus a *multi-draft* training set is the per-token loss weight
profile (header 0.1 / draft 1.0 / final 1.0, plain rehearsal uniform 1.0), and
that only exists once tokenized. #163 established that the profile is the whole
mechanism -- the restart and stop decisions must carry EQUAL weight or the model
emits exactly one section -- so shipping the weights is shipping the experiment.

**No ``val/``.** This run validated against exp53's published
``contacts_v1/val`` split, unchanged. Copying it under this name would duplicate
bucket data under a label implying it is multi-format, which it is not.

``delete`` is never passed to ``sync_bucket``. It defaults to False, and this is
a shared public bucket carrying every other experiment's corpora; an additive
sync is the only acceptable mode here.

    HF_TOKEN=... python publish_corpus.py --data ~/exp230_data --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

BUCKET = "open-athena/MarinFold"
PREFIX = "data/document_structures/contacts_v1_multi_exp230"
MULTI_TOKEN, MULTI_ID, VOCAB = "<contacts-v1.multi>", 7, 2845

README = """\
# contacts-v1.multi — exp230 fine-tuning corpus

`data/document_structures/contacts_v1_multi_exp230/`

The training corpus for [#230](https://github.com/Open-Athena/MarinFold/issues/230):
teaching `contacts-v1-exp199-1.5B` to write **many candidate contact maps** under a
`<contacts-v1.multi>` marker while staying a **clean single-document decoder** under
plain `<contacts-v1>`.

## What a document looks like

A `<contacts-v1.multi>` document is an ordinary contacts-v1 document with extra
`<begin_statements>` sections in front of the real one. Each extra section is a
**candidate** contact map; a repeated `<begin_statements>` means *discard the
previous candidate and start over*. Only the final section is closed by `<end>`,
so `<end>` keeps its meaning as the document terminator and no inference path
changes.

`<contacts-v1.multi>` is vocab **id 7 renamed in place** (it was
`<contacts-and-distances-v1>`). Vocab size stays 2,845, every other id is
untouched: no embedding resize, no id drift.

## Contents

| | |
|---|---|
| `train/` | 519,998 documents — 259,999 multi-draft + 259,999 plain rehearsal |
| `tokenized/` | the same corpus as `input_ids` + `loss_weights`, 254,493 rows of 8,192 |
| `tokenizer/` | the renamed 2,845-token tokenizer these documents assume |

The two halves are **exactly 1:1 by document and share zero proteins**. Both are
drawn arm-stratified from one decontaminated pool, so disjoint does not mean
differently distributed — their length distributions match to 0.03 residues of
mean L. **Exactly one document per protein**, so nothing is seen twice.

| arm | multi | plain |
|---|---:|---:|
| afdb (#53) | 123,222 | 123,222 |
| esm_atlas (#139) | 123,222 | 123,222 |
| pdb (#222, experimental) | 13,555 | 13,555 |

## The loss weights matter

`tokenized/` carries `loss_weights` alongside `input_ids`. Multi documents use
#163's **profile F** — header **0.1**, every draft section **1.0**, the final
section **1.0** — and plain rehearsal documents are **uniform 1.0** throughout.

`weight[i]` supervises predicting `token[i+1]`, so the last token of a draft is
where the model decides to **restart** and the last token of the final section is
where it decides to **stop**. Those two weights must be **equal**: #163 found that
any asymmetry collapses behaviour to exactly one section. That equality is the
mechanism, not a tuning choice.

Note that 1:1 by document is **84/16 by gradient** — a multi document is ~5.8x
longer, so plain rehearsal is 15.63% of the supervised loss weight.

## Drafts are on-policy

Every draft is a real rollout from exp199 itself (T=1.0, top-p 0.95, top-k
disabled — exp82's settled recipe), 32 per protein. Measured over all **8,319,968**
rollouts: precision **0.4095**, recall 0.4142, F1 0.4090. Sections appear in
**random order**, not sorted by quality, so a later section is not systematically
better than an earlier one.

Section sizes are drawn from a truncated power law `P(n) ~ n^-0.5` on `{1..m}`:
mean 58 contacts, p99 324, max 649. There is **no cap** — 2.63% of sections
exceed 250 contacts, so full-length drafts occur.

## Decontamination

Tier A at 30% identity (#225's rule: identity >=30% over >=50% query coverage, **or**
E <= 1e-3) against #226's 776 eval queries. Drop rates: afdb 2.64%, esm_atlas
1.96%, pdb 5.41%.

## Validation split

This corpus has no `val/`. The #230 run validated against exp53's published
`data/document_structures/contacts_v1/val`, unchanged, as the base-task
retention monitor.

## Provenance

See `provenance.json`. Full writeup, with ten sampled documents:
[DOCUMENTS.md](https://github.com/Open-Athena/MarinFold/blob/main/experiments/exp230_models_contacts_v1_multi_from_exp199/DOCUMENTS.md).
"""


def check_tokenizer(d: Path) -> None:
    t = json.loads((d / "tokenizer.json").read_text())
    vocab = t["model"]["vocab"]
    added = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
    inv = {i: s for s, i in vocab.items()} if isinstance(vocab, dict) else {}
    got = added.get(MULTI_ID, inv.get(MULTI_ID))
    if got != MULTI_TOKEN:
        raise SystemExit(f"FATAL: tokenizer id {MULTI_ID} is {got!r}, expected {MULTI_TOKEN!r}")
    if len(vocab) != VOCAB:
        raise SystemExit(f"FATAL: vocab {len(vocab)}, expected {VOCAB}")
    print(f"[corpus] tokenizer ok: id {MULTI_ID} = {MULTI_TOKEN}, vocab {len(vocab)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/exp230_data")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = Path(a.data).expanduser()
    corpus, tokenized, tokdir = data / "corpus", data / "tokenized", data / "tokenizer_multi"
    for d in (corpus, tokenized, tokdir):
        if not d.is_dir():
            raise SystemExit(f"FATAL: missing {d}")
    check_tokenizer(tokdir)

    prov = json.loads((corpus / "corpus.provenance.json").read_text())
    st = prov.get("stats", {})
    if st.get("multi") != st.get("plain"):
        raise SystemExit(f"FATAL: halves are not 1:1 ({st}) -- refusing to publish")
    if not prov.get("disjoint_halves"):
        raise SystemExit("FATAL: provenance does not record disjoint halves")
    print(f"[corpus] provenance ok: {st['multi']} multi + {st['plain']} plain, disjoint")

    import pyarrow.parquet as pq
    n_docs = sum(pq.ParquetFile(p).metadata.num_rows for p in sorted(corpus.glob("*.parquet")))
    n_tok = sum(pq.ParquetFile(p).metadata.num_rows for p in sorted(tokenized.glob("*.parquet")))
    print(f"[corpus] {n_docs:,} documents, {n_tok:,} tokenized rows")
    if n_docs != st["multi"] + st["plain"]:
        raise SystemExit(f"FATAL: {n_docs} rows on disk vs {st['multi'] + st['plain']} in provenance")

    token = os.environ.get("HF_TOKEN")
    if not token and not a.dry_run:
        raise SystemExit("HF_TOKEN must be set (open-athena-scoped; a personal token 403s)")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    root = f"hf://buckets/{a.bucket}/{a.prefix.strip('/')}"

    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp)
        (meta / "README.md").write_text(README)
        (meta / "provenance.json").write_text(json.dumps(
            {**prov, "n_documents": n_docs, "n_tokenized_rows": n_tok,
             "bucket_prefix": a.prefix}, indent=2) + "\n")
        jobs = [(str(meta), root),
                (str(tokdir), f"{root}/tokenizer"),
                (str(corpus), f"{root}/train"),
                (str(tokenized), f"{root}/tokenized")]
        for src, dest in jobs:
            print(f"[corpus] sync {src} -> {dest}")
            # delete is NEVER passed: this bucket carries every other experiment's
            # corpora and an additive sync is the only acceptable mode.
            plan = api.sync_bucket(source=src, dest=dest, dry_run=a.dry_run, verbose=True)
            print(f"[corpus]   {plan}")
    print("[corpus] DRY RUN, nothing uploaded" if a.dry_run else f"[corpus] published -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
