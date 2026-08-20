# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the contacts-v1 tokenizer variant used by the multi-draft format (#163).

The v2 multi-draft documents open with a distinct doc-type sentinel so the model
knows it is in "emit several candidate structures" mode rather than plain
contacts-v1 mode. There is no ``<contacts-v1.multi>`` token in the published
vocab, so this script produces one by **renaming id 7 in place**:

    id 7   <contacts-and-distances-v1>   ->   <contacts-v1.multi>

Everything else is untouched: **vocab size stays 2845 and every other id keeps
its value**, so embedding row 7 is literally the same row and every existing
checkpoint remains weight-compatible. Nothing about training or inference changes
numerically — this is purely so the document *text* does not carry a misleading
sentinel. (Appending a genuinely new token would instead force vocab 2845 -> 2846,
an embedding resize on warm-start, and a fresh random row.)

Why id 7 is the right one to take: it is the OTHER format's doc-type sentinel, so
it already occupies exactly this role in position 0, and it is never emitted inside
a contacts-v1 document — which is also why E8 has never trained its embedding on
in-document context, leaving it uncommitted for a new meaning. v2 freed it when the
``<CAND>`` marker was dropped.

.. warning::
   **This tokenizer can no longer read contacts-and-distances-v1 documents** — that
   format's doc-type sentinel no longer exists in the vocab, so its documents would
   hit the unknown token. Conversely the *published* contacts-v1 tokenizer cannot
   read multi-draft documents. A corpus must therefore ship with its tokenizer
   (which is the co-location convention regardless).

   For *training* the two are interchangeable: ids are identical, and the
   contacts-v1 val split contains neither spelling. The rename only matters where
   multi-draft document text is tokenized.

    uv run python make_multi_tokenizer.py --out ./contacts-v1-multi-tokenizer
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

OLD_TOKEN = "<contacts-and-distances-v1>"
NEW_TOKEN = "<contacts-v1.multi>"
EXPECTED_ID = 7
EXPECTED_VOCAB = 2845
SOURCE = "timodonnell/contacts-v1-tokenizer"


def rename_in_tokenizer_json(path: Path, old: str, new: str) -> int:
    """Rename a token in a ``tokenizer.json`` WordLevel vocab, in place.

    Returns the id it kept. Raises if the token is missing, if the new name is
    already taken, or if the vocab size would change — any of which would break
    the "same ids, same embedding" guarantee this whole approach rests on.
    """
    with open(path) as fh:
        spec = json.load(fh)

    vocab = spec["model"]["vocab"]
    if old not in vocab:
        raise SystemExit(f"{old!r} not in vocab — wrong tokenizer?")
    if new in vocab:
        raise SystemExit(f"{new!r} already in vocab — would collide")
    before = len(vocab)
    tid = vocab.pop(old)
    vocab[new] = tid
    if len(vocab) != before:
        raise SystemExit(f"vocab size changed {before} -> {len(vocab)}")

    # added_tokens carries its own copy of the string for special tokens
    for entry in spec.get("added_tokens", []):
        if entry.get("content") == old:
            entry["content"] = new

    with open(path, "w") as fh:
        json.dump(spec, fh, ensure_ascii=False)
    return tid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    src = AutoTokenizer.from_pretrained(a.source)
    before = dict(src.get_vocab())
    if len(before) != EXPECTED_VOCAB:
        print(f"  NOTE: source vocab is {len(before)}, expected {EXPECTED_VOCAB}")

    if a.out.exists():
        shutil.rmtree(a.out)
    src.save_pretrained(str(a.out))

    tid = rename_in_tokenizer_json(a.out / "tokenizer.json", OLD_TOKEN, NEW_TOKEN)
    print(f"[exp163] renamed {OLD_TOKEN} -> {NEW_TOKEN} (kept id {tid})")

    # ---- verify the guarantee: same size, same ids everywhere else ----
    out = AutoTokenizer.from_pretrained(str(a.out))
    after = dict(out.get_vocab())
    if len(after) != len(before):
        raise SystemExit(f"vocab size changed: {len(before)} -> {len(after)}")
    if after.get(NEW_TOKEN) != EXPECTED_ID:
        raise SystemExit(f"{NEW_TOKEN} got id {after.get(NEW_TOKEN)}, expected {EXPECTED_ID}")
    if OLD_TOKEN in after:
        raise SystemExit(f"{OLD_TOKEN} still present")
    drift = {t: (i, after[t]) for t, i in before.items()
             if t != OLD_TOKEN and after.get(t) != i}
    if drift:
        raise SystemExit(f"{len(drift)} token(s) changed id, e.g. {list(drift.items())[:3]}")

    probe = ("<contacts-v1.multi> <begin_sequence> <MET> <begin_statements> "
             "<contact> <p0> <p10> <end>")
    ids = out(probe, add_special_tokens=False).input_ids
    print(f"  vocab {len(after)} (unchanged) · every other id unchanged · "
          f"eos={out.eos_token_id} pad={out.pad_token_id}")
    print(f"  probe ids: {ids}")
    print(f"  round-trip: {out.decode(ids)!r}")
    for t in ("<contacts-v1>", "<begin_statements>", "<end>", "<contact>", NEW_TOKEN):
        print(f"    {t:>22} -> {out.convert_tokens_to_ids(t)}")
    print(f"\n[exp163] wrote {a.out}")
    print("  ⚠ ships WITH its corpus: it cannot read contacts-and-distances-v1 documents, "
          "and the published tokenizer cannot read multi-draft documents.")


if __name__ == "__main__":
    main()
