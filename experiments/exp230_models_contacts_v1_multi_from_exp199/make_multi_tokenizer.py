# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Rename vocab id 7 in place so a document can announce ``<contacts-v1.multi>``.

Vendored from #163's ``make_multi_tokenizer.py`` (which took its source from the
published contacts-v1 tokenizer repo); exp230 renames **exp199's own shipped
tokenizer** instead, so the model and its tokenizer provably agree.

    id 7   <contacts-and-distances-v1>   ->   <contacts-v1.multi>

Why a rename and not an append: appending a token would take exp199's vocab from
2,845 to 2,846 and require an offline embedding resize before warm start
(levanter does not resize), which is #175's path and carries its id-freeze
hazard for the coordinate formats.  A rename changes **no** id, so there is no
resize, no drift, and every published contacts-v1 checkpoint stays readable.
It costs one token: ``<contacts-and-distances-v1>``, the *other* format's
doc-type sentinel, which never appears inside a contacts-v1 document and which
exp199 therefore never saw in training -- its embedding row is effectively fresh.
exp200's RL stack already assumes id 7 spells ``<contacts-v1.multi>``.

**This tokenizer ships WITH the weights.**  Levanter exports whatever tokenizer
it trained under, and a notebook writing the literal string ``<contacts-v1.multi>``
against the published tokenizer would tokenize to garbage.

    uv run python make_multi_tokenizer.py \
        --source /data/exp208_replication/model/C_bf16 \
        --out /data/exp230_multi/tokenizer_multi
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

OLD_TOKEN = "<contacts-and-distances-v1>"
NEW_TOKEN = "<contacts-v1.multi>"
EXPECTED_ID = 7
EXPECTED_VOCAB = 2845


def rename_in_tokenizer_json(path: Path, old: str, new: str) -> int:
    """Rename a token in a ``tokenizer.json`` WordLevel vocab, in place.

    Returns the id it kept.  Raises if the token is missing, if the new name is
    already taken, or if the vocab size would change -- any of which would break
    the "same ids, same embedding" guarantee this whole approach rests on.
    """
    spec = json.loads(path.read_text())
    vocab = spec["model"]["vocab"]
    if old not in vocab:
        raise SystemExit(f"{old!r} not in vocab -- wrong tokenizer?")
    if new in vocab:
        raise SystemExit(f"{new!r} already in vocab -- would collide")
    before = len(vocab)
    tid = vocab.pop(old)
    vocab[new] = tid
    if len(vocab) != before:
        raise SystemExit(f"vocab size changed {before} -> {len(vocab)}")
    for entry in spec.get("added_tokens", []):
        if entry.get("content") == old:
            entry["content"] = new
    path.write_text(json.dumps(spec, ensure_ascii=False))
    return tid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="dir or HF id of the tokenizer exp199 shipped")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    src = AutoTokenizer.from_pretrained(a.source)
    before = dict(src.get_vocab())
    if len(before) != EXPECTED_VOCAB:
        raise SystemExit(f"source vocab is {len(before)}, expected {EXPECTED_VOCAB} -- "
                         "this is not exp199's tokenizer")

    if a.out.exists():
        shutil.rmtree(a.out)
    src.save_pretrained(str(a.out))
    tid = rename_in_tokenizer_json(a.out / "tokenizer.json", OLD_TOKEN, NEW_TOKEN)
    print(f"[exp230] renamed {OLD_TOKEN} -> {NEW_TOKEN} (kept id {tid})")

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
             "<contact> <p0> <p10> <begin_statements> <contact> <p0> <p10> <end>")
    ids = out(probe, add_special_tokens=False).input_ids
    print(f"  vocab {len(after)} (unchanged) - every other id unchanged - "
          f"eos={out.eos_token_id} pad={out.pad_token_id}")
    print(f"  probe ids: {ids}")
    print(f"  round-trip ok: {out.decode(ids) == probe}")
    for t in ("<contacts-v1>", "<begin_sequence>", "<begin_statements>", "<end>",
              "<contact>", "<p0>", NEW_TOKEN):
        print(f"    {t:>22} -> {out.convert_tokens_to_ids(t)}")
    print(f"\n[exp230] wrote {a.out}")
    print("  ships WITH the weights: the published tokenizer cannot read a "
          "multi-draft document, and this one cannot read a "
          "contacts-and-distances-v1 document.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
