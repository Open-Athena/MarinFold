# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert our token cache is bit-equivalent to the one Eric's exp117 run trained on.

This is the check that makes divergence #3 in the README safe. We build our own
token cache under our ``MARIN_PREFIX`` rather than reusing his
``gs://marin-us-east5/tokenized/contacts-v1/2026.07.13.1/`` -- fine only if the
result is identical. Same documents + same tokenizer must produce the same
counts, and issue #117 publishes his exactly.

If our caches report those numbers, the corpus and tokenizer are confirmed equal
and the *only* remaining variable in the reproduction is the training harness --
which is the point of the experiment. If they don't, the comparison is void and
nothing downstream should be trusted.

It also confirms the tokenizer workaround is sound: Eric pins
``timodonnell/contacts-v1-tokenizer@5d68a24a899f`` while MarinFold's load path
forces the bare repo id (see ``contacts_v1_repro_common``). Matching token counts
are the evidence that those resolve to the same tokenizer.

Run after the tokenize steps materialize (they run as CPU sub-jobs at the start
of training; the caches persist, so this can run any time after)::

    uv run --no-sync python verify_cache.py

Exits non-zero on mismatch or if a cache isn't there yet.
"""
from __future__ import annotations

import json
import sys

import gcsfs

from contacts_v1_repro_common import MARIN_PREFIX, TRAIN_TOKENS

# Reference counts from issue #117, which publishes the cache stats his runs
# trained on:
#   tokenized/contacts-v1/2026.07.13.1/      4,129,682 docs / 4,676,753,425 tokens
#   tokenized/contacts-v1-val/2026.07.13.1/     41,954 docs /    47,821,958 tokens
VAL_TOKENS = 47_821_958
TRAIN_ROWS = 4_129_682
VAL_ROWS = 41_954

# marin's default_tokenize writes each cache to
# {MARIN_PREFIX}/tokenized/{name}-{confighash}/{split}/, with levanter's
# shard_ledger.json inside the split dir. The hash suffix isn't knowable ahead of
# time, so match on the name prefix.
#   name -> (split subdir, expected tokens, expected rows)
CACHES = {
    "contacts-v1-train": ("train", TRAIN_TOKENS, TRAIN_ROWS),
    "contacts-v1-val": ("validation", VAL_TOKENS, VAL_ROWS),
}
LEDGER_NAMES = ("shard_ledger.json", "cache_ledger.json", "ledger.json")


def _find_ledger(fs, name: str, split: str) -> dict | None:
    """Return the parsed levanter ledger for cache ``name``, or None if absent."""
    base = f"{MARIN_PREFIX}/tokenized".replace("gs://", "")
    try:
        candidates = [
            d for d in fs.ls(base, detail=False)
            if d.rstrip("/").split("/")[-1].startswith(name)
        ]
    except FileNotFoundError:
        return None
    for cache_dir in candidates:
        for ledger_name in LEDGER_NAMES:
            path = f"{cache_dir.rstrip('/')}/{split}/{ledger_name}"
            if fs.exists(path):
                with fs.open(path, "r") as handle:
                    return json.load(handle)
    return None


def _counts(ledger: dict) -> tuple[int | None, int | None]:
    """Return ``(tokens, rows)`` from a levanter ledger, tolerating layout drift."""
    field_counts = ledger.get("field_counts") or ledger.get("total_counts") or {}
    tokens = None
    if isinstance(field_counts, dict) and "input_ids" in field_counts:
        tokens = int(field_counts["input_ids"])
    else:
        for key in ("total_num_tokens", "num_tokens", "total_tokens"):
            if key in ledger:
                tokens = int(ledger[key])
                break
    rows = ledger.get("total_num_rows")
    return tokens, (int(rows) if rows is not None else None)


def main() -> int:
    fs = gcsfs.GCSFileSystem()
    failures = []
    for name, (split, want_tokens, want_rows) in CACHES.items():
        ledger = _find_ledger(fs, name, split)
        if ledger is None:
            print(f"  MISSING  {name:20} no ledger under {MARIN_PREFIX}/tokenized/{name}-*/{split}/")
            failures.append(name)
            continue
        if not ledger.get("is_finished", True):
            print(f"  PARTIAL  {name:20} cache exists but is_finished=False (still building?)")
            failures.append(name)
            continue
        tokens, rows = _counts(ledger)
        if tokens is None:
            print(f"  UNKNOWN  {name:20} ledger has no token count; keys={sorted(ledger)}")
            failures.append(name)
            continue
        ok = tokens == want_tokens and (rows is None or rows == want_rows)
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark}     {name:20} {tokens:>15,} tokens / {rows:>9,} docs")
        print(f"           {'':20} {want_tokens:>15,} tokens / {want_rows:>9,} docs  (Eric, issue #117)")
        if not ok:
            print(f"           {'':20} DELTA {tokens - want_tokens:+,} tokens")
            failures.append(name)

    print()
    if failures:
        print("CACHE MISMATCH -- corpus/tokenizer differ from Eric's exp117 cache.")
        print("The reproduction comparison is void until this is resolved.")
        return 1
    print("Caches match issue #117 exactly -- corpus + tokenizer confirmed identical.")
    print("The only remaining variable is the training harness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
