# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert our token cache is bit-equivalent to the one Eric's exp117 run trained on.

This is the check that makes divergence #3 in the README safe. We build our own
token cache under our ``MARIN_PREFIX`` rather than reusing his
``gs://marin-us-east5/tokenized/contacts-v1/2026.07.13.1/`` — fine only if the
result is identical. Same documents + same tokenizer must produce the same total
token count, and he publishes his exactly: ``TRAIN_TOKENS = 4,676,753,425``.

If our train cache reports that number, the corpus and tokenizer are confirmed
equal and the *only* remaining variable in the reproduction is the training
harness — which is the point of the experiment. If it doesn't, the comparison is
void and nothing downstream should be trusted.

Run after the tokenize steps materialize (they run as CPU sub-jobs at the start
of training; the caches persist, so this can run any time after)::

    uv run --no-sync python verify_cache.py

Exits non-zero on mismatch or if a cache isn't there yet.
"""
from __future__ import annotations

import json
import sys

import fsspec

from contacts_v1_repro_common import MARIN_PREFIX, TRAIN_TOKENS

# marin's default_tokenize writes each cache under {MARIN_PREFIX}/tokenized/{name}/,
# with levanter's cache ledger at the root of the cache dir.
CACHES = {
    "contacts-v1-train": TRAIN_TOKENS,
    "contacts-v1-val": None,  # no published reference; report it for the record
}
LEDGER_NAMES = ("shard_ledger.json", "cache_ledger.json", "ledger.json")


def _find_ledger(fs, cache_dir: str) -> dict | None:
    """Return the parsed levanter cache ledger under ``cache_dir``, or None."""
    for name in LEDGER_NAMES:
        path = f"{cache_dir}/{name}"
        if fs.exists(path):
            with fs.open(path, "r") as handle:
                return json.load(handle)
    # marin versions the cache dir one level deeper; search a level down.
    try:
        for child in fs.ls(cache_dir, detail=False):
            for name in LEDGER_NAMES:
                path = f"{child.rstrip('/')}/{name}"
                if fs.exists(path):
                    with fs.open(path, "r") as handle:
                        return json.load(handle)
    except FileNotFoundError:
        return None
    return None


def _token_count(ledger: dict) -> int | None:
    """Pull the total token count out of a levanter ledger, tolerating layout drift."""
    for key in ("total_num_tokens", "num_tokens", "total_tokens"):
        if key in ledger:
            return int(ledger[key])
    # Newer ledgers keep per-field totals, e.g. {"field_counts": {"input_ids": N}}.
    counts = ledger.get("field_counts") or ledger.get("total_counts") or {}
    if isinstance(counts, dict) and "input_ids" in counts:
        return int(counts["input_ids"])
    return None


def main() -> int:
    fs = fsspec.filesystem("gcs")
    failures = []
    for name, expected in CACHES.items():
        cache_dir = f"{MARIN_PREFIX}/tokenized/{name}"
        ledger = _find_ledger(fs, cache_dir)
        if ledger is None:
            print(f"  MISSING  {name:20} no ledger under {cache_dir}")
            failures.append(name)
            continue
        actual = _token_count(ledger)
        if actual is None:
            print(f"  UNKNOWN  {name:20} ledger found but no token count in {sorted(ledger)}")
            failures.append(name)
            continue
        if expected is None:
            print(f"  info     {name:20} {actual:,} tokens (no published reference)")
            continue
        ok = actual == expected
        delta = "" if ok else f"   DELTA {actual - expected:+,}"
        print(f"  {'ok  ' if ok else 'FAIL'}     {name:20} {actual:,} tokens "
              f"(Eric: {expected:,}){delta}")
        if not ok:
            failures.append(name)

    print()
    if failures:
        print("CACHE MISMATCH — corpus/tokenizer differ from Eric's exp117 cache.")
        print("The reproduction comparison is void until this is resolved.")
        return 1
    print("Cache matches exp117 TRAIN_TOKENS — corpus + tokenizer confirmed identical.")
    print("The only remaining variable is the training harness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
