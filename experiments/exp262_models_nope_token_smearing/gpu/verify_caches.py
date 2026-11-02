# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone check that exp262's training caches are the decontaminated ones.

Runs the same verification ``exp262_train_cw.build_run`` now performs, but on
its own, so an already-launched run can be confirmed without restarting it.
Needs CoreWeave object-storage access, so run it as an in-cluster job.
"""

from exp262_train_cw import (
    AFDB_CACHE,
    ESM_CACHE,
    VALIDATION_CACHE,
    verify_decontaminated_cache_counts,
)
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats

if __name__ == "__main__":
    print(f"[verify] afdb  = {AFDB_CACHE}")
    print(f"[verify] esm   = {ESM_CACHE}")
    print(f"[verify] val   = {VALIDATION_CACHE}")
    verify_decontaminated_cache_counts()
    # The validation cache holds only a validation split, so it has no "train"
    # stats to read. It is exp232's validation cache unchanged, and it is NOT a
    # decontaminated artifact — it predates exp225. That is fine for comparing
    # two arms against each other on identical data, and it is not a
    # generalisation claim about FoldBench proteins.
    stats = read_tokenized_cache_stats(VALIDATION_CACHE, "validation")
    print(f"[verify] validation cache: {stats.total_elements:,} documents, {stats.total_tokens:,} tokens")
    print("[verify] OK - both training caches match exp225's post-decontamination counts")
