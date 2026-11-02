# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Count one epoch using the trainer's own packing implementation.

Run on a US-EAST-02A worker with the experiment's S3 environment. This reads
document-offset metadata in-region, builds the same packing indexes as training,
and prints JSON counts without reading model weights or document token arrays.
Its `packed_examples` total is what `config.EPOCH_PACKED_EXAMPLES` is pinned to,
and `train.py` refuses to launch if the trainer disagrees with that pin.
"""

import asyncio
import gc
import json

import numpy as np
from levanter.data.packing import GreedyPrepackedDataset
from levanter.store.cache import TreeCache

from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_VALIDATION,
    CORPORA,
    GLOBAL_BATCH_SIZE,
)

SEQ_LEN = 8192
MAX_SEGMENTS_PER_EXAMPLE = 64


def count(corpus) -> dict:
    """Pack one cache exactly as the trainer will, and report its shape."""
    cache = TreeCache.load(
        f"{corpus.cache}/{corpus.split}", {"input_ids": np.zeros((1,), dtype=np.int32)}
    )
    packed = GreedyPrepackedDataset(
        cache.jagged_array_tree(),
        SEQ_LEN,
        max_segments_per_example=MAX_SEGMENTS_PER_EXAMPLE,
        slice_strategy="left",
    )
    # Inspect the exact lengths used by the packer so the coverage audit includes
    # the existing 8192-token truncation policy.
    lengths = packed._lengths["input_ids"]
    row = {
        "corpus": corpus.name,
        "split": corpus.split,
        "documents": len(lengths),
        "raw_tokens": int(lengths.sum()),
        "packed_examples": asyncio.run(packed.async_len()),
        "documents_clipped": int((lengths > SEQ_LEN).sum()),
        "retained_tokens": int(np.minimum(lengths, SEQ_LEN).sum()),
    }
    del packed, cache, lengths
    gc.collect()
    return row


def main() -> None:
    results = []
    for corpus in CORPORA:
        row = count(corpus)
        print(json.dumps(row), flush=True)
        results.append(row)
    total = sum(row["packed_examples"] for row in results)
    print(
        json.dumps(
            {
                "packed_examples": total,
                "train_steps": (total + GLOBAL_BATCH_SIZE - 1) // GLOBAL_BATCH_SIZE,
            }
        ),
        flush=True,
    )
    # The held-out shard is not part of the epoch; it is counted so the report
    # can state what training gave up and how large the complex eval batch is.
    print(json.dumps(count(COMPLEX_VALIDATION)), flush=True)


if __name__ == "__main__":
    main()
