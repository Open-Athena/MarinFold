"""Count one epoch using the trainer's packing implementation.

Run on a US-EAST-02A worker with the experiment's S3 environment. This reads
document-offset metadata in-region, builds the same packing indexes as training,
and prints JSON counts without reading model weights or document token arrays.
"""

import asyncio
import gc
import json

import numpy as np
from levanter.data.packing import GreedyPrepackedDataset
from levanter.store.cache import TreeCache

from experiments.exp277_models_single_mpnn_pilot.config import CORPORA


def main() -> None:
    results = []
    for corpus in CORPORA:
        cache = TreeCache.load(
            f"{corpus.cache}/train", {"input_ids": np.zeros((1,), dtype=np.int32)}
        )
        packed = GreedyPrepackedDataset(
            cache.jagged_array_tree(),
            8192,
            max_segments_per_example=64,
            slice_strategy="left",
        )
        # Inspect the exact lengths used by the packer so the coverage audit
        # includes the existing 8192-token truncation policy.
        lengths = packed._lengths["input_ids"]
        row = {
            "corpus": corpus.name,
            "documents": len(lengths),
            "raw_tokens": int(lengths.sum()),
            "packed_examples": asyncio.run(packed.async_len()),
            "documents_clipped": int((lengths > 8192).sum()),
            "retained_tokens": int(np.minimum(lengths, 8192).sum()),
        }
        print(json.dumps(row), flush=True)
        results.append(row)
        del packed, cache, lengths
        gc.collect()
    total = sum(row["packed_examples"] for row in results)
    print(json.dumps({"packed_examples": total, "train_steps": (total + 127) // 128}))


if __name__ == "__main__":
    main()
