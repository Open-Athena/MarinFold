# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the exact finite 8,192-token packs produced from one V2 corpus."""

import argparse
import json
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from census_exp277_delta_documents import CENSUS_ROOT, list_parquets
from convert_exp277_caches_to_delta_stream import (
    CORPORA,
    MAX_DOCUMENT_TOKENS,
    OUTPUT_ROOT,
)

MAX_SEGMENTS_PER_EXAMPLE = 64
GLOBAL_BATCH_SIZE = 128


@jax.jit
def pack_batch(
    carry: tuple[jax.Array, jax.Array, jax.Array], lengths: jax.Array, valid_count: jax.Array
):
    """Advance Levanter's contiguous greedy packing recurrence over one padded batch."""

    def step(state: tuple[jax.Array, jax.Array, jax.Array], item: tuple[jax.Array, jax.Array]):
        index, length = item
        used, segments, finished = state
        flush = (used + length > MAX_DOCUMENT_TOKENS) | (segments >= MAX_SEGMENTS_PER_EXAMPLE)
        candidate = (
            jnp.where(flush, length, used + length),
            jnp.where(flush, 1, segments + 1),
            finished + flush.astype(jnp.int32),
        )
        next_state = jax.tree.map(lambda old, new: jnp.where(index < valid_count, new, old), state, candidate)
        return next_state, None

    indices = jnp.arange(lengths.shape[0], dtype=jnp.int32)
    return jax.lax.scan(step, carry, (indices, lengths))[0]


def count_corpus(corpus: str, documents_root: str) -> dict[str, int | float | str]:
    """Read token lengths in cache order and calculate exact pack utilization."""
    expected_documents = CORPORA[corpus].documents
    carry = (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    documents = 0
    tokens = 0
    for path in list_parquets(documents_root, corpus):
        with fsspec.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=1_000_000, columns=["token_count"]):
                lengths = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
                if lengths.size and (int(lengths.min()) <= 0 or int(lengths.max()) > MAX_DOCUMENT_TOKENS):
                    raise ValueError(f"invalid document length in {path}: {lengths.min()}..{lengths.max()}")
                padded_size = 1 << max(0, (len(lengths) - 1).bit_length())
                padded = np.zeros(padded_size, dtype=np.int32)
                padded[: len(lengths)] = lengths
                carry = pack_batch(carry, jnp.asarray(padded), jnp.asarray(len(lengths), dtype=jnp.int32))
                documents += len(lengths)
                tokens += int(lengths.sum(dtype=np.int64))
    used, segments, finished = (int(value) for value in jax.device_get(carry))
    if documents != expected_documents:
        raise ValueError(f"{corpus}: found {documents} documents, expected {expected_documents}")
    packed_examples = finished + int(segments > 0)
    stored_tokens = packed_examples * MAX_DOCUMENT_TOKENS
    return {
        "corpus": corpus,
        "documents": documents,
        "v2_tokens": tokens,
        "packed_examples": packed_examples,
        "stored_tokens": stored_tokens,
        "padding_tokens": stored_tokens - tokens,
        "padding_fraction": (stored_tokens - tokens) / stored_tokens,
        "final_pack_tokens": used,
        "max_segments_per_example": MAX_SEGMENTS_PER_EXAMPLE,
        "sequence_length": MAX_DOCUMENT_TOKENS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--documents-root", default=OUTPUT_ROOT)
    parser.add_argument("--output-root", default=f"{CENSUS_ROOT}/packing")
    parser.add_argument("--local-output", type=Path)
    args = parser.parse_args()
    summary = count_corpus(args.corpus, args.documents_root)
    output_path = f"{args.output_root.rstrip('/')}/{args.corpus}.json"
    with fsspec.open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    if args.local_output is not None:
        args.local_output.parent.mkdir(parents=True, exist_ok=True)
        args.local_output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
