# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize and pack the pilot corpus slice into flat uint16 token streams.

Mirrors what levanter's ``TextLmDatasetFormat`` does for the production runs:
each document is tokenized and an ``<eos>`` is appended, then documents are
concatenated and chunked into fixed-length sequences. Documents therefore start
at arbitrary offsets inside a training sequence, exactly as in production —
which matters here, because that packing is what already forces the model to be
invariant to *absolute* position and leaves only the relative question open.

(Levanter also asks for a BOS, but the contacts-v1 tokenizer defines no
``bos_token``, so production has none to add either. Documents carry their own
``<contacts-v1>`` opener.)

Writes ``tokens.u16`` plus a small JSON manifest per split.
"""

import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from marinfold.inference._tokenizer import load_tokenizer, model_source_path
from marinfold.registry import resolve_model

TEXT_COLUMN = "document"
_TOKENIZER = None


def _tokenizer_for(directory: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(Path(directory))
    return _TOKENIZER


def tokenize_shard(task: tuple[str, str]) -> np.ndarray:
    """Tokenize one parquet shard into a flat uint16 array."""
    shard, directory = task
    tokenizer = _tokenizer_for(directory)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("the contacts-v1 tokenizer has no <eos>; packing needs a separator")
    documents = pq.read_table(shard, columns=[TEXT_COLUMN]).column(TEXT_COLUMN).to_pylist()
    encoded = tokenizer(documents, add_special_tokens=False).input_ids
    pieces = []
    for ids in encoded:
        pieces.append(np.asarray(ids, dtype=np.uint16))
        pieces.append(np.asarray([eos], dtype=np.uint16))
    return np.concatenate(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("/data/tim/exp262_pilot/corpus"))
    parser.add_argument("--out", type=Path, default=Path("/data/tim/exp262_pilot/packed"))
    parser.add_argument("--workers", type=int, default=32)
    arguments = parser.parse_args()

    directory = str(Path(model_source_path(Path(resolve_model(None)))))
    tokenizer = load_tokenizer(Path(directory))
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary does not fit in uint16")

    for split in ("train", "val"):
        shards = sorted(glob.glob(str(arguments.corpus / split / "*.parquet")))
        if not shards:
            raise SystemExit(f"no parquet shards under {arguments.corpus / split}")
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            chunks = list(pool.map(tokenize_shard, [(shard, directory) for shard in shards], chunksize=1))
        stream = np.concatenate(chunks)
        destination = arguments.out / split
        destination.mkdir(parents=True, exist_ok=True)
        stream.tofile(destination / "tokens.u16")
        manifest = {
            "shards": len(shards),
            "tokens": int(stream.size),
            "vocab_size": int(tokenizer.vocab_size),
            "eos_token_id": int(tokenizer.eos_token_id),
            "tokenizer": directory,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[pack] {split}: {len(shards)} shards -> {stream.size / 1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
