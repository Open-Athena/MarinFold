# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Levanter token caches for contacts-v1-think with masked ``<think>`` targets.

The exp126 corpus is published as raw document parquet in the public
``open-athena/MarinFold`` HF bucket. This script turns those shards into a
region-local Levanter cache with two fields:

* ``input_ids``: contacts-v1 tokenizer ids, with the normal tokenizer EOS
  appended by Levanter's text preprocessor path.
* ``loss_weights``: per-position weights aligned with Levanter causal LM loss;
  a position gets weight 0 when its target token is ``<think>``.

Training then reads this cache through Levanter's prebuilt LM-dataset format,
packing with ``loss_weights_key="loss_weights"`` so the TPU hot path never
tokenizes or streams raw HF parquet.
"""

import argparse
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np

from huggingface_hub import snapshot_download
from levanter.data._preprocessor import BatchProcessor
from levanter.data.sharded_datasource import FirstRowsShardedDataSource, UrlDataSource
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.data.text.cache import build_lm_dataset_cache
from levanter.data.text.formats import LmDatasetFormatBase
from levanter.store.cache import CacheMetadata, CacheOptions, consolidate_shard_cache_ledgers
from levanter.tokenizers import MarinTokenizer, load_tokenizer as load_marin_tokenizer

HF_RESOLVE_ROOT = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
DATA_PREFIX = f"{HF_RESOLVE_ROOT}/data/document_structures/contacts_v1_think"
SHARDS = {"train": 2067, "validation": 22, "test": 22}
HF_SPLITS = {"train": "train", "validation": "val", "test": "test"}
# Two train shard slots are absent from the exp126 HF bucket. Levanter's URL
# datasource validates URLs eagerly, so skip them explicitly instead of letting
# workers fail after the cache job has launched.
MISSING_SHARDS = {"train": {858, 1423}}
CONTACTS_TOKENIZER = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
THINK_TOKEN_ID = 6


def shard_urls(split: str, *, max_shards: int | None = None) -> list[str]:
    """Return CDN parquet URLs for a split of the exp126 HF-bucket corpus."""
    if split not in SHARDS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SHARDS)}")
    total = SHARDS[split]
    count = min(max_shards, total) if max_shards is not None else total
    hf_split = HF_SPLITS[split]
    missing = MISSING_SHARDS.get(split, set())
    return [
        f"{DATA_PREFIX}/{hf_split}/contacts_v1-{idx:05d}-of-{total:05d}.parquet"
        for idx in range(count)
        if idx not in missing
    ]


class ThinkMaskedProcessor(BatchProcessor[dict, dict]):
    """Tokenize raw documents and mask causal targets whose token id is ``<think>``."""

    def __init__(self, tokenizer: MarinTokenizer, *, text_key: str, enforce_bos: bool, enforce_eos: bool):
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.tokenizer_processor = BatchTokenizer(
            tokenizer,
            text_field=text_key,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
            long_string_workaround=True,
        )

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        tokenized = self.tokenizer_processor(batch)
        out: list[dict] = []
        for example in tokenized:
            ids = np.asarray(example["input_ids"], dtype=np.int32)
            # Levanter's causal LM loss at position i predicts token i+1. To
            # mask target ``<think>`` tokens, zero the weight at the preceding
            # position. The final position has no in-document target.
            predicts_not_think = np.roll(ids, -1) != THINK_TOKEN_ID
            predicts_not_think[-1] = False
            out.append(
                {
                    "input_ids": ids,
                    "loss_weights": predicts_not_think.astype(np.float32),
                }
            )
        return out

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": self.tokenizer.vocab_size,
            "text_key": self.text_key,
            "think_token_id": THINK_TOKEN_ID,
            "mask_policy": "causal_target_token_id_ne_think",
            "tokenizer_metadata": self.tokenizer_processor.metadata,
        }

    @property
    def output_exemplar(self) -> dict:
        return _cache_exemplar()

    @property
    def num_cpus(self) -> int:
        return self.tokenizer_processor.num_cpus

    @property
    def num_gpus(self) -> int:
        return 0


@dataclass(frozen=True)
class ThinkMaskedTextFormat(LmDatasetFormatBase):
    """Levanter format for exp126 raw document rows plus masked loss weights."""

    text_key: str = "document"

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        return ThinkMaskedProcessor(
            tokenizer,
            text_key=self.text_key,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
        )


def _resolve_tokenizer(tokenizer_name: str) -> MarinTokenizer:
    """Load a tokenizer, honoring ``repo@revision`` shorthand."""
    if "@" not in tokenizer_name:
        return load_marin_tokenizer(tokenizer_name)
    repo_id, revision = tokenizer_name.rsplit("@", 1)
    local_path = snapshot_download(repo_id=repo_id, revision=revision)
    return load_marin_tokenizer(local_path)


def _cache_exemplar() -> dict:
    return {
        "input_ids": np.zeros((0,), dtype=np.int32),
        "loss_weights": np.zeros((0,), dtype=np.float32),
    }


def build_split(
    *,
    split: str,
    output_root: str,
    tokenizer_name: str,
    max_shards: int | None,
    max_rows: int | None,
) -> None:
    """Build one split under ``output_root/<split>``."""
    tokenizer = _resolve_tokenizer(tokenizer_name)
    urls = shard_urls(split, max_shards=max_shards)
    source = UrlDataSource(urls, columns=["document"])
    if max_rows is not None:
        source = FirstRowsShardedDataSource(source, max_rows)
    cache_dir = f"{output_root.rstrip('/')}/{split}"
    build_lm_dataset_cache(
        cache_dir,
        source,
        ThinkMaskedTextFormat(),
        tokenizer,
        options=CacheOptions.default(),
        enforce_eos=True,
    )
    print(f"built {split} cache: {cache_dir}")


def finalize_split_sharded_ledger(*, split: str, output_root: str) -> None:
    """Finalize an interrupted build by writing a sharded top-level ledger.

    Levanter's distributed cache builder writes one complete shard cache per
    source shard before materializing the top-level cache. If a coordinator dies
    after shard caches are complete but before ``shard_ledger.json`` is written,
    this creates a sharded top-level ledger that points at those completed shard
    caches without recopying all token arrays.
    """
    cache_dir = f"{output_root.rstrip('/')}/{split}"
    fs, fs_path = fsspec.core.url_to_fs(cache_dir)
    ledger_paths = sorted(fs.glob(f"{fs_path.rstrip('/')}/*/shard_ledger.json"))
    shard_paths = [fs.unstrip_protocol(posixpath.dirname(path)) for path in ledger_paths]
    if not shard_paths:
        raise FileNotFoundError(f"No completed shard ledgers found below {cache_dir}")
    consolidate_shard_cache_ledgers(
        shard_cache_paths=shard_paths,
        output_path=cache_dir,
        exemplar=_cache_exemplar(),
        metadata=CacheMetadata.empty(),
    )
    print(f"finalized {split} sharded ledger: {cache_dir} shards={len(shard_paths):,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/2026.07.29.2",
        help="Destination cache root. Splits are written below this path.",
    )
    parser.add_argument("--split", choices=sorted(SHARDS), action="append", help="Split(s) to build; default train+validation.")
    parser.add_argument("--tokenizer", default=CONTACTS_TOKENIZER)
    parser.add_argument("--max-shards", type=int, default=None, help="Use only the first N source shards for smoke tests.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use only the first N rows for smoke tests.")
    parser.add_argument(
        "--finalize-sharded-ledger",
        action="store_true",
        help="Write a top-level sharded ledger from completed per-shard caches instead of rebuilding data.",
    )
    args = parser.parse_args(argv)

    splits = args.split or ["train", "validation"]
    for split in splits:
        if args.finalize_sharded_ledger:
            finalize_split_sharded_ledger(split=split, output_root=args.output_root)
            continue
        build_split(
            split=split,
            output_root=args.output_root,
            tokenizer_name=args.tokenizer,
            max_shards=args.max_shards,
            max_rows=args.max_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
