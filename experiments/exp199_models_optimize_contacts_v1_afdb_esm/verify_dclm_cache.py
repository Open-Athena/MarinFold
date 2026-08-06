# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the exact regional DCLM cache consumed by ``train_dclm_nano.py``."""

import json

from rigging.filesystem import StoragePath, prefix_join, url_to_fs

from train_dclm_nano import DCLM_CACHE_URI, DCLM_TOKENIZER, dclm_tokens


def read_json(path: str) -> dict:
    return json.loads(StoragePath(path).read_text())


def main() -> None:
    info = read_json(prefix_join(DCLM_CACHE_URI, ".executor_info"))
    config = info["config"]
    if config["cache_path"].rstrip("/") != DCLM_CACHE_URI.rstrip("/"):
        raise ValueError("DCLM executor metadata points at a different cache")
    if config["tokenizer"] != DCLM_TOKENIZER:
        raise ValueError(f"Unexpected tokenizer: {config['tokenizer']}")

    ledger_path = prefix_join(DCLM_CACHE_URI, "train/shard_ledger.json")
    ledger = read_json(ledger_path)
    if not ledger.get("is_finished"):
        raise ValueError(f"DCLM ledger is not finished: {ledger_path}")
    if not ledger.get("shard_rows") or ledger.get("total_num_rows", 0) <= 0:
        raise ValueError(f"DCLM ledger has no token rows: {ledger_path}")
    metadata = ledger.get("metadata", {}).get("preprocessor_metadata", {})
    if metadata.get("tokenizer") != DCLM_TOKENIZER:
        raise ValueError(f"Ledger tokenizer does not match {DCLM_TOKENIZER}")

    input_ids_path = prefix_join(DCLM_CACHE_URI, "train/input_ids")
    fs, fs_path = url_to_fs(input_ids_path, use_listings_cache=False)
    if not fs.exists(fs_path):
        raise FileNotFoundError(f"DCLM input_ids store is missing: {input_ids_path}")

    handle = dclm_tokens()
    if handle.adopt_source != DCLM_CACHE_URI or handle.override_path is not None:
        raise ValueError("DCLM must be an adopted artifact, never a buildable pinned recipe")
    cache = handle.artifact_type.raw_load(DCLM_CACHE_URI)
    component = cache.as_component()
    if component.split != "train" or component.pack is not False:
        raise ValueError("DCLM component must read the train split as a continuous token stream")

    print(f"cache: {DCLM_CACHE_URI}")
    print(f"tokenizer: {DCLM_TOKENIZER}")
    print(f"shards: {len(ledger['shard_rows']):,}")
    print(f"ledger rows: {ledger['total_num_rows']:,}")
    print("component: split=train, pack=False (continuous token stream)")
    print("status: complete; training graph has no tokenization step")


if __name__ == "__main__":
    main()
