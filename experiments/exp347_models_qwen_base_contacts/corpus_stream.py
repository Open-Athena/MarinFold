"""Bounded-memory, exactly resumable access to the complete source-document pool.

All source shards are available from the first update. Within a rank's epoch,
source draws are weighted by remaining rows, without replacement; every row in
that rank's disjoint shard assignment is visited once. Each source shuffles its
shards and each 128-row batch. This is a source-balanced streaming shuffle, not a
uniform permutation of all 235M individual rows. Filters are applied after draws,
identically for both renderings, and their cumulative counts are checkpointed.
"""

import copy
import hashlib
import json
import random
from collections.abc import Generator
from typing import Any

import fsspec
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerBase

from common import MAX_LENGTH, SEED, convert_document, split_key

BATCH_ROWS = 128
COUNTERS = ("seen", "validation", "empty_contacts", "too_long", "accepted")


def encode_document(
    row: dict, source: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[dict | None, str]:
    """Apply the paired admission policy, retaining upstream cluster lineage."""
    lineage = row["cluster_key"] if source == "complex" else row["seq_cluster_id"]
    if not lineage:
        raise ValueError(f"Missing lineage in {source}")
    # Keep the pilot's AFDB holdout disjoint from both native and redesigned
    # training sequences. Redesigns carry their parent's seq_cluster_id.
    key = split_key("", lineage)
    if int(key[:8], 16) % 100 == 0:
        return None, "validation"
    doc = convert_document(row["document"])
    if not doc.contacts:
        return None, "empty_contacts"
    result: dict[str, Any] = {
        "entry_id": row["document_id"] if source == "complex" else row["entry_id"],
        "sequence": doc.sequence,
        "n_contacts": len(doc.contacts),
        "source": source,
        "split_key": key,
    }
    for name, prefix, completion in [
        ("contacts_v1", doc.raw_prefix, doc.raw_completion),
        ("prompted", doc.prompted_prefix, doc.prompted_completion),
    ]:
        encoded = tokenizer(
            prefix + completion, add_special_tokens=False, return_offsets_mapping=True
        )
        result[name] = encoded["input_ids"] + [tokenizer.eos_token_id]
        result[name + "_start"] = next(
            i
            for i, (_, end) in enumerate(encoded["offset_mapping"])
            if end > len(prefix)
        )
    if max(len(result[name]) for name in ["contacts_v1", "prompted"]) > MAX_LENGTH:
        return None, "too_long"
    return result, "accepted"


class CorpusStream:
    """Read projected parquet batches with deterministic per-source cursors."""

    def __init__(
        self,
        prefix: str,
        manifest: dict,
        tokenizer: PreTrainedTokenizerBase,
        rank: int,
        world: int,
    ) -> None:
        with fsspec.open(prefix + "/catalog.json", "rb") as handle:
            encoded = handle.read()
        if hashlib.sha256(encoded).hexdigest() != manifest["catalog_sha256"]:
            raise ValueError("Source catalog differs from its committed manifest")
        self.catalog = json.loads(encoded)
        self.catalog_hash = manifest["catalog_sha256"]
        self.tokenizer = tokenizer
        self.rank, self.world = rank, world
        self.sources = sorted(self.catalog)
        if not 0 <= rank < world or any(
            len(s["shards"]) < world for s in self.catalog.values()
        ):
            raise ValueError("Every rank must have source shards in every corpus")
        self.readers: dict[str, Generator[list[dict], None, None]] = {}
        self.batches: dict[str, tuple[int, int, list[dict]]] = {}
        self.files: dict[str, list[dict]] = {}
        self._state: dict = {}
        self.state = {
            "catalog_sha256": self.catalog_hash,
            "rank": rank,
            "world": world,
            "epoch": 0,
            "draw": 0,
            "cursors": {s: {"file": 0, "row": 0, "seen": 0} for s in self.sources},
            "counts": {s: dict.fromkeys(COUNTERS, 0) for s in self.sources},
        }

    @property
    def state(self) -> dict:
        """Return the next-row cursor and cumulative admission accounting."""
        return self._state

    @state.setter
    def state(self, value: dict) -> None:
        if (value["catalog_sha256"], value["rank"], value["world"]) != (
            self.catalog_hash,
            self.rank,
            self.world,
        ):
            raise ValueError(
                "Cannot resume with a changed source catalog or rank layout"
            )
        self.close()
        self._state = copy.deepcopy(value)
        self.files = {}
        for source in self.sources:
            files = self.catalog[source]["shards"].copy()
            random.Random(f"{SEED}:{value['epoch']}:{source}").shuffle(files)
            self.files[source] = files[self.rank :: self.world]
        self.totals = {
            s: sum(p["rows"] for p in files) for s, files in self.files.items()
        }

    def close(self) -> None:
        """Close remote readers when restoring or advancing to another epoch."""
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
        self.batches.clear()

    def _read_batches(
        self, source: str, shard: dict, skip: int
    ) -> Generator[list[dict], None, None]:
        fs, path = fsspec.core.url_to_fs(shard["uri"])
        info = fs.info(path)
        if info["size"] != shard["size"] or info.get("ETag") != shard["etag"]:
            raise ValueError(f"Source changed after its audit: {shard['uri']}")
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for index, batch in enumerate(
                parquet.iter_batches(
                    batch_size=BATCH_ROWS, columns=shard["columns"], use_threads=False
                )
            ):
                if index < skip:
                    continue
                rows = batch.to_pylist()
                random.Random(
                    f"{SEED}:{self.state['epoch']}:{source}:{shard['uri']}:{index}"
                ).shuffle(rows)
                yield rows

    def next_raw(self) -> tuple[str, dict]:
        """Read one original row; used also to test rank coverage and replay."""
        remaining = [
            self.totals[s] - self.state["cursors"][s]["seen"] for s in self.sources
        ]
        total = sum(remaining)
        if total == 0:
            self.state = {
                **self.state,
                "epoch": self.state["epoch"] + 1,
                "draw": 0,
                "cursors": {s: {"file": 0, "row": 0, "seen": 0} for s in self.sources},
            }
            return self.next_raw()
        if min(remaining) < 0:
            raise ValueError("Source cursor has exceeded its audited cardinality")
        draw = random.Random(
            f"{SEED}:{self.rank}:{self.state['epoch']}:{self.state['draw']}"
        ).randrange(total)
        for source, weight in zip(self.sources, remaining, strict=True):
            if draw < weight:
                break
            draw -= weight
        cursor = self.state["cursors"][source]
        files = self.files[source]
        while cursor["row"] == files[cursor["file"]]["rows"]:
            cursor["file"] += 1
            cursor["row"] = 0
        shard = files[cursor["file"]]
        batch_index, offset = divmod(cursor["row"], BATCH_ROWS)
        key = (cursor["file"], batch_index)
        cached = self.batches.get(source)
        if cached is None or cached[:2] != key:
            if cached is None or cached[0] != key[0]:
                old = self.readers.pop(source, None)
                if old is not None:
                    old.close()
                self.readers[source] = self._read_batches(source, shard, batch_index)
            rows = next(self.readers[source])
            self.batches[source] = (*key, rows)
        result = self.batches[source][2][offset]
        cursor["row"] += 1
        cursor["seen"] += 1
        self.state["draw"] += 1
        self.state["counts"][source]["seen"] += 1
        return source, result

    def next(self) -> dict:
        """Tokenize a paired eligible document, accounting for every exclusion."""
        for _ in range(sum(self.totals.values())):
            source, raw = self.next_raw()
            row, disposition = encode_document(raw, source, self.tokenizer)
            self.state["counts"][source][disposition] += 1
            if row is not None:
                return row
        raise ValueError("No eligible training documents in a complete rank epoch")
