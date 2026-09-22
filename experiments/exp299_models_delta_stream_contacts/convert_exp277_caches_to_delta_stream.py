# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert exp277's four regional contacts-v1 token caches to delta-stream V2.

The source caches already contain the exact decontaminated native and
ProteinMPNN-redesigned documents used by exp277. Converting them in place
avoids recomputing contacts or streaming the raw corpora across clouds.
"""

import argparse
import functools
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import fsspec
import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, TreeCache
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

# Pinned contacts-v1 tokenizer contract used by exp277.
CV1_EOS = 1
CV1_DOC_TYPE = 2
CV1_N_TERM = 3
CV1_C_TERM = 4
CV1_CONTACT = 5
CV1_BEGIN_SEQUENCE = 8
CV1_BEGIN_STATEMENTS = 9
CV1_END = 10
CV1_AA_FIRST = 86
CV1_AA_LAST = 105
CV1_POSITION_FIRST = 143
CV1_UNKNOWN_AA = 2844
CV1_NUM_POSITIONS = 2000

# Delta-stream V2 contract. IDs through 2079 are byte-for-byte identical to
# exp299. The extension above 2079 is reserved for offsets 1025..1999 so the
# complete exp277 corpus can be represented without dropping long-range pairs.
STOP_TOKEN_ID = 0
AA_BASE_TOKEN_ID = 1
UNKNOWN_AA_TOKEN_ID = 24
DOC_START_TOKEN_ID = 21
CONTACTS_BEGIN_TOKEN_ID = 22
DOC_END_TOKEN_ID = 23
DELTA_BASE_TOKEN_ID = 32
LEGACY_MAX_ABS_DELTA = 1024
MAX_ABS_DELTA = CV1_NUM_POSITIONS - 1
EXTENDED_NEGATIVE_BASE = DELTA_BASE_TOKEN_ID + 2 * LEGACY_MAX_ABS_DELTA
EXTENDED_POSITIVE_BASE = EXTENDED_NEGATIVE_BASE + (MAX_ABS_DELTA - LEGACY_MAX_ABS_DELTA)
VOCAB_SIZE = EXTENDED_POSITIVE_BASE + (MAX_ABS_DELTA - LEGACY_MAX_ABS_DELTA)
MAX_DOCUMENT_TOKENS = 8192

OUTPUT_ROOT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/"
    "exp277_full_epoch_documents/2026.09.22.1"
)
SMOKE_OUTPUT_ROOT = f"{OUTPUT_ROOT}-smoke"


@dataclass(frozen=True)
class Corpus:
    """One immutable exp277 source cache."""

    name: str
    cache: str
    documents: int
    source_shards: int


CORPORA = {
    corpus.name: corpus
    for corpus in (
        Corpus(
            "native-afdb",
            "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
            "tokenized/contacts_v1/afdb/2026.08.14/train",
            3_963_003,
            2_067,
        ),
        Corpus(
            "native-esm",
            "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
            "tokenized/contacts_v1/esm/2026.08.14/train",
            65_553_178,
            3_338,
        ),
        Corpus(
            "mpnn-afdb",
            "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
            "tokenized/mpnn-afdb/2026.09.09.1/train",
            31_702_680,
            199,
        ),
        Corpus(
            "mpnn-esm",
            "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
            "tokenized/mpnn-esm/2026.09.09.1/train",
            130_872_044,
            3_338,
        ),
    )
}


@dataclass(frozen=True)
class ParsedContactsV1:
    """Canonical sequence and undirected contacts recovered from one document."""

    amino_acids: tuple[int, ...]
    contacts: tuple[tuple[int, int], ...]


def delta_to_token(delta: int) -> int:
    """Encode a signed nonzero offset while preserving exp299's original IDs."""
    magnitude = abs(delta)
    if delta == 0 or magnitude > MAX_ABS_DELTA:
        raise ValueError(f"delta {delta} is outside the supported range")
    if magnitude <= LEGACY_MAX_ABS_DELTA:
        offset = magnitude - 1 if delta < 0 else LEGACY_MAX_ABS_DELTA + magnitude - 1
        return DELTA_BASE_TOKEN_ID + offset
    extension_offset = magnitude - LEGACY_MAX_ABS_DELTA - 1
    return (EXTENDED_NEGATIVE_BASE if delta < 0 else EXTENDED_POSITIVE_BASE) + extension_offset


def token_to_delta(token_id: int) -> int:
    """Inverse of :func:`delta_to_token`."""
    if DELTA_BASE_TOKEN_ID <= token_id < DELTA_BASE_TOKEN_ID + LEGACY_MAX_ABS_DELTA:
        return -(token_id - DELTA_BASE_TOKEN_ID + 1)
    positive_start = DELTA_BASE_TOKEN_ID + LEGACY_MAX_ABS_DELTA
    if positive_start <= token_id < EXTENDED_NEGATIVE_BASE:
        return token_id - positive_start + 1
    if EXTENDED_NEGATIVE_BASE <= token_id < EXTENDED_POSITIVE_BASE:
        return -(token_id - EXTENDED_NEGATIVE_BASE + LEGACY_MAX_ABS_DELTA + 1)
    if EXTENDED_POSITIVE_BASE <= token_id < VOCAB_SIZE:
        return token_id - EXTENDED_POSITIVE_BASE + LEGACY_MAX_ABS_DELTA + 1
    raise ValueError(f"token {token_id} is not a delta token")


def _position_index(token_id: int) -> int:
    index = token_id - CV1_POSITION_FIRST
    if not 0 <= index < CV1_NUM_POSITIONS:
        raise ValueError(f"token {token_id} is not a contacts-v1 position")
    return index


def parse_contacts_v1_ids(input_ids: Sequence[int]) -> ParsedContactsV1:
    """Recover canonical N-to-C sequence order and contacts from exp277 IDs."""
    ids = [int(token_id) for token_id in input_ids]
    if ids and ids[-1] == CV1_EOS:
        ids.pop()
    if len(ids) < 6 or ids[0] != CV1_DOC_TYPE or ids[-1] != CV1_END:
        raise ValueError("malformed contacts-v1 document framing")
    try:
        sequence_begin = ids.index(CV1_BEGIN_SEQUENCE)
        statements_begin = ids.index(CV1_BEGIN_STATEMENTS, sequence_begin + 1)
    except ValueError as exc:
        raise ValueError("contacts-v1 section marker is missing") from exc
    if sequence_begin != 1:
        raise ValueError(f"unexpected sequence marker index {sequence_begin}")

    sequence_tokens = ids[sequence_begin + 1 : statements_begin]
    if len(sequence_tokens) % 2:
        raise ValueError("sequence section does not contain two-token statements")
    n_term: int | None = None
    c_term: int | None = None
    amino_acid_by_position: dict[int, int] = {}
    for marker, value in zip(sequence_tokens[::2], sequence_tokens[1::2], strict=True):
        position = _position_index(value if marker in {CV1_N_TERM, CV1_C_TERM} else marker)
        if marker == CV1_N_TERM:
            if n_term is not None:
                raise ValueError("multiple N-terminal statements")
            n_term = position
        elif marker == CV1_C_TERM:
            if c_term is not None:
                raise ValueError("multiple C-terminal statements")
            c_term = position
        elif CV1_AA_FIRST <= value <= CV1_AA_LAST or value == CV1_UNKNOWN_AA:
            if position in amino_acid_by_position:
                raise ValueError(f"duplicate amino-acid position {position}")
            amino_acid_by_position[position] = (
                UNKNOWN_AA_TOKEN_ID if value == CV1_UNKNOWN_AA else value - CV1_AA_FIRST + AA_BASE_TOKEN_ID
            )
        else:
            raise ValueError(f"malformed sequence statement ({marker}, {value})")
    if n_term is None or c_term is None or not amino_acid_by_position:
        raise ValueError("sequence section is missing termini or amino acids")

    length = len(amino_acid_by_position)
    expected_positions = tuple((n_term + offset) % CV1_NUM_POSITIONS for offset in range(length))
    if expected_positions[-1] != c_term or set(expected_positions) != set(amino_acid_by_position):
        raise ValueError("termini and amino-acid positions do not form one canonical chain")
    amino_acids = tuple(amino_acid_by_position[position] for position in expected_positions)
    sequence_index = {position: index for index, position in enumerate(expected_positions)}

    structure_tokens = ids[statements_begin + 1 : -1]
    if len(structure_tokens) % 3:
        raise ValueError("structure section does not contain three-token contact statements")
    contacts: set[tuple[int, int]] = set()
    for contact_token, left_token, right_token in zip(
        structure_tokens[::3], structure_tokens[1::3], structure_tokens[2::3], strict=True
    ):
        if contact_token != CV1_CONTACT:
            raise ValueError(f"unexpected structure token {contact_token}")
        left_position = _position_index(left_token)
        right_position = _position_index(right_token)
        if left_position not in sequence_index or right_position not in sequence_index:
            raise ValueError("contact endpoint is outside the emitted sequence")
        left = sequence_index[left_position]
        right = sequence_index[right_position]
        if left == right:
            raise ValueError("self-contact in source document")
        pair = (min(left, right), max(left, right))
        if pair in contacts:
            raise ValueError(f"duplicate source contact {pair}")
        contacts.add(pair)
    return ParsedContactsV1(amino_acids=amino_acids, contacts=tuple(sorted(contacts)))


def build_delta_stream_ids(parsed: ParsedContactsV1) -> list[int]:
    """Serialize one canonical parsed document as bidirectional delta-stream V2."""
    contacts_by_residue: list[list[int]] = [[] for _ in parsed.amino_acids]
    for left, right in parsed.contacts:
        contacts_by_residue[left].append(right - left)
        contacts_by_residue[right].append(left - right)
    contact_ids: list[int] = []
    for deltas in contacts_by_residue:
        contact_ids.extend(delta_to_token(delta) for delta in sorted(deltas))
        contact_ids.append(STOP_TOKEN_ID)
    token_ids = [
        DOC_START_TOKEN_ID,
        *parsed.amino_acids,
        CONTACTS_BEGIN_TOKEN_ID,
        *contact_ids,
        DOC_END_TOKEN_ID,
    ]
    if len(token_ids) > MAX_DOCUMENT_TOKENS:
        raise ValueError(f"V2 document has {len(token_ids)} tokens, above {MAX_DOCUMENT_TOKENS}")
    return token_ids


def parse_delta_stream_ids(token_ids: Sequence[int]) -> ParsedContactsV1:
    """Strictly parse a complete V2 document for conversion-time round trips."""
    ids = [int(token_id) for token_id in token_ids]
    if len(ids) < 4 or ids[0] != DOC_START_TOKEN_ID or ids[-1] != DOC_END_TOKEN_ID:
        raise ValueError("malformed delta-stream framing")
    try:
        contacts_begin = ids.index(CONTACTS_BEGIN_TOKEN_ID, 1)
    except ValueError as exc:
        raise ValueError("missing CONTACTS_BEGIN") from exc
    amino_acids = tuple(ids[1:contacts_begin])
    valid_amino_acids = set(range(AA_BASE_TOKEN_ID, 21)) | {UNKNOWN_AA_TOKEN_ID}
    if not amino_acids or any(token_id not in valid_amino_acids for token_id in amino_acids):
        raise ValueError("invalid amino-acid prefix")
    contacts: set[tuple[int, int]] = set()
    residue = 0
    for token_id in ids[contacts_begin + 1 : -1]:
        if token_id == STOP_TOKEN_ID:
            residue += 1
            continue
        if residue >= len(amino_acids):
            raise ValueError("delta appears after the final residue segment")
        partner = residue + token_to_delta(token_id)
        if not 0 <= partner < len(amino_acids) or partner == residue:
            raise ValueError("delta endpoint is outside the sequence")
        contacts.add((min(residue, partner), max(residue, partner)))
    if residue != len(amino_acids):
        raise ValueError(f"expected {len(amino_acids)} STOP tokens, found {residue}")
    return ParsedContactsV1(amino_acids=amino_acids, contacts=tuple(sorted(contacts)))


def convert_record(
    source: Mapping[str, np.ndarray],
    *,
    corpus: str,
    source_shard_index: int,
    source_row: int,
    source_global_row: int,
    verify_round_trip: bool,
) -> dict[str, object]:
    """Convert and validate one cache record."""
    input_ids = np.asarray(source["input_ids"], dtype=np.int32)
    parsed = parse_contacts_v1_ids(input_ids)
    token_ids = build_delta_stream_ids(parsed)
    if verify_round_trip and parse_delta_stream_ids(token_ids) != parsed:
        raise ValueError(f"round-trip mismatch at {corpus}:{source_global_row}")
    max_abs_delta = max((right - left for left, right in parsed.contacts), default=0)
    uses_extended_delta = max_abs_delta > LEGACY_MAX_ABS_DELTA
    digest = hashlib.sha256(np.asarray(token_ids, dtype=np.int32).tobytes()).hexdigest()
    return {
        "corpus": corpus,
        "source_shard_index": source_shard_index,
        "source_row": source_row,
        "source_global_row": source_global_row,
        "seq_len": len(parsed.amino_acids),
        "contacts_undirected": len(parsed.contacts),
        "source_token_count": len(input_ids),
        "token_count": len(token_ids),
        "max_abs_delta": max_abs_delta,
        "uses_extended_delta": uses_extended_delta,
        "vocab_size": VOCAB_SIZE,
        "token_ids": token_ids,
        "token_sha256": digest,
    }


def convert_work_shard(
    items: Iterator[Mapping[str, object]],
    _shard_info: object,
    *,
    max_documents_per_source_shard: int | None,
    batch_size: int,
    verify_round_trip: bool,
) -> Iterable[dict[str, object]]:
    """Convert one or more source-cache shard ranges, failing loudly on any row."""
    for item in items:
        cache_path = str(item["cache"])
        start = int(item["start"])
        rows = int(item["rows"])
        if max_documents_per_source_shard is not None:
            rows = min(rows, max_documents_per_source_shard)
        cache = TreeCache.load(cache_path, {"input_ids": np.zeros((1,), dtype=np.int32)})
        for batch_start in range(start, start + rows, batch_size):
            batch_stop = min(start + rows, batch_start + batch_size)
            batch = cache.get_batch_sync(slice(batch_start, batch_stop))
            for offset, source in enumerate(batch):
                global_row = batch_start + offset
                yield convert_record(
                    source,
                    corpus=str(item["corpus"]),
                    source_shard_index=int(item["source_shard_index"]),
                    source_row=global_row - start,
                    source_global_row=global_row,
                    verify_round_trip=verify_round_trip,
                )


def source_work_items(
    corpus: Corpus, *, start_source_shard: int, max_source_shards: int | None
) -> list[dict[str, object]]:
    """Read and validate the source ledger, then return one item per cache shard."""
    ledger = CacheLedger.load(corpus.cache)
    if ledger.layout != "sharded" or not ledger.is_finished:
        raise ValueError(f"{corpus.name}: source cache is not a completed sharded cache")
    if ledger.total_num_rows != corpus.documents:
        raise ValueError(f"{corpus.name}: {ledger.total_num_rows} rows, expected {corpus.documents}")
    if len(ledger.finished_shards) != corpus.source_shards:
        raise ValueError(f"{corpus.name}: {len(ledger.finished_shards)} shards, expected {corpus.source_shards}")
    items: list[dict[str, object]] = []
    start = 0
    for source_shard_index, shard_name in enumerate(ledger.finished_shards):
        rows = int(ledger.shard_rows[shard_name])
        items.append(
            {
                "corpus": corpus.name,
                "cache": corpus.cache,
                "source_shard": shard_name,
                "source_shard_index": source_shard_index,
                "start": start,
                "rows": rows,
            }
        )
        start += rows
    if start != corpus.documents:
        raise ValueError(f"{corpus.name}: shard rows sum to {start}, expected {corpus.documents}")
    stop = None if max_source_shards is None else start_source_shard + max_source_shards
    return items[start_source_shard:stop]


def write_manifest(
    *,
    corpus: Corpus,
    output_root: str,
    work_items: list[dict[str, object]],
    max_documents_per_source_shard: int | None,
) -> None:
    """Record immutable source and conversion parameters beside output shards."""
    manifest = {
        "corpus": corpus.name,
        "source_cache": corpus.cache,
        "expected_source_documents": corpus.documents,
        "expected_source_shards": corpus.source_shards,
        "selected_source_shards": len(work_items),
        "max_documents_per_source_shard": max_documents_per_source_shard,
        "output_root": output_root,
        "vocab_size": VOCAB_SIZE,
        "legacy_max_abs_delta": LEGACY_MAX_ABS_DELTA,
        "max_abs_delta": MAX_ABS_DELTA,
    }
    with fsspec.open(f"{output_root.rstrip('/')}/source-manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


def run(args: argparse.Namespace) -> None:
    corpus = CORPORA[args.corpus]
    work_items = source_work_items(
        corpus,
        start_source_shard=args.start_source_shard,
        max_source_shards=args.max_source_shards,
    )
    if not work_items:
        raise ValueError("source-shard selection is empty")
    output_root = f"{args.output_root.rstrip('/')}/{corpus.name}"
    rows = Dataset.from_list(work_items).reshard(len(work_items)).map_shard(
        functools.partial(
            convert_work_shard,
            max_documents_per_source_shard=args.max_documents_per_source_shard,
            batch_size=args.batch_size,
            verify_round_trip=args.verify_round_trip,
        )
    )
    output = rows.write_parquet(f"{output_root}/shard-{{shard:05d}}-of-{{total:05d}}.parquet")
    context = ZephyrContext(
        max_workers=min(args.max_workers, len(work_items)),
        resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_memory, disk=args.worker_disk),
        coordinator_resources=ResourceConfig(cpu=1, ram="8GB", disk="16GB"),
        name=f"exp299-exp277-to-v2-{corpus.name}",
        chunk_storage_prefix=f"{output_root}/_zephyr_chunks",
        max_execution_retries=2,
    )
    context.execute(output)
    write_manifest(
        corpus=corpus,
        output_root=output_root,
        work_items=work_items,
        max_documents_per_source_shard=args.max_documents_per_source_shard,
    )
    print(f"converted {corpus.name} to {output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--start-source-shard", type=int, default=0)
    parser.add_argument("--max-source-shards", type=int)
    parser.add_argument("--max-documents-per-source-shard", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP299_CONVERT_MAX_WORKERS", "256")))
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="8GB")
    parser.add_argument("--worker-disk", default="16GB")
    parser.add_argument("--verify-round-trip", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.start_source_shard < 0:
        parser.error("--start-source-shard must be nonnegative")
    if args.max_source_shards is not None and args.max_source_shards <= 0:
        parser.error("--max-source-shards must be positive")
    if args.max_documents_per_source_shard is not None and args.max_documents_per_source_shard <= 0:
        parser.error("--max-documents-per-source-shard must be positive")
    run(args)


if __name__ == "__main__":
    main()
