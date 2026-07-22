# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct contacts-v1 training documents directly from AFDB structures.

This prototype deliberately avoids a materialized document corpus and token
cache. It range-reads only the small manifest columns from the pinned
``timodonnell/afdb-1.6M`` Parquets, fetches each selected mmCIF from its public
GCS URI, calls the canonical contacts-v1 generator, and immediately returns a
token-aligned training :class:`~marinfold.document_structures.documents.Document`.
The model-facing construction is independently configurable, so the same
selected residue/contact example can be compared under different attention and
coordinate layouts.

The iterator is intentionally synchronous and defaults to a one-protein CLI
smoke. Prefetch, distributed sharding, and integration with Levanter's training
loader are follow-up work after document parity is established.
"""

import argparse
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from os import PathLike
from typing import Any, BinaryIO
from urllib.parse import urlencode
from urllib.request import urlopen

import gemmi
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from marinfold.document_structures import read_object_bytes
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    GenerationResult,
    generate_document,
)
from marinfold.document_structures.contacts_v1.training_documents import (
    ContactDocumentStyle,
    ContactTargetScoring,
    DocumentConstructionConfig,
    build_contact_training_document,
)
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH
from marinfold.document_structures.documents import Document

AFDB_1_6M_REPO = "timodonnell/afdb-1.6M"
AFDB_1_6M_REVISION = "a7ededeb7caeaef3892f40f2e4df3bb9236b02f8"
HF_DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
AFDB_MANIFEST_COLUMNS = (
    "entry_id",
    "gcs_uri",
    "split",
    "uniprot_accession",
    "tax_id",
    "organism_name",
    "global_plddt",
    "seq_len",
    "seq_cluster_id",
    "struct_cluster_id",
)


@dataclass(frozen=True)
class AfdbStructureRecord:
    """Small AFDB manifest row pointing at one raw mmCIF structure."""

    entry_id: str
    gcs_uri: str
    split: str
    uniprot_accession: str
    tax_id: int
    organism_name: str
    global_plddt: float
    seq_len: int
    seq_cluster_id: str
    struct_cluster_id: str


@dataclass(frozen=True)
class ContactTrainingExample:
    """Raw-source provenance plus canonical and token-aligned documents."""

    source: AfdbStructureRecord
    generation: GenerationResult
    document: Document


def list_afdb_1_6m_shards(
    *,
    filesystem: HfFileSystem | None = None,
    repo: str = AFDB_1_6M_REPO,
    revision: str = AFDB_1_6M_REVISION,
) -> tuple[str, ...]:
    """List the pinned AFDB-1.6M Parquet shards in stable order."""
    fs = filesystem or HfFileSystem()
    base = f"datasets/{repo}"
    shards: list[str] = []
    for entry in fs.ls(base, detail=True, revision=revision):
        if entry["type"] != "directory":
            continue
        for child in fs.ls(entry["name"], detail=True, revision=revision):
            if child["name"].endswith(".parquet"):
                shards.append(child["name"])
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found for {repo}@{revision}")
    return tuple(sorted(shards))


def iter_afdb_1_6m_records(
    *,
    split: str,
    limit: int | None = None,
    filesystem: HfFileSystem | None = None,
    repo: str = AFDB_1_6M_REPO,
    revision: str = AFDB_1_6M_REVISION,
    batch_size: int = 128,
) -> Iterator[AfdbStructureRecord]:
    """Stream small manifest rows without downloading the inline CIF column."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive or None, got {limit}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    fs = filesystem or HfFileSystem()
    yielded = 0
    for shard_path in list_afdb_1_6m_shards(
        filesystem=fs, repo=repo, revision=revision
    ):
        with fs.open(shard_path, "rb", revision=revision) as source:
            for record in iter_afdb_records_from_parquet(
                source, split=split, batch_size=batch_size
            ):
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def iter_afdb_records_from_parquet(
    source: str | PathLike[str] | BinaryIO,
    *,
    split: str,
    batch_size: int = 128,
) -> Iterator[AfdbStructureRecord]:
    """Read the AFDB manifest projection from one local or open Parquet shard."""
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=list(AFDB_MANIFEST_COLUMNS)
    ):
        for row in batch.to_pylist():
            if row["split"] == split:
                yield _record_from_row(row)


def iter_afdb_1_6m_inline_cifs(
    *,
    limit: int,
    repo: str = AFDB_1_6M_REPO,
) -> Iterator[tuple[AfdbStructureRecord, str]]:
    """Fetch a few inline CIF rows through the HF Dataset Server for a local smoke.

    This path deliberately caps requests at ten rows. It is a credential-free
    correctness probe, not a training source; the pinned Parquet manifest plus
    region-local ``gcs_uri`` fetches remain the scalable path.
    """
    if limit <= 0 or limit > 10:
        raise ValueError(f"HF inline smoke limit must be in [1, 10], got {limit}")
    query = urlencode(
        {
            "dataset": repo,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": limit,
        }
    )
    with urlopen(f"{HF_DATASET_SERVER_ROWS_URL}?{query}") as response:
        payload = json.load(response)
    for wrapped_row in payload["rows"]:
        row = wrapped_row["row"]
        cif = row.get("cif_content")
        if not cif:
            raise ValueError(
                f"Dataset Server row {row.get('entry_id')!r} has no cif_content"
            )
        yield _record_from_row(row), str(cif)


def contact_training_example_from_cif(
    source: AfdbStructureRecord,
    cif: str | bytes,
    *,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    document_config: DocumentConstructionConfig = DocumentConstructionConfig(),
    rotamer_library: Any | None = None,
) -> ContactTrainingExample | None:
    """Generate and tokenize one contacts-v1 example from raw mmCIF bytes."""
    cif_text = cif.decode("utf-8", "replace") if isinstance(cif, bytes) else cif
    structure = gemmi.read_structure_string(cif_text)
    if not structure.name:
        structure.name = source.entry_id
    generation = generate_document(
        structure,
        entry_id=source.entry_id,
        context_length=context_length,
        config=config,
        rotamer_library=rotamer_library,
    )
    if generation is None:
        return None
    return ContactTrainingExample(
        source=source,
        generation=generation,
        document=build_contact_training_document(generation, config=document_config),
    )


def iter_contact_training_examples(
    records: Iterable[AfdbStructureRecord],
    *,
    fetch: Callable[[str], bytes] | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    document_config: DocumentConstructionConfig = DocumentConstructionConfig(),
    rotamer_library: Any | None = None,
) -> Iterator[ContactTrainingExample]:
    """Construct contacts-v1 examples lazily as source records are consumed."""
    fetch_one = fetch or _fetch_object_bytes
    resolved_library = (
        rotamer_library if rotamer_library is not None else contacts_rotamer_library()
    )
    for source in records:
        cif = fetch_one(source.gcs_uri)
        example = contact_training_example_from_cif(
            source,
            cif,
            context_length=context_length,
            config=config,
            document_config=document_config,
            rotamer_library=resolved_library,
        )
        if example is not None:
            yield example


def iter_inline_contact_training_examples(
    *,
    limit: int,
    repo: str = AFDB_1_6M_REPO,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    document_config: DocumentConstructionConfig = DocumentConstructionConfig(),
) -> Iterator[ContactTrainingExample]:
    """Construct a bounded credential-free sample from inline HF CIF rows."""
    rotamer_library = contacts_rotamer_library()
    for source, cif in iter_afdb_1_6m_inline_cifs(limit=limit, repo=repo):
        example = contact_training_example_from_cif(
            source,
            cif,
            context_length=context_length,
            config=config,
            document_config=document_config,
            rotamer_library=rotamer_library,
        )
        if example is not None:
            yield example


@cache
def contacts_rotamer_library() -> Any:
    """Load pyconfind's rotamer library once per document-construction process."""
    from pyconfind import load_library

    try:
        from pyconfind import cached_rotamer_library
    except ImportError:
        from pyconfind.data import cached_rotamer_library

    return load_library(cached_rotamer_library())


def _fetch_object_bytes(uri: str) -> bytes:
    data = read_object_bytes(uri)
    if data is None:
        raise RuntimeError(f"Fetch returned no data for {uri}")
    return data


def _record_from_row(row: dict[str, Any]) -> AfdbStructureRecord:
    return AfdbStructureRecord(
        entry_id=str(row["entry_id"]),
        gcs_uri=str(row["gcs_uri"]),
        split=str(row["split"]),
        uniprot_accession=str(row["uniprot_accession"]),
        tax_id=int(row["tax_id"]),
        organism_name=str(row["organism_name"]),
        global_plddt=float(row["global_plddt"]),
        seq_len=int(row["seq_len"]),
        seq_cluster_id=str(row["seq_cluster_id"]),
        struct_cluster_id=str(row["struct_cluster_id"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--limit", type=int, default=1, help="Proteins to construct (default: 1)."
    )
    parser.add_argument(
        "--source",
        choices=("hf-inline", "gcs"),
        default="hf-inline",
        help="Credential-free HF smoke or pinned manifest + GCS hot path.",
    )
    parser.add_argument("--repo", default=AFDB_1_6M_REPO)
    parser.add_argument("--revision", default=AFDB_1_6M_REVISION)
    parser.add_argument(
        "--document-style",
        choices=tuple(style.value for style in ContactDocumentStyle),
        default=ContactDocumentStyle.CAUSAL_SERIALIZED.value,
        help="Model-facing training-document representation.",
    )
    parser.add_argument(
        "--think-tokens",
        type=int,
        default=0,
        help="Fixed pause positions before full-attention target slots.",
    )
    parser.add_argument(
        "--target-scoring",
        choices=tuple(scoring.value for scoring in ContactTargetScoring),
        default=ContactTargetScoring.ORDERED_TOKENS.value,
        help="Fixed token targets or dynamic unordered-contact matching.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document_config = DocumentConstructionConfig(
        style=ContactDocumentStyle(args.document_style),
        target_scoring=ContactTargetScoring(args.target_scoring),
        think_tokens=args.think_tokens,
    )
    if args.source == "hf-inline":
        if args.split != "train":
            raise ValueError(
                "The HF inline smoke supports --split train only; use --source gcs for other splits"
            )
        examples = iter_inline_contact_training_examples(
            limit=args.limit,
            repo=args.repo,
            document_config=document_config,
        )
    else:
        records = iter_afdb_1_6m_records(
            split=args.split,
            limit=args.limit,
            repo=args.repo,
            revision=args.revision,
        )
        examples = iter_contact_training_examples(
            records,
            document_config=document_config,
        )
    count = 0
    for example in examples:
        print(
            json.dumps(
                {
                    "entry_id": example.source.entry_id,
                    "split": example.source.split,
                    "gcs_uri": example.source.gcs_uri,
                    "seq_len": example.generation.seq_len,
                    "contacts_emitted": example.generation.contacts_emitted,
                    "document_style": document_config.style.value,
                    "target_scoring": document_config.target_scoring.value,
                    "attention": example.document.attention.value,
                    "think_tokens": document_config.think_tokens,
                    "document_tokens": example.generation.num_tokens,
                    "training_tokens": len(example.document),
                    "supervised_tokens": len(example.document.query_positions),
                    "sha1": example.generation.sha1,
                },
                sort_keys=True,
            )
        )
        count += 1
    if count != args.limit:
        raise RuntimeError(f"Constructed {count} examples, expected {args.limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
