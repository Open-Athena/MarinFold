# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble ``bio2token-v2`` documents from structures.

Ties the pieces together: parse (gemmi) -> adapt (canonical tensors) ->
tokenize (pure-PyTorch bio2token encoder, bucketed + batched on the target
device) -> assemble the self-describing document (a residue sequence section
plus shuffled ``<pN> <atom> <btC>`` atom triples; see ``vocab.py``).

Two surfaces:

- :func:`generate_document` — one structure, one document. The convenience
  path used by the tests.
- :func:`generate_documents` — a *batch* of structures tokenized together.
  This is the worker's path: batching is what makes accelerator inference
  efficient (a single structure starves a TPU). See ``tokenizer.py``.

The model is loaded once per process (cached in a module global), so a batch
run pays the checkpoint load — and, on XLA, the first compile — once, not per
structure.
"""

from typing import Any

from adapt import ParsedStructure, parse_structure, to_bio2token_batch
from model import load_bio2token
from tokenizer import (
    DEFAULT_BUCKETS,
    DEFAULT_MAX_BATCH,
    DEFAULT_MAX_BATCH_TOKENS,
    resolve_device,
    tokenize_structures,
)
from vocab import MAX_POSITION, NAME, build_document

_MODEL_CACHE: dict[str, Any] = {}
_DEVICE_CACHE: dict[str, Any] = {}


def get_device(device: str = "cpu"):
    """Resolve + cache the torch device for ``device`` (initializes torch_xla)."""
    if device not in _DEVICE_CACHE:
        _DEVICE_CACHE[device] = resolve_device(device)
    return _DEVICE_CACHE[device]


def get_model(device: str = "cpu"):
    """Load + cache the bio2token model per (process, device)."""
    if device not in _MODEL_CACHE:
        _MODEL_CACHE[device] = load_bio2token(device=get_device(device))
    return _MODEL_CACHE[device]


def build_row(
    parsed: ParsedStructure,
    residue_index: list[int],
    atom_name: list[str],
    codes: list[int],
) -> dict[str, Any] | None:
    """Assemble one document-row dict from a parsed structure + its per-atom
    (residue index, atom name, code). Returns ``None`` if the chain is longer
    than the position-token range (a designed-in drop, matching the contacts
    structures — such a chain can't be uniquely position-numbered)."""
    if len(parsed.residues) > MAX_POSITION + 1:
        return None
    sequence = [(i, res.name) for i, res in enumerate(parsed.residues)]
    atoms = list(zip(residue_index, atom_name, codes, strict=True))
    document = build_document(sequence, atoms, entry_id=parsed.entry_id)
    return {
        "structure": NAME,
        "entry_id": parsed.entry_id,
        "sequence": parsed.sequence,
        "seq_length": len(parsed.residues),
        "num_atoms": len(codes),
        "num_tokens": document.count(" ") + 1,
        "document": document,
    }


def generate_documents(
    parsed_structures: list[ParsedStructure],
    *,
    device: str = "cpu",
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    max_batch: int = DEFAULT_MAX_BATCH,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    max_context: int | None = None,
) -> list[dict[str, Any] | None]:
    """Tokenize a batch of already-parsed structures and assemble their rows.

    Returns one entry per input structure, in order: a row dict, or ``None`` if
    the chain exceeds the position-token range or the document exceeds
    ``max_context`` tokens (designed-in filters — a structure that doesn't fit
    is skipped, not truncated mid-molecule which would corrupt it). The
    tokenization is bucketed + batched on ``device`` (see ``tokenizer.py``).
    """
    if not parsed_structures:
        return []
    batches = [to_bio2token_batch(p, add_batch_dim=False) for p in parsed_structures]
    coords = [b["structure"] for b in batches]
    token_tensors = tokenize_structures(
        coords, get_model(device), device=get_device(device),
        buckets=buckets, max_batch=max_batch, max_batch_tokens=max_batch_tokens)

    rows: list[dict[str, Any] | None] = []
    for parsed, batch, toks in zip(parsed_structures, batches, token_tensors, strict=True):
        row = build_row(
            parsed,
            batch["residue_index"].tolist(),
            batch["atom_name"],
            toks.tolist(),
        )
        if row is not None and max_context is not None and row["num_tokens"] > max_context:
            row = None
        rows.append(row)
    return rows


def generate_document(
    source,
    *,
    entry_id: str | None = None,
    device: str = "cpu",
    max_context: int | None = None,
) -> dict[str, Any] | None:
    """Parse + adapt + tokenize one structure into a document row.

    Returns a metadata row dict, or ``None`` if the structure is filtered
    (too-long chain / over ``max_context``). Parse/adapt/tokenize failures
    propagate (fail-loud).
    """
    parsed = parse_structure(source, entry_id=entry_id)
    return generate_documents([parsed], device=device, max_context=max_context)[0]
