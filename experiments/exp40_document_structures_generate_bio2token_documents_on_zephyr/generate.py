# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble ``bio2token-v1`` documents from structures.

Ties the pieces together: parse (gemmi) -> adapt (canonical tensors) ->
tokenize (pure-PyTorch bio2token encoder, bucketed + batched on the target
device) -> assemble the document string + metadata row.

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
from vocab import NAME, build_document, residue_token

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


def _residue_tokens(parsed: ParsedStructure) -> list[str]:
    return [residue_token(one, res.res_type)
            for res, one in zip(parsed.residues, parsed.sequence, strict=True)]


def build_row(parsed: ParsedStructure, codes: list[int]) -> dict[str, Any]:
    """Assemble one document-row dict from a parsed structure + its atom codes."""
    document = build_document(_residue_tokens(parsed), codes)
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
    the document exceeds ``max_context`` tokens (a designed-in filter — a
    structure that doesn't fit the budget is skipped, not truncated mid-molecule
    which would corrupt it). The tokenization is bucketed + batched on ``device``
    (see ``tokenizer.py``), so one batched forward pass covers all structures
    that share a length bucket.
    """
    if not parsed_structures:
        return []
    coords = [to_bio2token_batch(p, add_batch_dim=False)["structure"]
              for p in parsed_structures]
    dev = get_device(device)
    token_tensors = tokenize_structures(
        coords, get_model(device), device=dev, buckets=buckets,
        max_batch=max_batch, max_batch_tokens=max_batch_tokens)

    rows: list[dict[str, Any] | None] = []
    for parsed, toks in zip(parsed_structures, token_tensors, strict=True):
        row = build_row(parsed, toks.tolist())
        if max_context is not None and row["num_tokens"] > max_context:
            rows.append(None)
        else:
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

    Returns a metadata row dict, or ``None`` if the document exceeds
    ``max_context`` tokens. Parse/adapt/tokenize failures propagate (fail-loud).
    """
    parsed = parse_structure(source, entry_id=entry_id)
    return generate_documents([parsed], device=device, max_context=max_context)[0]
