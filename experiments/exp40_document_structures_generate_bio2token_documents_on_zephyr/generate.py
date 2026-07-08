# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate one ``bio2token-v1`` document from a structure.

Ties the pieces together: parse (gemmi) -> adapt (canonical tensors) ->
tokenize (pure-PyTorch bio2token encoder) -> assemble the document string +
metadata row. The model is loaded once per process (module global) — the
Zephyr per-worker memoization pattern, so a batch/dir run pays the checkpoint
load once, not per structure.
"""

from typing import Any

from bio2token.data.utils.tokens import AA_TO_TOKEN, RNA_TO_TOKEN

from adapt import ParsedStructure, parse_structure, to_bio2token_batch
from model import load_bio2token
from vocab import NAME, build_document, residue_token

_MODEL_CACHE: dict[str, Any] = {}


def get_model(device: str = "cpu"):
    """Load + cache the bio2token model per (process, device)."""
    if device not in _MODEL_CACHE:
        _MODEL_CACHE[device] = load_bio2token(device=device)
    return _MODEL_CACHE[device]


def _residue_tokens(parsed: ParsedStructure) -> list[str]:
    out = []
    for res, one in zip(parsed.residues, parsed.sequence):
        out.append(residue_token(one, res.res_type))
    return out


def generate_document(
    source,
    *,
    entry_id: str | None = None,
    device: str = "cpu",
    max_context: int | None = None,
) -> dict[str, Any] | None:
    """Parse + adapt + tokenize one structure into a document row.

    Returns a metadata row dict, or ``None`` if the document exceeds
    ``max_context`` tokens (a designed-in filter — a structure that doesn't
    fit the budget is skipped rather than truncated mid-molecule, which would
    corrupt it). Parse/adapt/tokenize failures propagate (fail-loud).
    """
    parsed = parse_structure(source, entry_id=entry_id)
    batch = to_bio2token_batch(parsed)
    structure = batch["structure"].to(device)
    codes = get_model(device).tokenize(structure)[0].tolist()

    document = build_document(_residue_tokens(parsed), codes)
    n_tokens = document.count(" ") + 1
    if max_context is not None and n_tokens > max_context:
        return None

    return {
        "structure": NAME,
        "entry_id": parsed.entry_id,
        "sequence": parsed.sequence,
        "seq_length": len(parsed.residues),
        "num_atoms": len(codes),
        "num_tokens": n_tokens,
        "document": document,
    }
