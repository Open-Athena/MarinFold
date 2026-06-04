# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-v1 document generation.

Library module — the CLI surface lives in ``cli.py`` next door, which
imports :func:`generate_documents` / :func:`generate_document` from here.

The format is defined in ``SPEC.md`` in this directory. One document per
input structure, fully deterministic given the structure's ``entry_id``:

1. Run pyconfind (``parse.analyze_structure``) to get the residue
   sequence and the contacts with contact degree > 0.
2. Pick a random n-terminal index ``start`` in ``[0, 2000)``; number
   residues ``start, start+1, …`` with wrap-around (so the model sees
   the whole index range, not just the low values most proteins reach).
3. Emit the sequence section — one ``<pos-X> <AA>`` statement per
   residue plus one ``<n-term>`` and one ``<c-term>`` statement — in
   random order.
4. Emit the structure section — select the N strongest contacts (N
   chosen to fill the context-length budget, dropping the weakest if
   they don't all fit), then emit ``<contact> <pos-X> <pos-Y>``
   statements for them in *random* order, each pair's order coin-flipped.

The pure builder :func:`build_document` takes already-computed residues
and contacts, so it (and its determinism / truncation / ordering) can be
unit-tested without pyconfind. :func:`generate_document` /
:func:`generate_documents` wire pyconfind in front of it.
"""

import hashlib
import math
import random
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parse import (
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    analyze_structure,
    iter_analyzed_structures,
)
from .vocab import CONTEXT_LENGTH, NUM_POSITION_INDICES


# Token counts the budget arithmetic depends on.
_SEQ_TOKENS_PER_RESIDUE = 2     # <pos-X> <AA>
_TERMINUS_STATEMENTS = 2        # <n-term> <pos-S>  and  <c-term> <pos-E>
_CONTACT_TOKENS_PER_STATEMENT = 3   # <contact> <pos-X> <pos-Y>
# <contacts-v1> <begin-sequence> … <begin-structure> … <end>
_FRAME_TOKENS = 4


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for contacts-v1 generation.

    The first four are pyconfind geometry knobs (SPEC defaults — confind's
    C++ defaults, native-only). ``num_position_indices`` is the size of the
    position-token space and must match ``vocab.NUM_POSITION_INDICES``; it
    also caps the longest chain we can serialize (one index per residue,
    no collisions).
    """

    native_only: bool = True
    contact_distance: float = 3.0
    dcut: float = 25.0
    clash_distance: float = 2.0
    num_position_indices: int = NUM_POSITION_INDICES


@dataclass(frozen=True)
class EmittedContact:
    """One contact written into a document.

    ``seq_i`` / ``seq_j`` (0-based, ``seq_i < seq_j``) and the matching
    ``resnum`` / ``resname`` fields are in canonical sequence order for
    interpretability. ``pos_i`` / ``pos_j`` are the document position
    indices for ``seq_i`` / ``seq_j``. ``flipped`` records the coin flip:
    when True the document writes ``<contact> <pos_j> <pos_i>`` (j first).
    """

    seq_i: int
    seq_j: int
    pos_i: int
    pos_j: int
    resnum_i: int
    resnum_j: int
    resname_i: str
    resname_j: str
    degree: float
    flipped: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq_i": self.seq_i,
            "seq_j": self.seq_j,
            "pos_i": self.pos_i,
            "pos_j": self.pos_j,
            "resnum_i": self.resnum_i,
            "resnum_j": self.resnum_j,
            "resname_i": self.resname_i,
            "resname_j": self.resname_j,
            "degree": self.degree,
            "flipped": self.flipped,
        }


@dataclass(frozen=True)
class GenerationResult:
    """One generated document plus the metadata worth saving alongside it.

    The flat scalars mirror the metadata columns of the published
    ``timodonnell/protein-docs`` datasets (``seq_len``,
    ``contacts_pre_filter``, ``contacts_emitted``, …). :meth:`metadata_row`
    is the flat parquet/jsonl row; :meth:`summary_dict` is the richer view
    (full sequence + per-contact degrees) for the local ``--summary-out``
    JSON.
    """

    entry_id: str
    document: str
    residues: tuple[ResidueInfo, ...]
    seq_len: int
    global_plddt: float
    start_index: int
    n_term_index: int
    c_term_index: int
    contacts_pre_filter: int
    contacts_emitted: int
    contacts_excluded: int
    truncated: bool
    # Contact-degree statistics. ``None`` when the protein has no contacts.
    highest_contact_degree: float | None
    lowest_nonzero_contact_degree: float | None
    lowest_included_contact_degree: float | None
    num_tokens: int
    contacts: tuple[EmittedContact, ...] = field(default_factory=tuple)

    @property
    def sha1(self) -> str:
        """SHA1 of the document string (matches the protein-docs ``sha1`` column)."""
        return hashlib.sha1(self.document.encode()).hexdigest()

    def metadata_row(self) -> dict[str, Any]:
        """Flat row (document + scalar metadata) for the docs parquet/jsonl."""
        return {
            "document": self.document,
            "entry_id": self.entry_id,
            "seq_len": self.seq_len,
            "global_plddt": self.global_plddt,
            "start_index": self.start_index,
            "n_term_index": self.n_term_index,
            "c_term_index": self.c_term_index,
            "contacts_pre_filter": self.contacts_pre_filter,
            "contacts_emitted": self.contacts_emitted,
            "contacts_excluded": self.contacts_excluded,
            "truncated": self.truncated,
            "highest_contact_degree": self.highest_contact_degree,
            "lowest_nonzero_contact_degree": self.lowest_nonzero_contact_degree,
            "lowest_included_contact_degree": self.lowest_included_contact_degree,
            "num_tokens": self.num_tokens,
            "sha1": self.sha1,
        }

    def summary_dict(self) -> dict[str, Any]:
        """Rich per-protein view for the local summary JSON."""
        row = self.metadata_row()
        row.pop("document")
        row["sequence"] = [r.resname for r in self.residues]
        row["contacts"] = [c.as_dict() for c in self.contacts]
        return row


def _generation_seed(entry_id: str) -> int:
    """Deterministic per-entry seed (first 8 sha1 hex digits)."""
    return int(hashlib.sha1(entry_id.encode()).hexdigest()[:8], 16)


def build_document(
    entry_id: str,
    residues: Sequence[ResidueInfo],
    contacts: Sequence[RawContact],
    *,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    global_plddt: float = math.nan,
) -> GenerationResult | None:
    """Build one contacts-v1 document from residues + contacts.

    Pure and deterministic given ``entry_id`` (the RNG seed). Returns
    ``None`` if the chain can't be serialized: fewer than 2 residues, or
    more residues than there are position indices (``config`` /
    ``vocab.NUM_POSITION_INDICES``).

    ``residues`` must be in sequence order; ``contacts`` reference them by
    0-based ``seq_i < seq_j`` index.
    """
    residues = list(residues)
    num_residues = len(residues)
    num_indices = config.num_position_indices
    if num_residues < 2 or num_residues > num_indices:
        return None

    rng = random.Random(_generation_seed(entry_id))

    # Residue numbering: random n-terminal index, then wrap around.
    start = rng.randrange(num_indices)
    pos_of_seq = [(start + k) % num_indices for k in range(num_residues)]
    n_term_index = pos_of_seq[0]
    c_term_index = pos_of_seq[-1]

    # Sequence section: per-residue assignments + the two termini, shuffled.
    seq_statements: list[tuple[str, ...]] = [
        (f"<pos-{pos_of_seq[k]}>", f"<{r.resname}>") for k, r in enumerate(residues)
    ]
    seq_statements.append(("<n-term>", f"<pos-{n_term_index}>"))
    seq_statements.append(("<c-term>", f"<pos-{c_term_index}>"))
    rng.shuffle(seq_statements)

    # Structure section. Rank by descending degree to pick which contacts
    # survive truncation. Stable sort keeps pyconfind's (seq_i, seq_j)
    # ordering as the deterministic tie-break at the truncation boundary.
    ordered = sorted(contacts, key=lambda c: -c.degree)
    contacts_pre_filter = len(ordered)
    highest_degree = ordered[0].degree if ordered else None
    lowest_nonzero_degree = ordered[-1].degree if ordered else None

    # Budget: frame + sequence section fixed; the N strongest contacts
    # fill the rest.
    fixed = (
        _FRAME_TOKENS
        + _SEQ_TOKENS_PER_RESIDUE * num_residues
        + _TERMINUS_STATEMENTS * 2
    )
    available = context_length - fixed
    max_contacts = max(0, available // _CONTACT_TOKENS_PER_STATEMENT)
    n_emit = min(contacts_pre_filter, max_contacts)
    contacts_excluded = contacts_pre_filter - n_emit
    truncated = contacts_excluded > 0

    selected = ordered[:n_emit]
    # Weakest contact that made it in (selected is still degree-sorted here).
    lowest_included_degree = selected[-1].degree if selected else None
    # List the selected contacts in random order — the model should not
    # learn a degree-sorted ordering. (Selection above is by strength; the
    # in-document order is randomized.)
    rng.shuffle(selected)

    emitted: list[EmittedContact] = []
    for c in selected:
        ri, rj = residues[c.seq_i], residues[c.seq_j]
        emitted.append(EmittedContact(
            seq_i=c.seq_i,
            seq_j=c.seq_j,
            pos_i=pos_of_seq[c.seq_i],
            pos_j=pos_of_seq[c.seq_j],
            resnum_i=ri.resnum,
            resnum_j=rj.resnum,
            resname_i=ri.resname,
            resname_j=rj.resname,
            degree=c.degree,
            flipped=rng.random() < 0.5,
        ))

    tokens: list[str] = ["<contacts-v1>", "<begin-sequence>"]
    for statement in seq_statements:
        tokens.extend(statement)
    tokens.append("<begin-structure>")
    for c in emitted:
        first, second = (c.pos_j, c.pos_i) if c.flipped else (c.pos_i, c.pos_j)
        tokens += ["<contact>", f"<pos-{first}>", f"<pos-{second}>"]
    tokens.append("<end>")

    return GenerationResult(
        entry_id=entry_id,
        document=" ".join(tokens),
        residues=tuple(residues),
        seq_len=num_residues,
        global_plddt=global_plddt,
        start_index=start,
        n_term_index=n_term_index,
        c_term_index=c_term_index,
        contacts_pre_filter=contacts_pre_filter,
        contacts_emitted=len(emitted),
        contacts_excluded=contacts_excluded,
        truncated=truncated,
        highest_contact_degree=highest_degree,
        lowest_nonzero_contact_degree=lowest_nonzero_degree,
        lowest_included_contact_degree=lowest_included_degree,
        num_tokens=len(tokens),
        contacts=tuple(emitted),
    )


def _result_from_analyzed(
    analyzed: AnalyzedStructure,
    *,
    context_length: int,
    config: GenerationConfig,
) -> GenerationResult | None:
    """Apply :func:`build_document` to an :class:`AnalyzedStructure`, warning on skip."""
    num_residues = len(analyzed.residues)
    if not (2 <= num_residues <= config.num_position_indices):
        warnings.warn(
            f"skipping {analyzed.entry_id}: {num_residues} residues outside "
            f"[2, {config.num_position_indices}]",
            stacklevel=2,
        )
        return None
    return build_document(
        analyzed.entry_id,
        analyzed.residues,
        analyzed.contacts,
        context_length=context_length,
        config=config,
        global_plddt=analyzed.global_plddt,
    )


def generate_document(
    structure,
    *,
    entry_id: str | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library=None,
) -> GenerationResult | None:
    """Generate one document from a structure file / ``gemmi.Structure``.

    The single-structure entry point future zephyr data jobs can call
    per input. Returns ``None`` (with a warning) for chains that can't be
    serialized; raises ``ValueError`` for unparseable / multi-chain inputs
    (see :func:`~marinfold.document_structures.contacts_v1.parse.analyze_structure`).
    """
    analyzed = analyze_structure(
        structure,
        entry_id=entry_id,
        native_only=config.native_only,
        contact_distance=config.contact_distance,
        dcut=config.dcut,
        clash_distance=config.clash_distance,
        rotamer_library=rotamer_library,
    )
    return _result_from_analyzed(analyzed, context_length=context_length, config=config)


def generate_documents(
    input_path,
    *,
    num_docs: int | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library=None,
) -> Iterator[GenerationResult]:
    """Yield one :class:`GenerationResult` per input structure (up to ``num_docs``).

    The driving entry point — ``cli.py`` parses args and calls this with
    the assembled :class:`GenerationConfig`. Structures that fail to parse,
    are multi-chain, or fall outside the serializable residue range are
    skipped with a warning.
    """
    produced = 0
    for analyzed in iter_analyzed_structures(
        Path(input_path),
        native_only=config.native_only,
        contact_distance=config.contact_distance,
        dcut=config.dcut,
        clash_distance=config.clash_distance,
        rotamer_library=rotamer_library,
    ):
        result = _result_from_analyzed(
            analyzed, context_length=context_length, config=config
        )
        if result is None:
            continue
        yield result
        produced += 1
        if num_docs is not None and produced >= num_docs:
            return
