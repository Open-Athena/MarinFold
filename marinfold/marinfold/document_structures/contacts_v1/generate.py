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
3. Emit the sequence section — one ``<pX> <AA>`` statement per
   residue plus one ``<n-term>`` and one ``<c-term>`` statement — in
   random order.
4. Emit the structure section — select the N strongest contacts (N
   chosen to fill the context-length budget, dropping the weakest if
   they don't all fit), then emit ``<contact> <pX> <pY>``
   statements for them in *random* order, each pair's order coin-flipped.

The pure builder :func:`build_document` takes already-computed residues
and contacts, so it (and its determinism / truncation / ordering) can be
unit-tested without pyconfind. :func:`generate_document` /
:func:`generate_documents` wire pyconfind in front of it.

Optionally the structure section is augmented with ``<think>`` (pause)
tokens when ``GenerationConfig(think=True)`` — the contacts-v1 analog of
issue #34 (which added them to contacts-and-distances-v2). ``<think>``
runs are placed *between* ``<contact>`` statements (never inside one),
their total count subtracted from the token budget so documents still fit
and end with ``<end>``. The switch defaults to ``False``; when off, the
generator draws no think-related randomness, so its output is byte-for-byte
what it was before this path existed. See ``SPEC.md`` for the distributions.
"""

import hashlib
import math
import random
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .parse import (
    DEFAULT_CIF_COLUMN,
    DEFAULT_ID_COLUMN,
    AnalyzedStructure,
    ChainSegment,
    RawContact,
    ResidueInfo,
    analyze_structure,
    chain_segments,
    iter_analyzed_structures,
    residues_from_sequence,
)
from .vocab import (
    BEGIN_SEQUENCE_TOKEN,
    BEGIN_STRUCTURE_TOKEN,
    CONTACT_TOKEN,
    CONTEXT_LENGTH,
    C_TERM_TOKEN,
    BACKTRACKING_DOC_TYPE_TOKEN,
    DOC_TYPE_TOKEN,
    END_TOKEN,
    NUM_POSITION_INDICES,
    N_TERM_TOKEN,
    SEQUENCE_ONLY_DOC_TYPE_TOKEN,
    THINK_TOKEN,
    position_token,
)


# Token counts the budget arithmetic depends on.
_SEQ_TOKENS_PER_RESIDUE = 2     # <pX> <AA>
# <n-term> <pS>  and  <c-term> <pE>, one pair per protein chain.
_TERMINUS_STATEMENTS_PER_CHAIN = 2
_TERMINUS_TOKENS_PER_STATEMENT = 2
_CONTACT_TOKENS_PER_STATEMENT = 3   # <contact> <pX> <pY>
# <contacts-v1> <begin_sequence> … <begin_statements> … <end>
_FRAME_TOKENS = 4


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for contacts-v1 generation.

    The first four are pyconfind geometry knobs (SPEC defaults — confind's
    C++ defaults, native-only). ``min_seq_separation`` is the minimum
    primary-sequence separation for a pair to count as a contact: residues
    fewer than this many positions apart in the chain (``|i - j| <
    min_seq_separation``) are never contacts (default 6, i.e. 5-or-closer
    pairs are excluded). ``min_contact_degree`` filters out weak contacts
    before selection: any contact with degree below it is never emitted,
    even if there is budget room (pyconfind returns a long tail of
    near-zero-degree contacts). ``num_position_indices`` is the size of the
    position-token space and must match ``vocab.NUM_POSITION_INDICES``; it
    also caps the longest chain we can serialize (one index per residue,
    no collisions). ``sequence_only`` switches the builder to the
    sequence-only document type (``<contacts-v1.sequence_only>``): it emits
    the sequence section only — no structure section, no contacts — and the
    pyconfind geometry / contact-selection knobs above are then ignored.
    ``backtracking`` swaps token 0 to ``<contacts-v1.backtracking>``, marking
    the document as one that may contain ``<retract>`` statements (#175); it
    is mutually exclusive with ``sequence_only``.
    """

    # Retraction-mode doc type (issue #175). Swaps token 0 to
    # ``<contacts-v1.backtracking>``, declaring that this document MAY contain
    # ``<retract>`` statements. It changes nothing else: the generator itself
    # never emits ``<retract>`` (retraction documents are synthesised by the
    # model-in-the-loop corpus job, #159), so this is purely the conditioning
    # signal a model needs to know which mode it is in. Default False keeps
    # every existing corpus byte-identical.
    backtracking: bool = False

    native_only: bool = True
    contact_distance: float = 3.0
    dcut: float = 25.0
    clash_distance: float = 2.0
    min_seq_separation: int = 6
    min_contact_degree: float = 0.001
    num_position_indices: int = NUM_POSITION_INDICES
    assembly: int | str | None = None
    sequence_only: bool = False

    # --- Multi-chain (multimer) documents ---
    # How many protein chains an *input structure* may have before
    # ``generate_document`` rejects it. The default of 1 is the historical
    # single-chain behavior: every corpus before exp222 is monomeric, and a
    # multi-chain AFDB/ESM-Atlas input means something went wrong upstream.
    # Raising it lets a whole complex become one document, with one
    # ``<n-term>``/``<c-term>`` pair per chain (see SPEC.md). This is an
    # input-acceptance policy only -- ``build_document`` serializes whatever
    # chains it is handed.
    max_chains: int = 1
    # Minimum number of unused position indices between one chain's C-terminus
    # and the next chain's N-terminus on the wrap-around ring. Only meaningful
    # for multi-chain documents; a single chain is allowed to occupy the whole
    # ring (which is what the pre-multimer generator did, and what keeps a
    # 2000-residue chain serializable).
    min_chain_gap: int = 1

    # --- <think> (pause) tokens (contacts-v1 analog of #34) ---
    # Master switch. When False (default) the generator draws no
    # think-related randomness and its output is byte-identical to the
    # pre-think generator (existing corpora / checkpoints are unaffected).
    # When True, ``<think>`` runs are inserted between ``<contact>``
    # statements in the structure section, per the distributions below
    # (identical to #34's, so both structures draw from the same laws).
    # Ignored on the ``sequence_only`` path (no structure section).
    think: bool = False
    # P(any think tokens right after <begin_statements>).
    think_initial_prob: float = 0.75
    # Geometric p for the length of the initial run (support >= 1).
    think_initial_geom_p: float = 0.13
    # Uniform range for k2; n_additional_runs = max(int(k2), 0).
    think_additional_count_range: tuple[float, float] = (-4.0, 4.0)
    # Geometric p for the length of each additional run (support >= 1).
    think_run_length_geom_p: float = 0.25


@dataclass(frozen=True)
class EmittedContact:
    """One contact written into a document.

    ``seq_i`` / ``seq_j`` (0-based, ``seq_i < seq_j``) and the matching
    ``resnum`` / ``resname`` fields are in canonical sequence order for
    interpretability. ``pos_i`` / ``pos_j`` are the document position
    indices for ``seq_i`` / ``seq_j``. ``flipped`` records the coin flip:
    when True the document writes ``<contact> <pJ> <pI>`` (j first).
    ``chain_i`` / ``chain_j`` are the author chain ids the two residues
    belong to; they differ exactly when this is an interface contact
    (:attr:`inter_chain`), which is always False for a monomer document.
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
    chain_i: str = ""
    chain_j: str = ""

    @property
    def inter_chain(self) -> bool:
        """True when the two residues sit on different protein chains."""
        return self.chain_i != self.chain_j

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
            "chain_i": self.chain_i,
            "chain_j": self.chain_j,
            "degree": self.degree,
            "flipped": self.flipped,
            "inter_chain": self.inter_chain,
        }


@dataclass(frozen=True)
class GenerationResult:
    """One generated document plus the metadata worth saving alongside it.

    The flat scalars mirror the metadata columns of the published
    ``timodonnell/protein-docs`` datasets (``seq_len``,
    ``contacts_pre_filter``, ``contacts_emitted``, …), plus the
    contact-definition knob that affects those counts
    (``min_seq_separation``) and the multi-chain block (``num_chains``,
    ``chain_lengths``, the per-chain termini, and the interface-contact
    counts). :meth:`metadata_row` is the flat parquet/jsonl row;
    :meth:`summary_dict` is the richer view (full sequence + per-contact
    degrees) for the local ``--summary-out`` JSON.
    """

    entry_id: str
    document: str
    residues: tuple[ResidueInfo, ...]
    seq_len: int
    global_plddt: float
    start_index: int
    n_term_index: int
    c_term_index: int
    min_seq_separation: int
    contacts_pre_filter: int
    contacts_passing_min_degree: int
    contacts_emitted: int
    contacts_excluded: int
    truncated: bool
    # Contact-degree statistics. ``None`` when the protein has no contacts.
    highest_contact_degree: float | None
    lowest_nonzero_contact_degree: float | None
    lowest_included_contact_degree: float | None
    num_tokens: int
    # Count of ``<think>`` tokens emitted (0 unless ``config.think``).
    think_tokens: int = 0
    contacts: tuple[EmittedContact, ...] = field(default_factory=tuple)

    # --- Multi-chain fields (all trivial for a monomer document) ---
    # Per-chain views, in *structure* order (the order the chains appear in
    # the residue list), NOT the randomized ring order. ``n_term_index`` /
    # ``c_term_index`` above are the first chain's, so they keep their old
    # meaning for the single-chain case.
    num_chains: int = 1
    chain_ids: tuple[str, ...] = field(default_factory=tuple)
    chain_lengths: tuple[int, ...] = field(default_factory=tuple)
    n_term_indices: tuple[int, ...] = field(default_factory=tuple)
    c_term_indices: tuple[int, ...] = field(default_factory=tuple)
    # Interface contacts: how many survived the definitional filters, and how
    # many made it into the document. Both are 0 for a monomer.
    contacts_pre_filter_inter_chain: int = 0
    contacts_emitted_inter_chain: int = 0

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
            "min_seq_separation": self.min_seq_separation,
            "contacts_pre_filter": self.contacts_pre_filter,
            "contacts_passing_min_degree": self.contacts_passing_min_degree,
            "contacts_emitted": self.contacts_emitted,
            "contacts_excluded": self.contacts_excluded,
            "truncated": self.truncated,
            "highest_contact_degree": self.highest_contact_degree,
            "lowest_nonzero_contact_degree": self.lowest_nonzero_contact_degree,
            "lowest_included_contact_degree": self.lowest_included_contact_degree,
            "num_tokens": self.num_tokens,
            "think_tokens": self.think_tokens,
            "num_chains": self.num_chains,
            "chain_ids": list(self.chain_ids),
            "chain_lengths": list(self.chain_lengths),
            "n_term_indices": list(self.n_term_indices),
            "c_term_indices": list(self.c_term_indices),
            "contacts_pre_filter_inter_chain": self.contacts_pre_filter_inter_chain,
            "contacts_emitted_inter_chain": self.contacts_emitted_inter_chain,
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


def _fixed_token_cost(
    num_residues: int, num_chains: int = 1, *, sequence_only: bool = False
) -> int:
    """Token cost of the framing + full sequence section (no contacts).

    Every chain contributes its own ``<n-term>`` and ``<c-term>`` statement,
    so the terminus cost scales with ``num_chains`` (it is a constant 4
    tokens for the single-chain case, as before).
    """
    frame_tokens = 3 if sequence_only else _FRAME_TOKENS
    return (
        frame_tokens
        + _SEQ_TOKENS_PER_RESIDUE * num_residues
        + _TERMINUS_STATEMENTS_PER_CHAIN * _TERMINUS_TOKENS_PER_STATEMENT * num_chains
    )


@dataclass(frozen=True)
class ChainLayout:
    """Where one chain ended up on the wrap-around position-index ring.

    ``seq_start`` / ``seq_stop`` are the chain's half-open bounds in the flat
    residue list; ``n_term_index`` / ``c_term_index`` are the position indices
    of its first and last residue. ``ring_slot`` is its place in the
    *randomized* chain order used to walk the ring, which is independent of
    its place in the residue list.
    """

    chain: str
    seq_start: int
    seq_stop: int
    n_term_index: int
    c_term_index: int
    ring_slot: int

    @property
    def length(self) -> int:
        return self.seq_stop - self.seq_start


def _layout_chains(
    rng: random.Random,
    segments: Sequence[ChainSegment],
    num_indices: int,
    min_chain_gap: int,
) -> tuple[list[int], tuple[ChainLayout, ...], int] | None:
    """Place every chain on the position-index ring, disjointly.

    The chains are laid out around a ring of ``num_indices`` positions in a
    random order, separated by random gaps: each chain gets one unbroken run
    of indices (wrapping past the top of the ring as needed), and no two runs
    overlap. Returns the per-residue position index (parallel to the flat
    residue list) and the per-chain layout in *structure* order, or ``None``
    if the chains plus their minimum gaps cannot fit.

    The gaps are a uniformly random composition of the leftover slack into
    ``k`` non-negative parts (stars and bars), each offset by
    ``min_chain_gap``. Chains plus gaps therefore tile the ring exactly, so
    the walk ends back where it started. The returned ``offset`` is where the
    walk began -- the n-terminal index of whichever chain drew ring slot 0,
    and for a monomer simply the chain's n-terminal index.

    **The single-chain case is unchanged from the pre-multimer generator**,
    and deliberately so: with ``k == 1`` the order shuffle and the
    composition draw are both skipped, the minimum gap is not applied (a lone
    chain may occupy the whole ring, which is what keeps a 2000-residue chain
    serializable), and the only RNG draw is the same single
    ``rng.randrange(num_indices)`` that used to pick the n-terminal index.
    Every existing corpus is therefore byte-identical.
    """
    num_chains = len(segments)
    total_residues = sum(segment.length for segment in segments)
    required_gap = 0 if num_chains == 1 else min_chain_gap
    slack = num_indices - total_residues - num_chains * required_gap
    if slack < 0:
        return None

    order = list(range(num_chains))
    gaps = [required_gap] * num_chains
    if num_chains > 1:
        rng.shuffle(order)
        cuts = sorted(rng.randint(0, slack) for _ in range(num_chains - 1))
        previous = 0
        for slot, cut in enumerate(cuts):
            gaps[slot] += cut - previous
            previous = cut
        gaps[-1] += slack - previous
    else:
        gaps[0] += slack

    offset = rng.randrange(num_indices)

    pos_of_seq = [0] * total_residues
    layouts: list[ChainLayout | None] = [None] * num_chains
    cursor = offset
    for ring_slot, segment_index in enumerate(order):
        segment = segments[segment_index]
        for k in range(segment.length):
            pos_of_seq[segment.start + k] = (cursor + k) % num_indices
        layouts[segment_index] = ChainLayout(
            chain=segment.chain,
            seq_start=segment.start,
            seq_stop=segment.stop,
            n_term_index=cursor % num_indices,
            c_term_index=(cursor + segment.length - 1) % num_indices,
            ring_slot=ring_slot,
        )
        cursor += segment.length + gaps[ring_slot]
    return pos_of_seq, tuple(layouts), offset  # type: ignore[arg-type]


def _geometric(rng: random.Random, p: float) -> int:
    """Sample from Geometric(p) with support {1, 2, 3, ...}.

    Uses inverse-CDF on a uniform sample from ``rng`` so the result is
    deterministic in the same RNG stream as the rest of generation.
    Defined for ``p in (0, 1]``; ``p == 1`` always returns 1 (every trial
    succeeds immediately). Ported verbatim from #34's generator so the two
    document structures draw ``<think>`` run lengths from identical laws.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p!r}")
    if p == 1.0:
        return 1
    # rng.random() in [0, 1); 1 - U in (0, 1] avoids log(0). The upper
    # endpoint is inclusive, though: when rng.random() == 0.0, u == 1.0 and
    # ceil(log(1)/log(1-p)) == 0, which would violate the documented support
    # {1, 2, 3, ...}. Clamp to 1 so that boundary draw stays in support (the
    # RNG stream is untouched, so determinism is preserved).
    u = 1.0 - rng.random()
    return max(1, int(math.ceil(math.log(u) / math.log(1.0 - p))))


def _sample_think_overhead(
    rng: random.Random, config: GenerationConfig
) -> tuple[int, list[int]]:
    """Sample ``(k1, additional_run_lengths)`` for one document.

    ``k1`` is the length of the run placed right after
    ``<begin_statements>`` (0 if the ``think_initial_prob`` gate misses).
    ``additional_run_lengths`` are the lengths of the extra runs the caller
    then assigns to random inter-statement slots. Same procedure and
    distributions as #34's ``_sample_think_overhead``.
    """
    if rng.random() < config.think_initial_prob:
        k1 = _geometric(rng, config.think_initial_geom_p)
    else:
        k1 = 0
    lo, hi = config.think_additional_count_range
    k2 = rng.uniform(lo, hi)
    n_additional = max(int(k2), 0)
    additional_lengths = [
        _geometric(rng, config.think_run_length_geom_p) for _ in range(n_additional)
    ]
    return k1, additional_lengths


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
    ``None`` if the structure can't be serialized: fewer than 2 residues, or
    more residues than there are position indices (``config`` /
    ``vocab.NUM_POSITION_INDICES``), or if the framing + sequence section
    alone already exceeds ``context_length``, or -- for a multi-chain
    structure -- if the chains plus their inter-chain gaps don't fit the
    position-index ring.

    ``residues`` must be in sequence order and **grouped by chain**;
    ``contacts`` reference them by 0-based ``seq_i < seq_j`` index. Multiple
    chains are laid out disjointly around the index ring and each gets its
    own ``<n-term>`` / ``<c-term>`` statement (see ``SPEC.md``); a
    single-chain structure produces byte-identical output to the
    pre-multimer generator.
    """
    residues = list(residues)
    num_residues = len(residues)
    num_indices = config.num_position_indices
    if num_residues < 2 or num_residues > num_indices:
        return None
    segments = chain_segments(residues)
    num_chains = len(segments)
    fixed = _fixed_token_cost(
        num_residues, num_chains, sequence_only=config.sequence_only
    )
    if fixed > context_length:
        return None

    rng = random.Random(_generation_seed(entry_id))

    # Think-token overhead is sampled FIRST (mirroring #34) so its count can
    # be subtracted from the contact budget before selection, and so the draw
    # order is a clean prepend to the existing stream. Gated on ``config.think``
    # (and never on the sequence-only path, which has no structure section):
    # when the switch is off this branch draws nothing, so ``start`` / the
    # shuffles / the flips see the exact same RNG stream as before this path
    # existed and the document is byte-identical.
    k1 = 0
    think_run_lengths: list[int] = []
    if config.think and not config.sequence_only:
        k1, think_run_lengths = _sample_think_overhead(rng, config)
    total_think_tokens = k1 + sum(think_run_lengths)

    # Residue numbering: lay every chain out on the wrap-around index ring
    # (for a single chain this is exactly the old "random n-terminal index,
    # then wrap around", drawing the same single random number).
    layout = _layout_chains(rng, segments, num_indices, config.min_chain_gap)
    if layout is None:
        return None
    pos_of_seq, chain_layouts, start = layout
    n_term_index = chain_layouts[0].n_term_index
    c_term_index = chain_layouts[0].c_term_index

    # Sequence section: per-residue assignments + each chain's two termini,
    # all shuffled together. The termini are appended in structure order, so
    # a single-chain document builds the identical pre-shuffle list it always
    # did and lands on the identical shuffle.
    seq_statements: list[tuple[str, ...]] = [
        (position_token(pos_of_seq[k]), f"<{r.resname}>")
        for k, r in enumerate(residues)
    ]
    for chain_layout in chain_layouts:
        seq_statements.append(
            (N_TERM_TOKEN, position_token(chain_layout.n_term_index))
        )
        seq_statements.append(
            (C_TERM_TOKEN, position_token(chain_layout.c_term_index))
        )
    rng.shuffle(seq_statements)

    # Chain membership per residue, for the intra-chain-only sequence
    # separation filter and the interface-contact bookkeeping below.
    chain_of_seq = [0] * num_residues
    for chain_index, segment in enumerate(segments):
        for k in range(segment.start, segment.stop):
            chain_of_seq[k] = chain_index

    # Sequence-only variant: emit the sequence section under the
    # <contacts-v1.sequence_only> doc type and stop -- no structure section,
    # no contacts. The two RNG draws that shape the sequence section (the
    # n-terminal start index, then this shuffle) are exactly the first two
    # draws of the full path, so a sequence-only document's sequence section
    # is byte-identical to the contacts-v1 document the same entry_id +
    # residues would have produced. The contact-statistics fields are not
    # meaningful here and are reported as 0 / None.
    if config.sequence_only:
        if config.backtracking:
            raise ValueError(
                "sequence_only and backtracking are mutually exclusive: a "
                "sequence-only document has no structure section, so there is "
                "nothing it could retract"
            )
        tokens = [SEQUENCE_ONLY_DOC_TYPE_TOKEN, BEGIN_SEQUENCE_TOKEN]
        for statement in seq_statements:
            tokens.extend(statement)
        tokens.append(END_TOKEN)
        return GenerationResult(
            entry_id=entry_id,
            document=" ".join(tokens),
            residues=tuple(residues),
            seq_len=num_residues,
            global_plddt=global_plddt,
            start_index=start,
            n_term_index=n_term_index,
            c_term_index=c_term_index,
            min_seq_separation=config.min_seq_separation,
            contacts_pre_filter=0,
            contacts_passing_min_degree=0,
            contacts_emitted=0,
            contacts_excluded=0,
            truncated=False,
            highest_contact_degree=None,
            lowest_nonzero_contact_degree=None,
            lowest_included_contact_degree=None,
            num_tokens=len(tokens),
            contacts=(),
            num_chains=num_chains,
            chain_ids=tuple(c.chain for c in chain_layouts),
            chain_lengths=tuple(c.length for c in chain_layouts),
            n_term_indices=tuple(c.n_term_index for c in chain_layouts),
            c_term_indices=tuple(c.c_term_index for c in chain_layouts),
        )

    # Sequence-separation filter (definitional): residues fewer than
    # min_seq_separation positions apart in the primary sequence are never
    # contacts, so they're dropped before anything is counted.
    #
    # This is an INTRA-chain rule. It exists to keep trivial local /
    # secondary-structure contacts out of the documents, and "how far apart
    # in the chain" is undefined for two residues on different chains --
    # every interface contact is between residues that are, in any
    # meaningful sense, infinitely far apart in primary sequence. Applying
    # the rule across chains would silently delete most of the interface,
    # which is the entire content of a multimer document. For a single-chain
    # document the chain test is always true and this reduces exactly to the
    # original filter.
    contacts = [
        c for c in contacts
        if chain_of_seq[c.seq_i] != chain_of_seq[c.seq_j]
        or (c.seq_j - c.seq_i) >= config.min_seq_separation
    ]

    # Structure section. These counts/stats are over the post-seq-sep pool.
    # Rank by descending degree (stable sort keeps pyconfind's
    # (seq_i, seq_j) ordering as the deterministic tie-break), then drop
    # contacts below the minimum-degree threshold before picking which
    # survive truncation.
    ordered = sorted(contacts, key=lambda c: -c.degree)
    contacts_pre_filter = len(ordered)
    contacts_pre_filter_inter_chain = sum(
        1 for c in ordered if chain_of_seq[c.seq_i] != chain_of_seq[c.seq_j]
    )
    highest_degree = ordered[0].degree if ordered else None
    lowest_nonzero_degree = ordered[-1].degree if ordered else None

    eligible = [c for c in ordered if c.degree >= config.min_contact_degree]
    contacts_passing = len(eligible)

    # Budget: frame + sequence section fixed; the pre-sampled think tokens
    # are reserved next; the N strongest eligible contacts fill the rest.
    available = context_length - fixed - total_think_tokens
    max_contacts = max(0, available // _CONTACT_TOKENS_PER_STATEMENT)
    n_emit = min(contacts_passing, max_contacts)
    contacts_excluded = contacts_pre_filter - n_emit
    # "Truncated" means a budget overflow dropped an *eligible* contact —
    # not that the min-degree filter removed weak ones.
    truncated = n_emit < contacts_passing

    selected = eligible[:n_emit]
    # Weakest contact that made it in (eligible is still degree-sorted here).
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
            chain_i=ri.chain,
            chain_j=rj.chain,
            degree=c.degree,
            flipped=rng.random() < 0.5,
        ))

    # Assign each additional think run to an inter-statement slot in
    # [0, n_stmts - 1] uniformly at random, with replacement (slot i =
    # "right before emitted[i]"); the initial run sits at slot 0, right after
    # <begin_statements>. Runs landing in the same slot are concatenated. This
    # is the last RNG use, so it stays a clean suffix on the stream; it draws
    # nothing when think is off (``think_run_lengths`` is empty). When there
    # are no contacts, only the initial run can be placed (no statement to
    # anchor the additional runs to) — mirroring #34's no-statement edge case.
    think_at_slot: dict[int, int] = {}
    if k1 > 0:
        think_at_slot[0] = k1
    if emitted and think_run_lengths:
        n_stmts = len(emitted)
        for length in think_run_lengths:
            slot = rng.randint(0, n_stmts - 1)
            think_at_slot[slot] = think_at_slot.get(slot, 0) + length

    doc_type = BACKTRACKING_DOC_TYPE_TOKEN if config.backtracking else DOC_TYPE_TOKEN
    tokens: list[str] = [doc_type, BEGIN_SEQUENCE_TOKEN]
    for statement in seq_statements:
        tokens.extend(statement)
    tokens.append(BEGIN_STRUCTURE_TOKEN)
    think_emitted = 0
    for idx, c in enumerate(emitted):
        n_think = think_at_slot.get(idx, 0)
        if n_think:
            tokens += [THINK_TOKEN] * n_think
            think_emitted += n_think
        first, second = (c.pos_j, c.pos_i) if c.flipped else (c.pos_i, c.pos_j)
        tokens += [CONTACT_TOKEN, position_token(first), position_token(second)]
    if not emitted and k1 > 0:
        # No contacts but the initial run still fired: emit it so the document
        # records the sampled overhead (rare — only for proteins with no
        # above-threshold, seq-sep-respecting contacts). Clamp to the budget
        # headroom so the "never overflow context_length" invariant holds even
        # for a tiny custom context_length: with no contacts, ``fixed`` already
        # equals the full non-think document length, so ``context_length -
        # fixed`` is exactly the remaining think budget.
        k_emit = min(k1, max(0, context_length - fixed))
        tokens += [THINK_TOKEN] * k_emit
        think_emitted += k_emit
    tokens.append(END_TOKEN)

    return GenerationResult(
        entry_id=entry_id,
        document=" ".join(tokens),
        residues=tuple(residues),
        seq_len=num_residues,
        global_plddt=global_plddt,
        start_index=start,
        n_term_index=n_term_index,
        c_term_index=c_term_index,
        min_seq_separation=config.min_seq_separation,
        contacts_pre_filter=contacts_pre_filter,
        contacts_passing_min_degree=contacts_passing,
        contacts_emitted=len(emitted),
        contacts_excluded=contacts_excluded,
        truncated=truncated,
        highest_contact_degree=highest_degree,
        lowest_nonzero_contact_degree=lowest_nonzero_degree,
        lowest_included_contact_degree=lowest_included_degree,
        num_tokens=len(tokens),
        think_tokens=think_emitted,
        contacts=tuple(emitted),
        num_chains=num_chains,
        chain_ids=tuple(c.chain for c in chain_layouts),
        chain_lengths=tuple(c.length for c in chain_layouts),
        n_term_indices=tuple(c.n_term_index for c in chain_layouts),
        c_term_indices=tuple(c.c_term_index for c in chain_layouts),
        contacts_pre_filter_inter_chain=contacts_pre_filter_inter_chain,
        contacts_emitted_inter_chain=sum(1 for c in emitted if c.inter_chain),
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
    num_chains = len(chain_segments(analyzed.residues))
    fixed = _fixed_token_cost(
        num_residues, num_chains, sequence_only=config.sequence_only
    )
    if fixed > context_length:
        warnings.warn(
            f"skipping {analyzed.entry_id}: fixed sequence section needs "
            f"{fixed} tokens > context_length {context_length}",
            stacklevel=2,
        )
        return None
    # A multi-chain structure also has to fit the position-index ring with a
    # gap between chains, which is stricter than the residue count alone.
    # build_document returns None in that case; warn so the caller's ledger
    # can attribute the drop instead of silently losing the structure.
    result = build_document(
        analyzed.entry_id,
        analyzed.residues,
        analyzed.contacts,
        context_length=context_length,
        config=config,
        global_plddt=analyzed.global_plddt,
    )
    if result is None:
        warnings.warn(
            f"skipping {analyzed.entry_id}: {num_chains} chains totalling "
            f"{num_residues} residues do not fit "
            f"{config.num_position_indices} position indices with a "
            f"{config.min_chain_gap}-index gap between chains",
            stacklevel=2,
        )
    return result


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
        assembly=config.assembly,
        rotamer_library=rotamer_library,
        max_chains=config.max_chains,
    )
    return _result_from_analyzed(analyzed, context_length=context_length, config=config)


def generate_sequence_only_document(
    sequence: str,
    *,
    entry_id: str,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
) -> GenerationResult | None:
    """Generate a sequence-only document from a one-letter AA sequence.

    The structure-free entry point a sequence-database job (e.g. UniRef50;
    see exp64) calls per sequence: map the one-letter ``sequence`` to
    residues (:func:`~marinfold.document_structures.contacts_v1.parse.residues_from_sequence`)
    and emit the sequence section only, under the
    ``<contacts-v1.sequence_only>`` doc type. No pyconfind, no contacts.

    ``config`` is forced to ``sequence_only=True`` (so callers may pass a
    plain :class:`GenerationConfig`); ``entry_id`` is the deterministic
    generation seed. Returns ``None`` for sequences that cannot be
    serialized — fewer than 2 residues, or more than ``config`` /
    ``vocab.NUM_POSITION_INDICES`` residues (can't be uniquely indexed
    under wrap-around).
    """
    seq_config = config if config.sequence_only else replace(config, sequence_only=True)
    residues = residues_from_sequence(sequence)
    return build_document(
        entry_id,
        residues,
        (),
        context_length=context_length,
        config=seq_config,
    )


def generate_documents(
    input_path,
    *,
    num_docs: int | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library=None,
    cif_column: str = DEFAULT_CIF_COLUMN,
    id_column: str | None = DEFAULT_ID_COLUMN,
) -> Iterator[GenerationResult]:
    """Yield one :class:`GenerationResult` per input structure (up to ``num_docs``).

    The driving entry point — ``cli.py`` parses args and calls this with
    the assembled :class:`GenerationConfig`. ``input_path`` may be a
    structure file / directory, or a ``.parquet`` shard / directory of
    shards in the afdb-24M layout (structures read from ``cif_column``, ids
    from ``id_column``). Structures that fail to parse, are multi-chain, or
    fall outside the serializable residue range are skipped with a warning.
    """
    produced = 0
    for analyzed in iter_analyzed_structures(
        Path(input_path),
        cif_column=cif_column,
        id_column=id_column,
        native_only=config.native_only,
        contact_distance=config.contact_distance,
        dcut=config.dcut,
        clash_distance=config.clash_distance,
        assembly=config.assembly,
        rotamer_library=rotamer_library,
        max_chains=config.max_chains,
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
