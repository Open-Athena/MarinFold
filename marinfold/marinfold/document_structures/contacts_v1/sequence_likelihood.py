# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Any-order amino-acid conditionals from a contacts-v1 model.

A contacts-v1 document opens with the protein's residues as a **randomly
ordered** list of ``<pN> <AA>`` statements, and the current models are trained
with a fresh permutation drawn per example (issue #199's ``-aug`` arm, which
is the default checkpoint). That makes a contacts-v1 checkpoint an *any-order
autoregressive model over amino acids*: condition on every residue but one and
read the distribution over the one left out. It is the same conditional a
masked protein LM computes with a mask token, and it is the object zero-shot
variant-effect benchmarks score.

Nothing here is off-distribution. Under a uniform shuffle any statement can be
last, so asking for ``P(residue i | all the others)`` asks the model something
it was trained on directly.

**One pass, every residue.** The naive readout — one prompt per masked
residue, target statement appended last — costs ``L`` forward passes per
protein per ordering. This module does not do that. A document is an ordinary
causal-LM sequence, so a single teacher-forced pass yields the conditional at
*every* residue at once: at the slot holding a ``<pN>`` token, the next-token
distribution is exactly ``P(amino acid at N | every statement before it)``.
Ensembling over ``K`` orderings therefore costs ``K`` passes, not ``K·L``.

**Context size is the price.** What each residue is conditioned on varies with
where its statement landed in the shuffle: early means few others were
visible, last means all of them. :attr:`AAConditionals.context_sizes` records
this per slot, so callers can select (full-context conditionals, for a
masked-marginals score) or stratify (the whole range, for a context-size
curve). A residue lands in the last 10% of the ordering about ``K/10`` times,
which sets how large ``K`` needs to be for a given context threshold.

Amino-acid columns are the canonical 20 in one-letter alphabetical order
(:data:`AA_ALPHABET`). The document vocab also allows ``<UNK>`` after a
``<pN>``, so the mass on the 20 does not sum to 1; the leftover is reported as
:attr:`AAConditionals.target_mass` and the returned log-probabilities are
renormalized over the 20. That renormalization cancels exactly in any
same-slot log-ratio, so it costs nothing and makes the numbers comparable.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import (
    ONE_LETTER_TO_THREE,
    ResidueInfo,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE_TOKEN,
    BEGIN_STRUCTURE_TOKEN,
    C_TERM_TOKEN,
    N_TERM_TOKEN,
    NUM_POSITION_INDICES,
)
from marinfold.inference.core import Backend

# The canonical 20 in one-letter alphabetical order. This is the column order
# of every array in this module; it is a scoring convention (benchmarks name
# substitutions in one-letter form), deliberately independent of the vocab's
# own three-letter ordering.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TOKENS = tuple(f"<{ONE_LETTER_TO_THREE[c]}>" for c in AA_ALPHABET)
_AA_COLUMN = {c: i for i, c in enumerate(AA_ALPHABET)}

# What may legally follow a `<pN>` in the sequence section: the canonical 20
# plus the `<UNK>` the generator emits for a non-standard residue.
_RESIDUE_TAIL_TOKENS = frozenset(AA_TOKENS) | {"<UNK>"}

# Floor for the renormalized conditionals, matching inference.py's. bf16
# softmax can round a hopeless amino acid to exactly zero, and these feed
# log-ratios.
_PROB_FLOOR = 1e-12


@dataclass(frozen=True)
class AAConditionals:
    """Per-residue amino-acid conditionals under ``K`` document orderings.

    Every array is indexed by *sequence* position (0-based, N- to C-terminal),
    not by the shuffled statement order — so ``logprobs[k, i]`` is residue
    ``i`` of the chain as scored under ordering ``k``.

    Attributes:
        entry_id: Seed stem the orderings were derived from.
        seq_len: Chain length ``L``.
        logprobs: ``(K, L, 20)`` log ``P(aa | context)``, renormalized over
            the 20 columns of :data:`AA_ALPHABET`.
        context_sizes: ``(K, L)`` count of *other residues* whose identity the
            model had already seen at that slot — 0 to ``L - 1``. The two
            terminus statements are not counted (they carry chain topology,
            not residue identity), though they are present in the prompt.
        target_mass: ``(K, L)`` softmax mass that fell on the 20 amino-acid
            tokens before renormalization. Well below 1 means the model put
            real weight on ``<UNK>`` or off-grammar tokens — worth checking,
            not itself an error.
    """

    entry_id: str
    seq_len: int
    logprobs: np.ndarray
    context_sizes: np.ndarray
    target_mass: np.ndarray

    @property
    def num_orderings(self) -> int:
        return int(self.logprobs.shape[0])

    def context_fractions(self) -> np.ndarray:
        """:attr:`context_sizes` as a fraction of the ``L - 1`` available."""
        denom = max(self.seq_len - 1, 1)
        return self.context_sizes.astype(np.float64) / denom

    def mean_logprobs(self, *, min_context_fraction: float = 0.0) -> np.ndarray:
        """``(L, 20)`` log-probs averaged over the qualifying orderings.

        Averages in log space — the ensemble is over conditionals, and it is
        the *log*-ratio between two amino acids at one slot that a
        variant-effect score sums. Slots below ``min_context_fraction`` are
        dropped before averaging.

        Residues with no qualifying ordering come back as ``nan``; with the
        default threshold of 0 there are none. Raising the threshold trades
        sharper conditioning for fewer samples per residue, so check the
        companion :meth:`sample_counts` when you raise it.
        """
        keep = self.context_fractions() >= min_context_fraction
        counts = keep.sum(axis=0)
        weighted = np.where(keep[:, :, None], self.logprobs, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = weighted / counts[:, None]
        return np.where(counts[:, None] > 0, out, np.nan)

    def sample_counts(self, *, min_context_fraction: float = 0.0) -> np.ndarray:
        """``(L,)`` orderings per residue that clear ``min_context_fraction``."""
        return (self.context_fractions() >= min_context_fraction).sum(axis=0)


@dataclass(frozen=True)
class _Realization:
    """One ordering of a protein's sequence section, ready to score."""

    token_strings: list[str]
    # readout_index[i] = the token slot whose next-token distribution is
    # residue i's amino acid, i.e. the slot holding residue i's <pN>.
    readout_index: np.ndarray
    context_size: np.ndarray


def _realize(
    entry_id: str, residues: Sequence[ResidueInfo], ordering: int
) -> _Realization:
    """Build one shuffled sequence section and locate its amino-acid slots.

    The document is generated with no contacts and truncated just before
    ``<begin_statements>``: the structure section is irrelevant to a causal
    readout of slots that precede it, so generating it would be wasted
    compute. The doc-type token stays ``<contacts-v1>`` — the models were
    trained on structure-bearing documents, so that is the in-distribution
    prompt, not ``<contacts-v1.sequence_only>``.
    """
    result = build_document(
        f"{entry_id}#ord{ordering}", residues, [], config=GenerationConfig()
    )
    if result is None:
        raise ValueError(
            f"contacts-v1 cannot serialize {entry_id!r} with "
            f"{len(residues)} residues: chains must have 2..{NUM_POSITION_INDICES} "
            f"residues and their sequence section must fit the context."
        )

    tokens = result.document.split()
    start = tokens.index(BEGIN_SEQUENCE_TOKEN) + 1
    stop = tokens.index(BEGIN_STRUCTURE_TOKEN)
    section = tokens[start:stop]
    if len(section) % 2 != 0:
        raise ValueError(
            f"contacts-v1 sequence section for {entry_id!r} is not a whole "
            f"number of 2-token statements ({len(section)} tokens)."
        )

    seq_len = result.seq_len
    readout_index = np.full(seq_len, -1, dtype=np.int64)
    context_size = np.zeros(seq_len, dtype=np.int32)
    seen = 0
    # Statements are uniform 2-token pairs, so a residue statement is exactly
    # one whose *head* is a position token; `<n-term> <pN>` / `<c-term> <pN>`
    # carry their position in the tail slot instead and are skipped here.
    for statement, (head, tail) in enumerate(zip(section[0::2], section[1::2])):
        if head in {N_TERM_TOKEN, C_TERM_TOKEN}:
            continue
        position = _position_index(head, entry_id)
        seq_index = (position - result.n_term_index) % NUM_POSITION_INDICES
        if not 0 <= seq_index < seq_len:
            raise ValueError(
                f"position token {head} in {entry_id!r} maps to out-of-range "
                f"sequence index {seq_index} (chain length {seq_len})."
            )
        if tail not in _RESIDUE_TAIL_TOKENS:
            raise ValueError(
                f"residue statement {head} in {entry_id!r} is followed by "
                f"{tail!r}, which is not an amino-acid token."
            )
        # `start` offsets the section back into whole-document coordinates;
        # the head's own slot is the one whose logits predict the tail.
        readout_index[seq_index] = start + 2 * statement
        context_size[seq_index] = seen
        seen += 1

    if seen != seq_len or int(readout_index.min()) < 0:
        raise ValueError(
            f"contacts-v1 sequence section for {entry_id!r} defines {seen} of "
            f"{seq_len} residues exactly once; the document is malformed."
        )
    return _Realization(
        token_strings=tokens[:stop],
        readout_index=readout_index,
        context_size=context_size,
    )


def _position_index(token: str, entry_id: str) -> int:
    """``"<p37>" -> 37``, rejecting anything that isn't a position token."""
    if not (token.startswith("<p") and token.endswith(">")):
        raise ValueError(
            f"expected a position token at the head of a residue statement in "
            f"{entry_id!r}; got {token!r}."
        )
    return int(token[2:-1])


def amino_acid_conditionals(
    backend: Backend,
    residues: Sequence[ResidueInfo],
    *,
    entry_id: str = "sequence",
    num_orderings: int = 1,
    batch_size: int | None = None,
) -> AAConditionals:
    """Score every residue's amino-acid conditional under ``K`` orderings.

    One forward pass per ordering. Orderings are drawn deterministically from
    ``entry_id`` (each is seeded ``f"{entry_id}#ord{k}"``), so a rerun with the
    same arguments reproduces the same numbers, and two proteins with the same
    ``entry_id`` are compared under the same shuffles.

    Args:
        backend: A backend implementing
            :meth:`~marinfold.inference.core.Backend.teacher_forced_target_probs`
            (transformers today).
        residues: The chain in sequence order.
        entry_id: Seed stem for the orderings.
        num_orderings: ``K``. 1 gives a single random conditioning set per
            residue; the masked-marginals use wants enough that every residue
            lands late in some ordering (see the module docstring).
        batch_size: Orderings per forward pass, a memory bound. Passed
            through to the backend.

    Returns:
        An :class:`AAConditionals` with ``(K, L, 20)`` log-probabilities.
    """
    if num_orderings < 1:
        raise ValueError(f"num_orderings must be >= 1; got {num_orderings}.")

    realizations = [
        _realize(entry_id, residues, k) for k in range(num_orderings)
    ]
    tokenizer = backend.tokenizer
    aa_ids = [_token_id(tokenizer, token) for token in AA_TOKENS]

    token_ids_batch = [
        _encode_exact(tokenizer, r.token_strings) for r in realizations
    ]
    probs = backend.teacher_forced_target_probs(
        token_ids_batch, aa_ids, batch_size=batch_size
    )

    readout = np.stack([r.readout_index for r in realizations])  # (K, L)
    gathered = np.take_along_axis(probs, readout[:, :, None], axis=1)  # (K, L, 20)
    mass = gathered.sum(axis=-1)
    renormalized = gathered / np.clip(mass, _PROB_FLOOR, None)[:, :, None]
    return AAConditionals(
        entry_id=entry_id,
        seq_len=len(residues),
        logprobs=np.log(np.clip(renormalized, _PROB_FLOOR, None)).astype(np.float32),
        context_sizes=np.stack([r.context_size for r in realizations]),
        target_mass=mass.astype(np.float32),
    )


def substitution_log_ratios(
    conditionals: AAConditionals,
    wt_sequence: str,
    *,
    min_context_fraction: float = 0.0,
) -> np.ndarray:
    """``(L, 20)`` of ``log P(aa | context) − log P(wt aa | context)``.

    The per-site term of a masked-marginals variant-effect score (ESM-1v's
    convention): a variant's score is the sum of this over its mutated
    positions, and a wild-type residue scores exactly 0 by construction.

    The ratio is taken *within* each ordering and then averaged, which is what
    makes the ensemble meaningful — the nuisance offsets an ordering
    contributes to both terms cancel before averaging rather than after.

    Sites whose wild-type letter is non-canonical (``X``, ``B``, ``U``, a gap —
    the document carries ``<UNK>`` there) have no reference column, so their
    whole row comes back ``nan`` rather than a plausible-looking ratio. Summing
    over a variant that touches such a site therefore yields ``nan`` and is
    visible, instead of quietly contributing a wrong term.

    Args:
        conditionals: Output of :func:`amino_acid_conditionals`.
        wt_sequence: The wild-type one-letter sequence the conditionals were
            computed for.
        min_context_fraction: Drop slots conditioned on less than this
            fraction of the other residues.

    Raises:
        ValueError: ``wt_sequence`` length does not match the conditionals.
    """
    if len(wt_sequence) != conditionals.seq_len:
        raise ValueError(
            f"wt_sequence has {len(wt_sequence)} residues but the conditionals "
            f"cover {conditionals.seq_len}."
        )

    canonical = np.array([c in _AA_COLUMN for c in wt_sequence])
    wt_columns = np.array([_AA_COLUMN.get(c, 0) for c in wt_sequence])
    keep = conditionals.context_fractions() >= min_context_fraction  # (K, L)
    wt_logprob = np.take_along_axis(
        conditionals.logprobs, wt_columns[None, :, None], axis=2
    )  # (K, L, 1)
    ratios = conditionals.logprobs - wt_logprob
    counts = keep.sum(axis=0)
    summed = np.where(keep[:, :, None], ratios, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = summed / counts[:, None]
    usable = (counts > 0) & canonical
    return np.where(usable[:, None], out, np.nan)


def _encode_exact(tokenizer, token_strings: list[str]) -> list[int]:
    """Encode a domain-token list, asserting the 1:1 mapping holds.

    Every contacts-v1 token is a dedicated vocab entry, so encoding the
    space-joined document must give exactly one id per token. A mismatch means
    the wrong tokenizer — which would otherwise show up as a plausible-looking
    but meaningless set of conditionals.
    """
    ids = list(tokenizer.encode(" ".join(token_strings), add_special_tokens=False))
    if len(ids) != len(token_strings):
        raise ValueError(
            f"Tokenizer produced {len(ids)} ids for {len(token_strings)} "
            f"contacts-v1 tokens. The tokenizer is not the contacts-v1 one — "
            f"make sure it is co-located with the model."
        )
    return ids


def _token_id(tokenizer, token: str) -> int:
    """Resolve one domain token to its id; fail loudly on an UNK collapse."""
    tid = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk_id is not None and tid == unk_id):
        raise ValueError(
            f"Tokenizer has no dedicated id for {token!r} (got {tid}). The "
            f"tokenizer is missing the contacts-v1 vocabulary — make sure it "
            f"is co-located with the model."
        )
    return int(tid)
