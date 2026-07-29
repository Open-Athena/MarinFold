# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-crops-v1 documents ↔ 3D coordinates.

The decode half of Component 1, shared by every inference plan. It is the exact
inverse of the SPEC's emission rules, and it is deliberately the only place in
this experiment that knows how a token becomes a number.

Three jobs:

* :func:`parse_observations` — walk a token stream and yield one
  :class:`Observation` per coordinate mention: which atom, where, and with what
  uncertainty. Pass-1 mentions give a 10 Å box; Pass-2 crop statements give
  ones + tenths against their header's box, with a **visit index** counted from
  how many earlier ``<crop>`` headers named that same box (the SPEC's
  σ = 1/(i+1)² refinement schedule).
* :class:`CoordinateEstimate` — fold observations into a precision-weighted
  running mean per atom, carrying a per-atom variance that becomes the
  prediction's B-factor.
* :func:`sequence_prefix` / :func:`synthesize_pass1` — build prompts. The
  sequence section comes from the format's own deterministic builder;
  ``synthesize_pass1`` re-emits a Pass-1 section from a current estimate, which
  is how Plan F feeds the global structure back to the model.

**Digit arithmetic.** A coordinate is a tenths-resolution integer
``n = round(v * 10)``, split as ``hundreds = (n // 1000) % 10``,
``tens = (n // 100) % 10``, ``ones = (n // 10) % 10``, ``tenths = n % 10``. A
10 Å box index along one axis is ``n // 100 = hundreds * 10 + tens``. So a
Pass-2 statement inside a header naming box ``c`` decodes to
``(100 * c + 10 * ones + tenths) / 10``. The SPEC's worked example — header
``<xyz-200> <xyz-070>``, body ``<xyz-526> <xyz-631>`` → (205.6, 72.3, 6.1) — is
pinned as a test.

**Atom identity.** Documents name residues by position token, which is
``(start + k) % NUM_POSITION_INDICES`` for sequence index ``k`` and a per-document
random ``start``. We build the sequence section ourselves, so ``start`` is known
and the map inverts exactly.
"""

import math
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from marinfold.document_structures.contacts_and_crops_v1 import (
    CONTEXT_LENGTH,
    NUM_POSITION_INDICES,
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_and_crops_v1.vocab import (
    BEGIN_STRUCTURE_TOKEN,
    CROP_TOKEN,
    END_TOKEN,
    atom_token,
    position_token,
    xyz_token_for_digits,
)
from marinfold.document_structures.contacts_v1.parse import residues_from_sequence

# Token cost of each statement shape, from the SPEC's budget arithmetic.
PASS1_MENTION_TOKENS = 4
CROP_HEADER_TOKENS = 3
CROP_ATOM_TOKENS = 4
FRAME_TOKENS = 4
SEQ_TOKENS_PER_RESIDUE = 2
TERMINUS_TOKENS = 4

# Per-axis variance of a Pass-1 observation: the atom is somewhere in a 10 Å
# cell (variance 10²/12) *and* the position that was boxed carried σ=2 Å noise.
PASS1_VARIANCE = 10.0**2 / 12.0 + GenerationConfig().pass1_box_noise_sigma**2

# Per-axis variance floor of a Pass-2 observation: the tenths quantization.
TENTHS_VARIANCE = 0.1**2 / 12.0

_XYZ_RE = re.compile(r"^<xyz-(\d{3})>$")
_POSITION_RE = re.compile(r"^<p(\d+)>$")
_ATOM_RE = re.compile(r"^<([A-Z]{1,3}\d?)>$")


def xyz_digits(token: str) -> tuple[int, int, int] | None:
    """The (x, y, z) digits packed in an ``<xyz-DDD>`` token, else ``None``."""
    match = _XYZ_RE.match(token)
    if match is None:
        return None
    triple = int(match.group(1))
    return triple // 100, (triple // 10) % 10, triple % 10


def position_index(token: str) -> int | None:
    """The index in a ``<pXXX>`` position token, else ``None``."""
    match = _POSITION_RE.match(token)
    return int(match.group(1)) if match else None


def box_from_header(hundreds: str, tens: str) -> tuple[int, int, int] | None:
    """The 10 Å cell a ``<crop> <xyz-HHH> <xyz-TTT>`` header names."""
    high = xyz_digits(hundreds)
    low = xyz_digits(tens)
    if high is None or low is None:
        return None
    return tuple(h * 10 + t for h, t in zip(high, low))


def box_center(cell: Sequence[int]) -> np.ndarray:
    """Centre of a 10 Å cell, in Å.

    Cell ``c`` holds the coordinates whose tenths-integer ``round(v * 10)`` lies
    in ``[100c, 100c + 99]``, i.e. ``v`` in ``[10c - 0.05, 10c + 9.95]``. The
    midpoint is ``10c + 4.95``.
    """
    return np.array([10.0 * c + 4.95 for c in cell], dtype=np.float64)


def position_from_crop(
    cell: Sequence[int], ones: str, tenths: str
) -> np.ndarray | None:
    """Decode a Pass-2 statement's ones + tenths against its header's cell."""
    ones_digits = xyz_digits(ones)
    tenths_digits = xyz_digits(tenths)
    if ones_digits is None or tenths_digits is None:
        return None
    return np.array(
        [
            (100 * c + 10 * o + p) / 10.0
            for c, o, p in zip(cell, ones_digits, tenths_digits)
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Observation:
    """One coordinate mention decoded from a document.

    Attributes:
        seq_index: 0-based index into the input sequence.
        atom_name: heavy-atom name from the format's 37-name vocabulary.
        position: decoded (x, y, z) in the document's frame, Å.
        variance: per-axis variance of this observation, Å².
        source: ``"pass1"`` or ``"crop"``.
        visit_index: for ``"crop"``, how many earlier headers named this box
            (0 for a box's first appearance); ``-1`` for Pass-1 mentions.
    """

    seq_index: int
    atom_name: str
    position: np.ndarray
    variance: float
    source: str
    visit_index: int

    @property
    def key(self) -> tuple[int, str]:
        """Atom identity: ``(seq_index, atom_name)``."""
        return (self.seq_index, self.atom_name)


def _seq_index_of(token: str, start: int, length: int) -> int | None:
    """Invert ``pos = (start + k) % NUM_POSITION_INDICES`` for a position token."""
    pos = position_index(token)
    if pos is None:
        return None
    k = (pos - start) % NUM_POSITION_INDICES
    return k if k < length else None


def parse_observations(
    tokens: Sequence[str],
    *,
    start: int,
    length: int,
    valid_atoms: Sequence[frozenset[str]] | None = None,
    refine_noise_base: float = GenerationConfig().pass2_refine_noise_base,
) -> Iterator[Observation]:
    """Decode every coordinate mention in a token stream.

    Walks the structure section as a statement machine: ``<contact>`` consumes
    3 tokens and is ignored here, ``<crop>`` opens a box, and a ``<pXXX>``-led
    4-token statement is a Pass-1 mention before the first ``<crop>`` and a crop
    member after it. Malformed or out-of-range statements are skipped — a
    *sampled* document has no grammar guarantee, and a decoder that raised on
    the first bad statement would throw away the rest of a usable structure.

    Args:
        tokens: the document's tokens (the whole document or just the structure
            section; anything before ``<begin_statements>`` is ignored).
        start: the document's residue start index, so position tokens invert.
        length: input-sequence length, to reject out-of-range positions.
        valid_atoms: optional per-residue allowed atom names. When given, a
            statement naming an atom the residue cannot have is dropped —
            worth doing, since such a mention cannot be scored against ground
            truth anyway and would otherwise pollute the coverage count.
        refine_noise_base: the SPEC's Pass-2 σ base (1.0 Å); a box's ``i``-th
            appearance has σ = base / (i+1)².
    """
    tokens = list(tokens)
    if BEGIN_STRUCTURE_TOKEN in tokens:
        tokens = tokens[tokens.index(BEGIN_STRUCTURE_TOKEN) + 1 :]

    current_cell: tuple[int, int, int] | None = None
    visit_counts: dict[tuple[int, int, int], int] = {}
    current_visit = -1

    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == END_TOKEN:
            return
        if token == CROP_TOKEN:
            if i + CROP_HEADER_TOKENS > n:
                return
            cell = box_from_header(tokens[i + 1], tokens[i + 2])
            i += CROP_HEADER_TOKENS
            if cell is None:
                # Unusable header: stay in "no current box" until the next one,
                # so the crop's members are dropped rather than mis-placed.
                current_cell = None
                continue
            current_visit = visit_counts.get(cell, 0)
            visit_counts[cell] = current_visit + 1
            current_cell = cell
            continue
        if token.startswith("<contact>"):
            i += 3
            continue

        seq_index = _seq_index_of(token, start, length)
        if seq_index is None:
            i += 1  # not a statement we understand; resynchronize
            continue
        if i + PASS1_MENTION_TOKENS > n:
            return
        atom_match = _ATOM_RE.match(tokens[i + 1])
        first, second = tokens[i + 2], tokens[i + 3]
        i += PASS1_MENTION_TOKENS
        if atom_match is None:
            continue
        atom_name = atom_match.group(1)
        if valid_atoms is not None and atom_name not in valid_atoms[seq_index]:
            continue

        if current_cell is None:
            cell = box_from_header(first, second)
            if cell is None:
                continue
            yield Observation(
                seq_index=seq_index,
                atom_name=atom_name,
                position=box_center(cell),
                variance=PASS1_VARIANCE,
                source="pass1",
                visit_index=-1,
            )
        else:
            position = position_from_crop(current_cell, first, second)
            if position is None:
                continue
            sigma = refine_noise_base / (current_visit + 1) ** 2
            yield Observation(
                seq_index=seq_index,
                atom_name=atom_name,
                position=position,
                variance=sigma**2 + TENTHS_VARIANCE,
                source="crop",
                visit_index=current_visit,
            )


class CoordinateEstimate:
    """Precision-weighted running per-atom position estimate.

    Every observation contributes with weight ``1 / variance``, so a Pass-1 box
    (σ ≈ 3.5 Å) is swamped the moment any crop refines the atom, and a box's
    later, sharper reads dominate its earlier ones — which is exactly the
    SPEC's σ = 1/(i+1)² schedule read backwards. The accumulated weight also
    yields the posterior variance, which becomes the prediction's B-factor.
    """

    def __init__(self) -> None:
        self._weighted_sum: dict[tuple[int, str], np.ndarray] = {}
        self._weight: dict[tuple[int, str], float] = {}
        self._n_crop: dict[tuple[int, str], int] = {}

    def add(self, observation: Observation) -> None:
        """Fold one observation into the estimate."""
        key = observation.key
        weight = 1.0 / observation.variance
        if key not in self._weight:
            self._weighted_sum[key] = np.zeros(3, dtype=np.float64)
            self._weight[key] = 0.0
            self._n_crop[key] = 0
        self._weighted_sum[key] += weight * observation.position
        self._weight[key] += weight
        if observation.source == "crop":
            self._n_crop[key] += 1

    def add_all(self, observations) -> "CoordinateEstimate":
        """Fold an iterable of observations in; returns self for chaining."""
        for observation in observations:
            self.add(observation)
        return self

    def __len__(self) -> int:
        return len(self._weight)

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._weight

    def keys(self):
        """The atom keys with at least one observation."""
        return self._weight.keys()

    def position(self, key: tuple[int, str]) -> np.ndarray:
        """Posterior mean position of one atom, Å."""
        return self._weighted_sum[key] / self._weight[key]

    def sigma(self, key: tuple[int, str]) -> float:
        """Posterior per-axis standard deviation of one atom, Å."""
        return math.sqrt(1.0 / self._weight[key])

    def n_crop_observations(self, key: tuple[int, str]) -> int:
        """How many Pass-2 crop statements contributed to this atom."""
        return self._n_crop[key]

    def refined_keys(self) -> set[tuple[int, str]]:
        """Atoms with at least one crop observation (i.e. not box-only)."""
        return {key for key, count in self._n_crop.items() if count > 0}

    def occupied_cells(self) -> dict[tuple[int, int, int], list[tuple[int, str]]]:
        """Current estimate grouped into 10 Å cells — Plan C/F's box list."""
        cells: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
        for key in self._weight:
            position = self.position(key)
            cell = tuple(int(round(float(v) * 10.0)) // 100 for v in position)
            cells.setdefault(cell, []).append(key)
        return cells


def sequence_prefix(entry_id: str, sequence: str) -> list[str] | None:
    """Tokens of the sequence section, through ``<begin_statements>``.

    Produced by the format's **own** deterministic builder with no contacts and
    no atoms, so it is byte-identical to the sequence section a full document
    for this ``entry_id`` would carry: the residue start index and the statement
    shuffle are RNG draws 1 and 2, which happen before the frame draws that a
    coordinate-free call skips.

    Returns ``None`` when the chain cannot be serialized (fewer than 2 residues,
    or longer than the position vocabulary).
    """
    residues = residues_from_sequence(sequence)
    result = build_document(entry_id, residues, [], {})
    if result is None:
        return None
    tokens = result.document.split()
    return tokens[: tokens.index(BEGIN_STRUCTURE_TOKEN) + 1]


def start_index(entry_id: str, sequence: str) -> int:
    """The residue start index the sequence section for ``entry_id`` uses."""
    residues = residues_from_sequence(sequence)
    result = build_document(entry_id, residues, [], {})
    if result is None:
        raise ValueError(f"{entry_id}: sequence cannot be serialized")
    return result.start_index


def pass1_budget(sequence_length: int, *, n_contacts: int = 0,
                 context_length: int = CONTEXT_LENGTH,
                 fine_reserve: int = GenerationConfig().fine_reserve) -> tuple[int, int]:
    """``(pass1_cap, structure_budget)`` in tokens, per the SPEC's arithmetic."""
    fixed = FRAME_TOKENS + SEQ_TOKENS_PER_RESIDUE * sequence_length + TERMINUS_TOKENS
    structure_budget = max(0, context_length - fixed - 3 * n_contacts)
    return max(0, structure_budget - fine_reserve), structure_budget


def synthesize_pass1(
    estimate: CoordinateEstimate,
    *,
    start: int,
    cap_tokens: int,
    rng: random.Random,
    noise_sigma: float = GenerationConfig().pass1_box_noise_sigma,
) -> list[str]:
    """Re-emit a Pass-1 section from a current coordinate estimate.

    Plan F's global feedback channel. Draws atoms exactly the way the format
    does — with replacement, weight ``1 / (1 + k_r)`` where ``k_r`` counts how
    often residue ``r`` has been drawn — adds the training-time σ=2 Å box noise,
    and fills the cap. Two properties matter and both come from copying the
    format rather than inventing a scheme: the section is a *plausible* Pass-1
    section (the model never saw a short one — Pass 1 always fills its cap), and
    keeping the noise means the loop cannot lock onto its own output.

    Set ``noise_sigma=0`` for the clean-feedback ablation.
    """
    keys = sorted(estimate.keys())
    if not keys or cap_tokens < PASS1_MENTION_TOKENS:
        return []
    positions = np.stack([estimate.position(key) for key in keys])
    seq_indices = np.array([key[0] for key in keys], dtype=np.int64)
    draws = np.zeros(int(seq_indices.max()) + 1, dtype=np.float64)

    tokens: list[str] = []
    emitted = 0
    while (emitted + 1) * PASS1_MENTION_TOKENS <= cap_tokens:
        weights = 1.0 / (1.0 + draws[seq_indices])
        cumulative = np.cumsum(weights)
        pick = min(
            int(np.searchsorted(cumulative, rng.random() * float(cumulative[-1]), side="right")),
            len(keys) - 1,
        )
        x, y, z = positions[pick]
        if noise_sigma > 0.0:
            x += rng.gauss(0.0, noise_sigma)
            y += rng.gauss(0.0, noise_sigma)
            z += rng.gauss(0.0, noise_sigma)
        cell = [min(99, max(0, int(round(min(999.9, max(0.0, v)) * 10)) // 100))
                for v in (x, y, z)]
        seq_index, atom_name = keys[pick]
        tokens += [
            position_token((start + seq_index) % NUM_POSITION_INDICES),
            atom_token(atom_name),
            xyz_token_for_digits(*(c // 10 for c in cell)),
            xyz_token_for_digits(*(c % 10 for c in cell)),
        ]
        draws[seq_indices[pick]] += 1.0
        emitted += 1
    return tokens


def crop_header(cell: Sequence[int]) -> list[str]:
    """The three tokens that open a crop on ``cell``."""
    return [
        CROP_TOKEN,
        xyz_token_for_digits(*(c // 10 for c in cell)),
        xyz_token_for_digits(*(c % 10 for c in cell)),
    ]


def render_crop(
    cell: Sequence[int],
    keys: Sequence[tuple[int, str]],
    estimate: CoordinateEstimate,
    *,
    start: int,
) -> list[str]:
    """Re-emit an already-refined box as a crop, from the current estimate.

    Plan F's local feedback channel: this is how a box's neighbours (and its own
    earlier visits) get back into the prompt. Atom order is shuffled by the
    caller if desired; the format shuffles within a crop, so a stable order here
    would leak residue adjacency the format hides.
    """
    tokens = crop_header(cell)
    for key in keys:
        position = estimate.position(key)
        digits = [int(round(min(999.9, max(0.0, float(v))) * 10)) for v in position]
        tokens += [
            position_token((start + key[0]) % NUM_POSITION_INDICES),
            atom_token(key[1]),
            xyz_token_for_digits(*((d // 10) % 10 for d in digits)),
            xyz_token_for_digits(*(d % 10 for d in digits)),
        ]
    return tokens


def estimate_to_atom_array(
    estimate: CoordinateEstimate, sequence: str, *, unrefined_sigma_floor: float = 0.0
):
    """Render a coordinate estimate as a canonical :class:`AtomArray`.

    Residue names come from the input sequence, so the file is consistent with
    what the model was conditioned on. The B-factor column carries each atom's
    posterior standard deviation, which is what the scorer's
    ``--refined-max-sigma`` splits refined from coarse-box-only on.

    Args:
        estimate: the folded observations.
        sequence: the one-letter input sequence the document was built from.
        unrefined_sigma_floor: optional lower bound applied to atoms with no
            crop observation, so a Pass-1-only atom cannot claim fine precision
            just because many boxes agreed.
    """
    from canonical_pdb import build_atom_array

    residues = residues_from_sequence(sequence)
    atoms = []
    for key in sorted(estimate.keys()):
        seq_index, atom_name = key
        if not 0 <= seq_index < len(residues):
            continue
        x, y, z = estimate.position(key)
        sigma = estimate.sigma(key)
        if estimate.n_crop_observations(key) == 0:
            sigma = max(sigma, unrefined_sigma_floor)
        atoms.append(
            (seq_index + 1, residues[seq_index].resname, atom_name,
             float(x), float(y), float(z), float(sigma))
        )
    return build_atom_array(atoms) if atoms else None
