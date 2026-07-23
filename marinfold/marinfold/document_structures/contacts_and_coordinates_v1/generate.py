# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-coordinates-v1 document generation.

Library module — the CLI surface lives in ``cli.py`` next door. The format
is defined in ``SPEC.md`` in this directory. One document per input
structure, fully deterministic given the structure's ``entry_id``.

A document is contacts-v1's sequence section and ``<contact>`` statements
verbatim, followed by a coordinate section: a stream of ``<pX> <ATOM>
<xyz-...>+`` mention events that reveal atom positions coarse-to-fine, with
per-mention noise (see ``SPEC.md`` for the noise calibration). Positions
are expressed in a random rotated + translated frame inside a
``cube_size`` Å cube, so orientation and location are free data
augmentation that leaves every physical distance unchanged.

The RNG draw order is load-bearing (``SPEC.md`` → *RNG draw order*):

1. residue n-terminal start index
2. sequence-section shuffle
3. rotation quaternion
4. translation offset (x, y, z)
5. contact sample → shuffle → per-pair order flips
6. mention-event stream: per event, atom choice → depth → noise (x, y, z)

The pure builder :func:`build_document` takes already-computed residues,
contacts, and per-atom coordinates, so it (and its determinism / frame /
noise / scheduling) is unit-testable without pyconfind or gemmi.
:func:`generate_document` / :func:`generate_documents` wire the structure
analysis in front of it.
"""

import hashlib
import math
import random
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from marinfold.document_structures.contacts_v1.generate import EmittedContact

from .parse import (
    DEFAULT_CIF_COLUMN,
    DEFAULT_ID_COLUMN,
    AnalyzedCoordStructure,
    AtomCoord,
    RawContact,
    ResidueInfo,
    analyze_coordinates,
    iter_coordinate_structures,
)
from .vocab import (
    BEGIN_SEQUENCE_TOKEN,
    BEGIN_STRUCTURE_TOKEN,
    CONTACT_TOKEN,
    CONTEXT_LENGTH,
    C_TERM_TOKEN,
    DOC_TYPE_TOKEN,
    END_TOKEN,
    NUM_POSITION_INDICES,
    N_TERM_TOKEN,
    atom_token,
    position_token,
    xyz_token_for_digits,
)


# Token counts the budget arithmetic depends on.
_SEQ_TOKENS_PER_RESIDUE = 2      # <pX> <AA>
_TERMINUS_TOKENS = 4            # <n-term> <pS>  and  <c-term> <pE>  (2 x 2)
_CONTACT_TOKENS_PER_STATEMENT = 3   # <contact> <pX> <pY>
# <contacts-and-coordinates-v1> <begin_sequence> … <begin_statements> … <end>
_FRAME_TOKENS = 4
_MENTION_HEADER_TOKENS = 2      # <pX> <ATOM>, before the 1..max_depth xyz tokens

# Bin width (Å) at each revealed place, coarsest -> finest: hundreds, tens,
# ones, tenths. The format emits the first ``max_depth`` places; the default
# ``max_depth = 3`` stops at ones (1 Å resolution). Index 3 (tenths) exists
# only to support raising ``max_depth`` to 4 as a future knob — no vocab
# change (SPEC → Coordinates / Suggested default parameters).
_BIN_WIDTHS = (100.0, 10.0, 1.0, 0.1)
_MAX_SUPPORTED_DEPTH = len(_BIN_WIDTHS)


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for contacts-and-coordinates-v1 generation.

    The pyconfind geometry knobs (``native_only`` … ``assembly``) and the
    contact-definition knobs (``min_seq_separation`` / ``min_contact_degree``)
    are inherited from contacts-v1 unchanged. The rest govern this format's
    coordinate section and are the SPEC's *Suggested default parameters*:

    - ``n_contacts_zero_prob`` / ``n_contacts_max``: contacts are a uniform
      random sample here (a conditioning hint), not strongest-first —
      ``P(N = 0) = n_contacts_zero_prob``, else ``N ~ Uniform{1 ..
      min(n_contacts_max, eligible)}``.
    - ``cube_size`` / ``cube_margin``: the ``<xyz-DDD>`` range per axis and
      the placement margin; a structure spanning more than
      ``cube_size - 2*cube_margin`` Å on any axis (after rotation) is skipped.
    - ``max_depth``: the finest place emitted — 3 (hundreds/tens/ones →
      1 Å resolution) by default. Raising it to 4 reintroduces a tenths digit
      with no vocab change (SPEC → Coordinates).
    - ``noise_divisor``: the per-depth Gaussian noise is
      ``sigma_d = bin_width_d / noise_divisor`` — a uniform ~95.45%
      bin-center reliability at every depth.
    - ``depth_kernel_epsilon``: floor weight in the depth-scheduling kernel,
      so no depth is ever impossible.
    - ``force_full_precision_first_event``: the document's very first
      coordinate statement always gets full precision, i.e. depth ``max_depth``
      (SPEC → Mention scheduling).
    """

    native_only: bool = True
    contact_distance: float = 3.0
    dcut: float = 25.0
    clash_distance: float = 2.0
    min_seq_separation: int = 6
    min_contact_degree: float = 0.001
    num_position_indices: int = NUM_POSITION_INDICES
    assembly: int | str | None = None
    n_contacts_zero_prob: float = 0.3
    n_contacts_max: int = 50
    cube_size: float = 1000.0
    cube_margin: float = 10.0
    max_depth: int = 3
    noise_divisor: float = 4.0
    depth_kernel_epsilon: float = 0.05
    force_full_precision_first_event: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= _MAX_SUPPORTED_DEPTH:
            raise ValueError(
                f"max_depth must be in [1, {_MAX_SUPPORTED_DEPTH}], "
                f"got {self.max_depth}"
            )

    def depth_sigmas(self) -> tuple[float, ...]:
        """Per-depth Gaussian noise stdev (Å): ``bin_width_d / noise_divisor``.

        One entry per revealed place, coarsest first. At the default
        ``max_depth = 3`` this is ``(25.0, 2.5, 0.25)``.
        """
        return tuple(
            _BIN_WIDTHS[d] / self.noise_divisor for d in range(self.max_depth)
        )


@dataclass(frozen=True)
class GenerationResult:
    """One generated document plus the metadata worth saving alongside it.

    The sequence / contact fields mirror contacts-v1; the rest are the
    coordinate-specific stats the SPEC's *Metadata schema* note calls for
    (frame parameters, mention-event counts, depth histogram, truncation).
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
    num_contacts_eligible: int
    contacts_emitted: int
    num_eligible_atoms: int
    num_events: int
    num_distinct_atoms_mentioned: int
    max_depth: int
    # One count per depth 1..max_depth (finest place emitted).
    depth_histogram: tuple[int, ...]
    rotation_quaternion: tuple[float, float, float, float]  # (w, x, y, z)
    translation: tuple[float, float, float]
    truncated: bool
    num_tokens: int
    contacts: tuple[EmittedContact, ...] = field(default_factory=tuple)

    @property
    def sha1(self) -> str:
        """SHA1 of the document string."""
        return hashlib.sha1(self.document.encode()).hexdigest()

    def metadata_row(self) -> dict[str, Any]:
        """Flat row (document + scalar metadata) for the docs parquet/jsonl.

        Emits one ``depth{i}_count`` column per depth 1..``max_depth`` (three
        at the default 1 Å resolution); the schema is stable across a run
        because every document in a run shares one ``max_depth``.
        """
        qw, qx, qy, qz = self.rotation_quaternion
        tx, ty, tz = self.translation
        row: dict[str, Any] = {
            "document": self.document,
            "entry_id": self.entry_id,
            "seq_len": self.seq_len,
            "global_plddt": self.global_plddt,
            "start_index": self.start_index,
            "n_term_index": self.n_term_index,
            "c_term_index": self.c_term_index,
            "min_seq_separation": self.min_seq_separation,
            "num_contacts_eligible": self.num_contacts_eligible,
            "contacts_emitted": self.contacts_emitted,
            "num_eligible_atoms": self.num_eligible_atoms,
            "num_events": self.num_events,
            "num_distinct_atoms_mentioned": self.num_distinct_atoms_mentioned,
            "max_depth": self.max_depth,
        }
        for i, count in enumerate(self.depth_histogram, start=1):
            row[f"depth{i}_count"] = count
        row.update({
            "quat_w": qw,
            "quat_x": qx,
            "quat_y": qy,
            "quat_z": qz,
            "trans_x": tx,
            "trans_y": ty,
            "trans_z": tz,
            "truncated": self.truncated,
            "num_tokens": self.num_tokens,
            "sha1": self.sha1,
        })
        return row

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


def _fixed_token_cost(num_residues: int) -> int:
    """Token cost of the framing + full sequence section (no contacts/coords)."""
    return _FRAME_TOKENS + _SEQ_TOKENS_PER_RESIDUE * num_residues + _TERMINUS_TOKENS


def _coord_scale(max_depth: int) -> int:
    """Integer scale making the finest revealed place the ones digit of ``round(v*scale)``.

    ``max_depth = 3`` (finest place = ones) → 1; ``max_depth = 4`` (tenths) → 10.
    """
    return 10 ** (max_depth - 3)


def _max_coord(max_depth: int) -> float:
    """Clamp bound: largest coordinate whose digits fit ``max_depth`` places.

    999 at 1 Å resolution (``max_depth = 3``), 999.9 with a tenths digit.
    Clamping a (noisy) coordinate here keeps a boundary atom from producing an
    out-of-range digit (SPEC → Coordinate frame).
    """
    return (10 ** max_depth - 1) / _coord_scale(max_depth)


def _coordinate_digits(value: float, max_depth: int) -> tuple[int, ...]:
    """The ``max_depth`` decimal digits of one clamped coordinate, coarsest first.

    At the default 1 Å resolution (SPEC → Digit extraction) this is entirely
    integer: round ``v`` to the nearest Å and read the hundreds / tens / ones
    digits with ``//`` / ``%``. A future ``max_depth = 4`` variant quantizes
    once as ``round(v * 10)`` (never divide by a float ``0.1`` — ``180.2 / 0.1``
    is ``1801.9999999999998`` in IEEE-754 and would corrupt the digit).
    """
    scale = _coord_scale(max_depth)
    clamped = min(_max_coord(max_depth), max(0.0, value))
    n = round(clamped * scale)
    # Read the ``max_depth`` decimal digits of ``n`` (a max_depth-digit int in
    # [0, 10**max_depth - 1]), most-significant first. Use an integer power-of-10
    # divisor directly: ``100 * scale`` is a float for max_depth < 3 (scale is a
    # negative power of ten there), which would make every extracted digit a
    # float and crash token formatting for the depths the config validator
    # advertises. This form is byte-identical for max_depth 3 and 4.
    return tuple((n // 10 ** (max_depth - 1 - i)) % 10 for i in range(max_depth))


def _xyz_tokens(x: float, y: float, z: float, depth: int, max_depth: int) -> list[str]:
    """The ``depth`` coordinate tokens for one mention, coarsest place first."""
    xd = _coordinate_digits(x, max_depth)
    yd = _coordinate_digits(y, max_depth)
    zd = _coordinate_digits(z, max_depth)
    return [xyz_token_for_digits(xd[p], yd[p], zd[p]) for p in range(depth)]


def _random_rotation_matrix(
    rng: random.Random,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """A uniform-SO(3) rotation from a normalized random quaternion.

    Four i.i.d. standard Gaussians, normalized to a unit quaternion, give a
    rotation drawn uniformly from SO(3) (no scipy dependency; stays inside
    the one ``rng`` stream the whole document is seeded from). Returns the
    3x3 matrix and the ``(w, x, y, z)`` quaternion for metadata.
    """
    q = [rng.gauss(0.0, 1.0) for _ in range(4)]
    norm = math.sqrt(sum(c * c for c in q))
    if norm == 0.0:  # astronomically unlikely; fall back to identity
        return np.eye(3), (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = (c / norm for c in q)
    matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return matrix, (w, x, y, z)


def _apply_frame(
    rng: random.Random,
    coords: np.ndarray,
    config: GenerationConfig,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float]] | None:
    """Rotate about the centroid, then randomly place inside the cube.

    Returns the transformed ``(M, 3)`` coordinates, the rotation quaternion,
    and the translation vector — or ``None`` if the rotated structure spans
    more than ``cube_size - 2*margin`` Å on any axis (too large to fit; the
    caller skips the protein). Draws the rotation (4 Gaussians) then the
    translation (3 uniforms, x/y/z) in that order.
    """
    centroid = coords.mean(axis=0)
    matrix, quaternion = _random_rotation_matrix(rng)
    # A rigid transform of finite coordinates is finite; some BLAS builds
    # (macOS Accelerate under numpy 2.x) raise spurious divide/overflow/
    # invalid FP-status warnings on this matmul regardless. Mask the FP
    # flags for this one call — it hides no real error (inputs are finite).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rotated = (coords - centroid) @ matrix.T

    lo = rotated.min(axis=0)
    hi = rotated.max(axis=0)
    span = hi - lo
    limit = config.cube_size - 2.0 * config.cube_margin
    if float(span.max()) > limit:
        return None

    translation = np.empty(3)
    for axis in range(3):
        slack = limit - float(span[axis])  # >= 0 given the check above
        target_lo = config.cube_margin + rng.uniform(0.0, slack)
        translation[axis] = target_lo - float(lo[axis])
    transformed = rotated + translation
    return transformed, quaternion, (
        float(translation[0]), float(translation[1]), float(translation[2]),
    )


def _sample_depth(rng: random.Random, t: float, config: GenerationConfig) -> int:
    """Sample a mention depth (1..max_depth) from the coarse-to-fine kernel.

    ``t`` in [0, 1] is the fraction of the coordinate budget already emitted.
    The ``max_depth`` depths have centers evenly spaced over [0, 1]
    (``[0, 1/2, 1]`` at the default max_depth 3) and raw weight
    ``max(0, 1 - 3*|t - c_d|) + epsilon``; weights are normalized and sampled.
    At ``t = 0`` depth 1 carries ~91% of the mass (max_depth 3); by ``t = 1``
    the finest depth does — so the document trends hundreds → ones overall
    while still allowing the occasional early-deep / late-shallow mention.
    """
    max_depth = config.max_depth
    if max_depth == 1:
        return 1
    centers = [i / (max_depth - 1) for i in range(max_depth)]
    weights = [max(0.0, 1.0 - 3.0 * abs(t - c)) + config.depth_kernel_epsilon
               for c in centers]
    total = math.fsum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for depth_minus_1, weight in enumerate(weights):
        cumulative += weight
        if r < cumulative:
            return depth_minus_1 + 1
    return max_depth  # float-rounding fallback


def _select_contacts(
    rng: random.Random,
    contacts: Sequence[RawContact],
    residues: Sequence[ResidueInfo],
    pos_of_seq: Sequence[int],
    config: GenerationConfig,
) -> tuple[list[EmittedContact], int]:
    """Uniformly sample the conditioning contacts (SPEC → Contacts).

    Not strongest-first: with probability ``n_contacts_zero_prob`` emit none,
    else ``N ~ Uniform{1 .. min(n_contacts_max, eligible)}`` sampled uniformly
    from the above-threshold pool, listed in random order, each pair's order
    coin-flipped. Returns the emitted contacts and the eligible-pool size.
    """
    pool = [
        c for c in contacts
        if (c.seq_j - c.seq_i) >= config.min_seq_separation
        and c.degree >= config.min_contact_degree
    ]
    num_eligible = len(pool)

    # Always draw the zero-vs-nonzero coin first, so the RNG stream doesn't
    # depend on whether the pool happened to be empty.
    draw = rng.random()
    if draw < config.n_contacts_zero_prob or num_eligible == 0:
        return [], num_eligible

    n = rng.randint(1, min(config.n_contacts_max, num_eligible))
    selected = rng.sample(pool, n)
    rng.shuffle(selected)
    emitted = [
        EmittedContact(
            seq_i=c.seq_i,
            seq_j=c.seq_j,
            pos_i=pos_of_seq[c.seq_i],
            pos_j=pos_of_seq[c.seq_j],
            resnum_i=residues[c.seq_i].resnum,
            resnum_j=residues[c.seq_j].resnum,
            resname_i=residues[c.seq_i].resname,
            resname_j=residues[c.seq_j].resname,
            degree=c.degree,
            flipped=rng.random() < 0.5,
        )
        for c in selected
    ]
    return emitted, num_eligible


def _eligible_atoms(
    residues: Sequence[ResidueInfo],
    atoms_by_seq_index: Mapping[int, Sequence[AtomCoord]],
) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Flatten per-residue atoms to a parallel (identity list, coord array).

    Identity is ``(seq_index, atom_name)``; coordinates are the raw
    structure positions, ``(M, 3)``. Order is residue sequence order, then
    the residue's atom order — a fixed, deterministic enumeration.
    """
    identities: list[tuple[int, str]] = []
    coords: list[tuple[float, float, float]] = []
    for r in residues:
        for name, x, y, z in atoms_by_seq_index.get(r.seq_index, ()):
            identities.append((r.seq_index, name))
            coords.append((x, y, z))
    array = np.array(coords, dtype=float) if coords else np.empty((0, 3))
    return identities, array


def build_document(
    entry_id: str,
    residues: Sequence[ResidueInfo],
    contacts: Sequence[RawContact],
    atoms_by_seq_index: Mapping[int, Sequence[AtomCoord]],
    *,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    global_plddt: float = math.nan,
) -> GenerationResult | None:
    """Build one contacts-and-coordinates-v1 document.

    Pure and deterministic given ``entry_id`` (the RNG seed). Returns
    ``None`` if the chain can't be serialized: fewer than 2 residues, more
    residues than there are position indices, the framing + sequence section
    already exceeds ``context_length``, or the rotated structure is too large
    to fit the cube (SPEC → Coordinate frame).

    ``residues`` must be in sequence order; ``contacts`` reference them by
    0-based ``seq_i < seq_j``; ``atoms_by_seq_index`` maps a residue's
    ``seq_index`` to its eligible heavy atoms.
    """
    residues = list(residues)
    num_residues = len(residues)
    num_indices = config.num_position_indices
    if num_residues < 2 or num_residues > num_indices:
        return None
    fixed = _fixed_token_cost(num_residues)
    if fixed > context_length:
        return None

    rng = random.Random(_generation_seed(entry_id))

    # (1) Residue numbering: random n-terminal index, then wrap around.
    start = rng.randrange(num_indices)
    pos_of_seq = [(start + k) % num_indices for k in range(num_residues)]
    n_term_index = pos_of_seq[0]
    c_term_index = pos_of_seq[-1]

    # (2) Sequence section: per-residue assignments + the two termini, shuffled.
    seq_statements: list[tuple[str, ...]] = [
        (position_token(pos_of_seq[k]), f"<{r.resname}>")
        for k, r in enumerate(residues)
    ]
    seq_statements.append((N_TERM_TOKEN, position_token(n_term_index)))
    seq_statements.append((C_TERM_TOKEN, position_token(c_term_index)))
    rng.shuffle(seq_statements)

    # (3) + (4) Frame transform: rotate about centroid, place inside the cube.
    identities, coords = _eligible_atoms(residues, atoms_by_seq_index)
    num_eligible_atoms = len(identities)
    if num_eligible_atoms > 0:
        framed = _apply_frame(rng, coords, config)
        if framed is None:
            return None  # too large to fit the cube; skip the protein
        transformed, quaternion, translation = framed
    else:
        quaternion, translation = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        transformed = coords

    # (5) Contacts: uniform sample, random order, per-pair coin flips.
    emitted_contacts, num_contacts_eligible = _select_contacts(
        rng, contacts, residues, pos_of_seq, config
    )

    # (6) Coordinate section: mention events until the budget runs out.
    coord_budget = (
        context_length - fixed
        - _CONTACT_TOKENS_PER_STATEMENT * len(emitted_contacts)
    )
    sigmas = config.depth_sigmas()
    max_depth = config.max_depth
    coord_tokens: list[str] = []
    coord_token_count = 0
    depth_histogram = [0] * max_depth
    mentioned: set[tuple[int, str]] = set()
    num_events = 0
    truncated = False

    while num_eligible_atoms > 0 and coord_budget > 0:
        # (6a) pick an atom uniformly, with replacement.
        atom_idx = rng.randrange(num_eligible_atoms)
        # (6b) pick a depth: the first event is forced to full precision.
        if num_events == 0 and config.force_full_precision_first_event:
            depth = max_depth
        else:
            depth = _sample_depth(rng, coord_token_count / coord_budget, config)
        cost = _MENTION_HEADER_TOKENS + depth
        if coord_token_count + cost > coord_budget:
            # The sampled event doesn't fit; drop it and stop (truncation).
            truncated = True
            break
        # (6c) draw fresh noise for this mention and read off the digits.
        tx, ty, tz = transformed[atom_idx]
        sigma = sigmas[depth - 1]
        nx = tx + rng.gauss(0.0, sigma)
        ny = ty + rng.gauss(0.0, sigma)
        nz = tz + rng.gauss(0.0, sigma)
        seq_index, atom_name = identities[atom_idx]
        coord_tokens.append(position_token(pos_of_seq[seq_index]))
        coord_tokens.append(atom_token(atom_name))
        coord_tokens.extend(_xyz_tokens(nx, ny, nz, depth, max_depth))
        coord_token_count += cost
        depth_histogram[depth - 1] += 1
        mentioned.add((seq_index, atom_name))
        num_events += 1

    # Assemble the token stream.
    tokens: list[str] = [DOC_TYPE_TOKEN, BEGIN_SEQUENCE_TOKEN]
    for statement in seq_statements:
        tokens.extend(statement)
    tokens.append(BEGIN_STRUCTURE_TOKEN)
    for c in emitted_contacts:
        first, second = (c.pos_j, c.pos_i) if c.flipped else (c.pos_i, c.pos_j)
        tokens += [CONTACT_TOKEN, position_token(first), position_token(second)]
    tokens.extend(coord_tokens)
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
        num_contacts_eligible=num_contacts_eligible,
        contacts_emitted=len(emitted_contacts),
        num_eligible_atoms=num_eligible_atoms,
        num_events=num_events,
        num_distinct_atoms_mentioned=len(mentioned),
        max_depth=max_depth,
        depth_histogram=tuple(depth_histogram),
        rotation_quaternion=quaternion,
        translation=translation,
        truncated=truncated,
        num_tokens=len(tokens),
        contacts=tuple(emitted_contacts),
    )


def _result_from_analyzed(
    analyzed: AnalyzedCoordStructure,
    *,
    context_length: int,
    config: GenerationConfig,
) -> GenerationResult | None:
    """Apply :func:`build_document` to an :class:`AnalyzedCoordStructure`."""
    num_residues = len(analyzed.residues)
    if not (2 <= num_residues <= config.num_position_indices):
        warnings.warn(
            f"skipping {analyzed.entry_id}: {num_residues} residues outside "
            f"[2, {config.num_position_indices}]",
            stacklevel=2,
        )
        return None
    fixed = _fixed_token_cost(num_residues)
    if fixed > context_length:
        warnings.warn(
            f"skipping {analyzed.entry_id}: fixed sequence section needs "
            f"{fixed} tokens > context_length {context_length}",
            stacklevel=2,
        )
        return None
    result = build_document(
        analyzed.entry_id,
        analyzed.residues,
        analyzed.contacts,
        analyzed.atoms_by_seq_index,
        context_length=context_length,
        config=config,
        global_plddt=analyzed.global_plddt,
    )
    if result is None:
        warnings.warn(
            f"skipping {analyzed.entry_id}: could not be serialized "
            f"(too few residues or too large for the coordinate cube)",
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

    The single-structure entry point a zephyr data job calls per input.
    Returns ``None`` (with a warning) for chains that can't be serialized;
    raises ``ValueError`` for unparseable / multi-chain inputs.
    """
    analyzed = analyze_coordinates(
        structure,
        entry_id=entry_id,
        native_only=config.native_only,
        contact_distance=config.contact_distance,
        dcut=config.dcut,
        clash_distance=config.clash_distance,
        assembly=config.assembly,
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
    cif_column: str = DEFAULT_CIF_COLUMN,
    id_column: str | None = DEFAULT_ID_COLUMN,
) -> Iterator[GenerationResult]:
    """Yield one :class:`GenerationResult` per input structure (up to ``num_docs``).

    The driving entry point — ``cli.py`` parses args and calls this. See
    :func:`~marinfold.document_structures.contacts_and_coordinates_v1.parse.iter_coordinate_structures`
    for accepted ``input_path`` shapes (structure files or afdb-24M parquet).
    """
    produced = 0
    for analyzed in iter_coordinate_structures(
        Path(input_path),
        cif_column=cif_column,
        id_column=id_column,
        native_only=config.native_only,
        contact_distance=config.contact_distance,
        dcut=config.dcut,
        clash_distance=config.clash_distance,
        assembly=config.assembly,
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
