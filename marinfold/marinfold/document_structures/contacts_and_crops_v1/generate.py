# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-crops-v1 document generation.

Library module — the CLI surface lives in ``cli.py`` next door. The format
is defined in ``SPEC.md`` in this directory. One document per input
structure, fully deterministic given the structure's ``entry_id``.

A document is contacts-v1's sequence section and ``<contact>`` statements
verbatim, followed by a **two-pass coordinate section** designed to fit
8192 tokens (vs ccoord's 32768):

- **Pass 1 (coarse boxes):** every atom, sampled with replacement, gets a
  cheap 4-token mention ``<pX> <ATOM> <xyz-HHH> <xyz-TTT>`` placing it in its
  10 Å box, budget-truncated.
- **Pass 2 (crops):** a small reserved budget reveals full 0.1 Å detail
  inside a handful of selected spatial boxes, each opened by a
  ``<crop> <xyz-HHH> <xyz-TTT>`` header naming the box exactly, then its
  member atoms as ones+tenths.

Positions are expressed in a random rotated + translated frame inside a
``cube_size`` Å cube (free data augmentation; leaves every physical distance
unchanged). The frame transform, contact selection, and the ``<xyz-DDD>``
digit encoding are shared with ccoord.

The RNG draw order is load-bearing (``SPEC.md`` → *RNG draw order*):

1. residue n-terminal start index
2. sequence-section shuffle
3. rotation quaternion
4. translation offset (x, y, z)
5. contact sample → shuffle → per-pair order flips
6. Pass-1 atom-draw sequence: per draw, weighted atom choice, then box-noise
   (x, y, z)
7. Pass-2 crop sequence: per step, select-coin, box choice, then per-candidate
   membership-noise (x, y, z) + keep draw

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
from collections import defaultdict
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
    CROP_TOKEN,
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
_SEQ_TOKENS_PER_RESIDUE = 2         # <pX> <AA>
_TERMINUS_TOKENS = 4                # <n-term> <pS>  and  <c-term> <pE>  (2 x 2)
_CONTACT_TOKENS_PER_STATEMENT = 3   # <contact> <pX> <pY>
# <contacts-and-crops-v1> <begin_sequence> … <begin_statements> … <end>
_FRAME_TOKENS = 4
# Pass 1: <pX> <ATOM> <xyz-HHH> <xyz-TTT> — header + hundreds + tens.
_PASS1_MENTION_TOKENS = 4
# Pass 2: <crop> <xyz-HHH> <xyz-TTT> header, then per atom <pX> <ATOM>
# <xyz-OOO> <xyz-TTT> (ones + tenths, box reused from the header).
_CROP_HEADER_TOKENS = 3
_CROP_ATOM_TOKENS = 4

# 10 Å boxes tile the coordinate cube. The <xyz-DDD> vocab covers [0, 999.9]
# Å per axis, so a box index (round(v*10) // 100) is always in [0, 99] —
# a fixed 100^3 grid, independent of cube_size.
_GRID = 100


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for contacts-and-crops-v1 generation.

    The pyconfind geometry knobs (``native_only`` … ``assembly``) and the
    contact-definition knobs (``min_seq_separation`` / ``min_contact_degree``)
    are inherited from contacts-v1 unchanged; ``n_contacts_*`` / ``cube_*``
    are shared with ccoord. The rest govern this format's two-pass coordinate
    section and are the SPEC's *Suggested default parameters*:

    - ``fine_reserve``: tokens held back from Pass 1 so Pass 2 always has room
      for a handful of fine crops regardless of chain length.
    - ``pass1_box_noise_sigma``: isotropic Gaussian σ (Å) added once per
      Pass-1 mention before boxing — at σ=2 the tens box is correct ~98.8% of
      the time at a cell's center, so a small fraction land in the wrong box
      on purpose (Pass 2 later shows them correctly).
    - ``pass2_select_random`` / ``pass2_select_frontier``: the first two of
      the 3-way Pass-2 box-selection probabilities (random atom's box /
      frontier neighbor); the remainder (``1 - the two``) is the re-show
      probability (∝ prior show count).
    - ``pass2_refine_noise_base``: a box's ``i``-th appearance draws atom
      noise σ = ``base / (i+1)**2`` Å → 1.0, 0.25, 0.111, … so repeated reads
      sharpen a box toward a crisp tenths.
    - ``pass2_keep_prob``: independent per-candidate keep probability applied
      on top of the membership test.
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
    fine_reserve: int = 2000
    pass1_box_noise_sigma: float = 2.0
    pass2_select_random: float = 0.45
    pass2_select_frontier: float = 0.45
    pass2_refine_noise_base: float = 1.0
    pass2_keep_prob: float = 0.99

    def __post_init__(self) -> None:
        if self.fine_reserve < 0:
            raise ValueError(f"fine_reserve must be >= 0, got {self.fine_reserve}")
        reshow = 1.0 - self.pass2_select_random - self.pass2_select_frontier
        if (
            self.pass2_select_random < 0.0
            or self.pass2_select_frontier < 0.0
            or reshow < -1e-9
        ):
            raise ValueError(
                "pass2_select_random + pass2_select_frontier must be in [0, 1] "
                f"(got {self.pass2_select_random} + {self.pass2_select_frontier})"
            )

    @property
    def pass2_select_reshow(self) -> float:
        """Re-show probability — the remainder of the 3-way selection split."""
        return 1.0 - self.pass2_select_random - self.pass2_select_frontier

    def refine_sigma(self, visit_index: int) -> float:
        """Per-box refinement noise σ (Å) on a box's ``visit_index``-th read."""
        return self.pass2_refine_noise_base / (visit_index + 1) ** 2


@dataclass(frozen=True)
class GenerationResult:
    """One generated document plus the metadata worth saving alongside it.

    The sequence / contact fields mirror contacts-v1; the rest are the
    crop-specific stats the SPEC's *Metadata* note calls for (frame
    parameters, Pass-1 mention count, crop counts, per-box visit histogram,
    truncation).
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
    num_pass1_mentions: int
    num_crops: int
    num_empty_crops: int
    num_distinct_crop_boxes: int
    crop_atoms_emitted: int
    max_box_visits: int
    # Per-distinct-box show counts, descending — the visit histogram.
    box_visit_counts: tuple[int, ...]
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
        """Flat row (document + scalar metadata) for the docs parquet/jsonl."""
        qw, qx, qy, qz = self.rotation_quaternion
        tx, ty, tz = self.translation
        return {
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
            "num_pass1_mentions": self.num_pass1_mentions,
            "num_crops": self.num_crops,
            "num_empty_crops": self.num_empty_crops,
            "num_distinct_crop_boxes": self.num_distinct_crop_boxes,
            "crop_atoms_emitted": self.crop_atoms_emitted,
            "max_box_visits": self.max_box_visits,
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
        }

    def summary_dict(self) -> dict[str, Any]:
        """Rich per-protein view for the local summary JSON."""
        row = self.metadata_row()
        row.pop("document")
        row["sequence"] = [r.resname for r in self.residues]
        row["contacts"] = [c.as_dict() for c in self.contacts]
        row["box_visit_counts"] = list(self.box_visit_counts)
        return row


def _generation_seed(entry_id: str) -> int:
    """Deterministic per-entry seed (first 8 sha1 hex digits)."""
    return int(hashlib.sha1(entry_id.encode()).hexdigest()[:8], 16)


def _fixed_token_cost(num_residues: int) -> int:
    """Token cost of the framing + full sequence section (no contacts/coords)."""
    return _FRAME_TOKENS + _SEQ_TOKENS_PER_RESIDUE * num_residues + _TERMINUS_TOKENS


def _digits(value: float) -> tuple[int, int, int, int]:
    """The (hundreds, tens, ones, tenths) digits of one coordinate.

    Quantize once as ``n = round(clamp(value, 0, 999.9) * 10)`` (a
    tenths-resolution integer) and read the four places with integer
    arithmetic (SPEC → *The ``<xyz-DDD>`` vocabulary*). Never divide by a
    float ``0.1`` — ``180.2 / 0.1`` is ``1801.9999999999998`` in IEEE-754 and
    would corrupt the tenths digit. Clamping keeps a boundary atom from
    producing an out-of-range digit.
    """
    clamped = min(999.9, max(0.0, value))
    n = round(clamped * 10)
    return (n // 1000) % 10, (n // 100) % 10, (n // 10) % 10, n % 10


def _cell(value: float) -> int:
    """The 10 Å box index (0..99) of one coordinate along an axis.

    Consistent with :func:`_digits`: the box index is ``round(clamp*10) //
    100`` = ``hundreds_digit * 10 + tens_digit``, so the header's hundreds and
    tens tokens name exactly this cell.
    """
    clamped = min(999.9, max(0.0, value))
    return round(clamped * 10) // 100


def _box_header_tokens(cell: tuple[int, int, int]) -> tuple[str, str]:
    """The two tokens naming a box exactly: (hundreds token, tens token)."""
    cx, cy, cz = cell
    hundreds = xyz_token_for_digits(cx // 10, cy // 10, cz // 10)
    tens = xyz_token_for_digits(cx % 10, cy % 10, cz % 10)
    return hundreds, tens


def _ones_tenths_tokens(x: float, y: float, z: float) -> tuple[str, str]:
    """The ones and tenths tokens of a (noised) position — Pass-2 reuse-box."""
    _, _, ox, px = _digits(x)
    _, _, oy, py = _digits(y)
    _, _, oz, pz = _digits(z)
    return xyz_token_for_digits(ox, oy, oz), xyz_token_for_digits(px, py, pz)


def _neighbors(cell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """The in-bounds 26-neighborhood (face+edge+corner) of a box cell."""
    cx, cy, cz = cell
    out: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if 0 <= nx < _GRID and 0 <= ny < _GRID and 0 <= nz < _GRID:
                    out.append((nx, ny, nz))
    return out


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


def _weighted_index(rng: random.Random, weights: np.ndarray) -> int:
    """Draw one index ∝ ``weights`` with a single ``rng.random()`` draw."""
    cdf = np.cumsum(weights)
    total = float(cdf[-1])
    r = rng.random() * total
    idx = int(np.searchsorted(cdf, r, side="right"))
    return min(idx, len(weights) - 1)


def _weighted_key(rng: random.Random, keyed_weights: list[tuple[Any, float]]) -> Any:
    """Pick one key from ``(key, weight)`` pairs ∝ weight (one draw)."""
    total = math.fsum(w for _, w in keyed_weights)
    r = rng.random() * total
    cumulative = 0.0
    for key, weight in keyed_weights:
        cumulative += weight
        if r < cumulative:
            return key
    return keyed_weights[-1][0]


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
    """Build one contacts-and-crops-v1 document.

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

    # Budget for the whole structure (Pass 1 + Pass 2) coordinate content.
    structure_budget = (
        context_length - fixed
        - _CONTACT_TOKENS_PER_STATEMENT * len(emitted_contacts)
    )
    structure_budget = max(0, structure_budget)
    pass1_cap = max(0, structure_budget - config.fine_reserve)

    struct_tokens = 0
    truncated = False

    # (6) Pass 1 — coarse boxes: sample atoms with replacement, weight
    #     1/(1 + k_r), σ=2 box noise, until the next 4-token mention overflows
    #     the Pass-1 cap.
    pass1_tokens: list[str] = []
    num_pass1_mentions = 0
    if num_eligible_atoms > 0:
        atom_seq = np.fromiter(
            (seq_index for seq_index, _ in identities),
            dtype=np.int64,
            count=num_eligible_atoms,
        )
        residue_draws = np.zeros(num_residues, dtype=np.float64)  # k_r
        sigma1 = config.pass1_box_noise_sigma
        while struct_tokens + _PASS1_MENTION_TOKENS <= pass1_cap:
            weights = 1.0 / (1.0 + residue_draws[atom_seq])
            idx = _weighted_index(rng, weights)
            tx, ty, tz = (float(v) for v in transformed[idx])
            nx = tx + rng.gauss(0.0, sigma1)
            ny = ty + rng.gauss(0.0, sigma1)
            nz = tz + rng.gauss(0.0, sigma1)
            cell = (_cell(nx), _cell(ny), _cell(nz))
            hundreds, tens = _box_header_tokens(cell)
            seq_index, atom_name = identities[idx]
            pass1_tokens += [
                position_token(pos_of_seq[seq_index]),
                atom_token(atom_name),
                hundreds,
                tens,
            ]
            struct_tokens += _PASS1_MENTION_TOKENS
            residue_draws[atom_seq[idx]] += 1.0
            num_pass1_mentions += 1

    # (7) Pass 2 — crops: reveal 0.1 Å detail inside selected boxes until the
    #     structure budget is spent.
    pass2_tokens: list[str] = []
    num_crops = 0
    num_empty_crops = 0
    crop_atoms_emitted = 0
    shown_counts: dict[tuple[int, int, int], int] = {}
    if num_eligible_atoms > 0:
        true_cells = [
            (_cell(float(transformed[i, 0])),
             _cell(float(transformed[i, 1])),
             _cell(float(transformed[i, 2])))
            for i in range(num_eligible_atoms)
        ]
        cell_atoms: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, cell in enumerate(true_cells):
            cell_atoms[cell].append(i)
        frontier: set[tuple[int, int, int]] = set()
        keep_prob = config.pass2_keep_prob
        p_random = config.pass2_select_random
        p_frontier = config.pass2_select_frontier

        def _random_atom_box() -> tuple[int, int, int]:
            return true_cells[rng.randrange(num_eligible_atoms)]

        while struct_tokens + _CROP_HEADER_TOKENS <= structure_budget:
            # (7a) select a box (with replacement, 3-way). Always draw the
            #      coin first so the stream is state-independent; fall back to
            #      a random atom's box when a branch has nothing to pick.
            coin = rng.random()
            if not shown_counts or coin < p_random:
                box = _random_atom_box()
            elif coin < p_random + p_frontier:
                if frontier:
                    box = sorted(frontier)[rng.randrange(len(frontier))]
                else:
                    box = _random_atom_box()
            else:
                box = _weighted_key(
                    rng, [(b, float(c)) for b, c in shown_counts.items()]
                )

            visit_index = shown_counts.get(box, 0)
            sigma = config.refine_sigma(visit_index)
            hundreds, tens = _box_header_tokens(box)
            pass2_tokens += [CROP_TOKEN, hundreds, tens]
            struct_tokens += _CROP_HEADER_TOKENS
            num_crops += 1

            # (7b) membership via neighbor bleed-in: consider atoms whose true
            #      box is `box` or a neighbor; include iff the σ-noised
            #      position floors back into `box`, then a 0.99 keep on top.
            candidates: list[int] = list(cell_atoms.get(box, ()))
            for nb in _neighbors(box):
                candidates.extend(cell_atoms.get(nb, ()))
            emitted_here = 0
            for ai in candidates:
                tx, ty, tz = (float(v) for v in transformed[ai])
                nx = tx + rng.gauss(0.0, sigma)
                ny = ty + rng.gauss(0.0, sigma)
                nz = tz + rng.gauss(0.0, sigma)
                if (_cell(nx), _cell(ny), _cell(nz)) != box:
                    continue
                if rng.random() >= keep_prob:
                    continue
                if struct_tokens + _CROP_ATOM_TOKENS > structure_budget:
                    # A big box overflowed the budget: emit what fit (partial
                    # last crop), never skip the whole box.
                    truncated = True
                    break
                ones, tenths = _ones_tenths_tokens(nx, ny, nz)
                seq_index, atom_name = identities[ai]
                pass2_tokens += [
                    position_token(pos_of_seq[seq_index]),
                    atom_token(atom_name),
                    ones,
                    tenths,
                ]
                struct_tokens += _CROP_ATOM_TOKENS
                emitted_here += 1

            crop_atoms_emitted += emitted_here
            if emitted_here == 0:
                num_empty_crops += 1

            # (7c) bookkeeping: record the visit, drop the box from the
            #      frontier, and (only if it holds atoms) let it extend the
            #      frontier so empties don't explore vacuum.
            shown_counts[box] = visit_index + 1
            frontier.discard(box)
            if box in cell_atoms:
                for nb in _neighbors(box):
                    if nb not in shown_counts:
                        frontier.add(nb)
            if truncated:
                break

    # Assemble the token stream.
    tokens: list[str] = [DOC_TYPE_TOKEN, BEGIN_SEQUENCE_TOKEN]
    for statement in seq_statements:
        tokens.extend(statement)
    tokens.append(BEGIN_STRUCTURE_TOKEN)
    for c in emitted_contacts:
        first, second = (c.pos_j, c.pos_i) if c.flipped else (c.pos_i, c.pos_j)
        tokens += [CONTACT_TOKEN, position_token(first), position_token(second)]
    tokens.extend(pass1_tokens)
    tokens.extend(pass2_tokens)
    tokens.append(END_TOKEN)

    box_visit_counts = tuple(sorted(shown_counts.values(), reverse=True))
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
        num_pass1_mentions=num_pass1_mentions,
        num_crops=num_crops,
        num_empty_crops=num_empty_crops,
        num_distinct_crop_boxes=len(shown_counts),
        crop_atoms_emitted=crop_atoms_emitted,
        max_box_visits=max(box_visit_counts, default=0),
        box_visit_counts=box_visit_counts,
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
    :func:`~marinfold.document_structures.contacts_and_crops_v1.parse.iter_coordinate_structures`
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
