# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v2 doc generation, vectorized.

Faithful port of the exp34 v2 algorithm — same RNG stream order, same
statement build order, same float arithmetic. Two differences from
exp34's implementation:

* Consumes the columnar :class:`parse.ParsedStructure` (numpy arrays
  for cb_or_ca_xyz / plddt_per_residue + CSR-style atom table). All
  per-atom Python loops in the hot path are gone.
* The CB-CB eligibility loop is one ``np.triu_indices`` + a vectorized
  distance computation instead of an O(N²) Python nested loop. The
  resulting ``contacts_long``/``contacts_medium``/``contacts_short``
  lists are in the same row-major order as the legacy nested loop, so
  ``rng.sample`` sees identical inputs.

Byte-identity vs exp34's reference is enforced by
``tests/test_byte_identity.py``. Any change here that perturbs the RNG
stream order — adding or removing a draw, swapping ``rng.choice`` for
``rng.randrange`` on a different-sized sequence, etc. — will break that
test and is forbidden without an explicit version bump.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import numpy as np

from parse import ParsedStructure, _ATOM_NAMES_TUPLE
from vocab import NAME, THINK_TOKEN


CONTACT_TOKENS_PER_STATEMENT = 3   # <mode> <p_i> <p_j>
DISTANCE_TOKENS_PER_STATEMENT = 6  # <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for :func:`generate_one`.

    The non-``think_*`` fields reproduce v1's defaults exactly.
    The ``think_*`` fields are the v2 think-token knobs, pinned to the
    values from issue #34 (0.75 / Geom(0.13) / Uniform(-4, 4) / Geom(0.25)).
    """

    # ---- inherited from v1 -------------------------------------------------
    contact_cutoff_angstrom: float = 8.0
    long_range_sep: int = 24
    medium_range_sep: int = 12
    short_range_sep: int = 6
    contact_f_range: tuple[float, float] = (-0.1, 0.2)
    contact_rank_mean: float = 2.0
    distance_rank_mean: float = 0.0
    rank_std: float = 1.0
    residue_plddt_min: float = 70.0
    plddt_bin_edges: tuple[float, ...] = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)
    # ---- v2 think-token knobs ----------------------------------------------
    think_initial_prob: float = 0.75
    think_initial_geom_p: float = 0.13
    think_additional_count_range: tuple[float, float] = (-4.0, 4.0)
    think_run_length_geom_p: float = 0.25


def _plddt_bin_token(global_plddt: float, bin_edges: tuple[float, ...]) -> str:
    for i, edge in enumerate(bin_edges):
        if global_plddt < edge:
            if i == 0:
                return f"plddt_lt{int(edge)}"
            return f"plddt_{int(bin_edges[i - 1])}_{int(edge)}"
    return f"plddt_{int(bin_edges[-1])}_100"


def _distance_token(dist_angstrom: float) -> str:
    bin_idx = int(math.ceil(dist_angstrom / 0.5))
    if bin_idx < 1:
        bin_idx = 1
    if bin_idx > 64:
        bin_idx = 64
    return f"d{bin_idx * 0.5:.1f}"


def _geometric(rng: random.Random, p: float) -> int:
    """Sample Geometric(p), support {1, 2, ...}, via inverse-CDF."""
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p!r}")
    if p == 1.0:
        return 1
    u = 1.0 - rng.random()
    return int(math.ceil(math.log(u) / math.log(1.0 - p)))


def _sample_think_overhead(rng: random.Random, cfg: GenerationConfig
                           ) -> tuple[int, list[int]]:
    if rng.random() < cfg.think_initial_prob:
        k1 = _geometric(rng, cfg.think_initial_geom_p)
    else:
        k1 = 0
    lo, hi = cfg.think_additional_count_range
    k2 = rng.uniform(lo, hi)
    n_additional = max(int(k2), 0)
    additional_lengths = [
        _geometric(rng, cfg.think_run_length_geom_p) for _ in range(n_additional)
    ]
    return k1, additional_lengths


@dataclass(frozen=True)
class _Statement:
    rank: float
    tokens: tuple[str, ...]


def generate_one(
    structure: ParsedStructure,
    *,
    context_length: int,
    cfg: GenerationConfig,
) -> str | None:
    """Build one v2 document string for ``structure``, or ``None`` if no fit.

    Byte-identical to exp34's ``_generate_one`` on the same ``entry_id``
    + structure (enforced by ``tests/test_byte_identity.py``).
    """
    num_residues = structure.num_residues
    if num_residues < 2:
        return None

    seed = int(hashlib.sha1(structure.entry_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Draw #1: think-token overhead, MUST come before any contact/distance
    # sampling so the RNG stream layout matches exp34 + future generators
    # that share the v2 entry_id seeding contract.
    k1, additional_run_lengths = _sample_think_overhead(rng, cfg)
    total_think_tokens = k1 + sum(additional_run_lengths)

    # ---- vectorized CB-CB contact eligibility ------------------------------
    # Legacy did a per-residue Python loop building cb_positions, then a
    # nested loop over all (i < j) computing euclidean. Here we keep the
    # exact same row-major (i, j) emission order via np.triu_indices(k=1)
    # and the same FP via np.sqrt((diffs**2).sum(axis=1)) on float64.
    cb_xyz = structure.cb_or_ca_xyz
    plddt_arr = structure.plddt_per_residue
    valid_mask = (plddt_arr >= cfg.residue_plddt_min) & np.isfinite(cb_xyz[:, 0])
    valid_indices = np.flatnonzero(valid_mask) + 1  # 1-based residue indices
    valid_xyz = cb_xyz[valid_mask]

    contacts_long: list[tuple[int, int]] = []
    contacts_medium: list[tuple[int, int]] = []
    contacts_short: list[tuple[int, int]] = []
    if valid_indices.size >= 2:
        ii_arr, jj_arr = np.triu_indices(valid_indices.size, k=1)
        diffs = valid_xyz[ii_arr] - valid_xyz[jj_arr]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        i_arr = valid_indices[ii_arr]
        j_arr = valid_indices[jj_arr]
        sep_arr = j_arr - i_arr
        elig = (
            (sep_arr > 1)
            & (sep_arr >= cfg.short_range_sep)
            & (dists <= cfg.contact_cutoff_angstrom)
        )
        long_mask = elig & (sep_arr >= cfg.long_range_sep)
        medium_mask = elig & (sep_arr >= cfg.medium_range_sep) & (sep_arr < cfg.long_range_sep)
        short_mask = elig & (sep_arr < cfg.medium_range_sep)
        contacts_long = list(zip(i_arr[long_mask].tolist(), j_arr[long_mask].tolist()))
        contacts_medium = list(zip(i_arr[medium_mask].tolist(), j_arr[medium_mask].tolist()))
        contacts_short = list(zip(i_arr[short_mask].tolist(), j_arr[short_mask].tolist()))

    # 5 overhead = <task> + <begin_sequence> + <begin_statements> + <end> + <plddt_bin>
    fixed_overhead = 5 + num_residues + total_think_tokens
    available_tokens = context_length - fixed_overhead
    if available_tokens <= 0:
        return None

    # Draws #2-4: per-mode fractions.
    f_long = max(0.0, rng.uniform(*cfg.contact_f_range))
    f_medium = max(0.0, rng.uniform(*cfg.contact_f_range))
    f_short = max(0.0, rng.uniform(*cfg.contact_f_range))

    n_long = min(int(available_tokens * f_long) // CONTACT_TOKENS_PER_STATEMENT, len(contacts_long))
    n_medium = min(int(available_tokens * f_medium) // CONTACT_TOKENS_PER_STATEMENT, len(contacts_medium))
    n_short = min(int(available_tokens * f_short) // CONTACT_TOKENS_PER_STATEMENT, len(contacts_short))

    contact_tokens_used = (n_long + n_medium + n_short) * CONTACT_TOKENS_PER_STATEMENT
    n_distance = (available_tokens - contact_tokens_used) // DISTANCE_TOKENS_PER_STATEMENT

    # Draws #5-7: contact sampling.
    selected_long = rng.sample(contacts_long, n_long) if n_long > 0 else []
    selected_medium = rng.sample(contacts_medium, n_medium) if n_medium > 0 else []
    selected_short = rng.sample(contacts_short, n_short) if n_short > 0 else []

    # ---- distance statements ----------------------------------------------
    # ``distance_indices`` = 1-based residue indices with at least one
    # in-vocab atom — preserves legacy iteration order (ascending).
    atoms_per_residue = np.diff(structure.atom_offsets)
    distance_indices = (np.flatnonzero(atoms_per_residue > 0) + 1).tolist()
    atom_offsets = structure.atom_offsets
    atom_name_id = structure.atom_name_id
    atom_xyz = structure.atom_xyz

    distance_statements: list[tuple[int, int, str, str, str]] = []
    if len(distance_indices) >= 2:
        for _ in range(n_distance):
            # Draw: residue pair (with up to 10 retries when sep <= 1, exactly
            # matching the legacy retry loop).
            a, b = rng.sample(distance_indices, 2)
            i, j = (a, b) if a < b else (b, a)
            if j - i <= 1:
                ok = False
                for _retry in range(10):
                    a, b = rng.sample(distance_indices, 2)
                    i, j = (a, b) if a < b else (b, a)
                    if j - i > 1:
                        ok = True
                        break
                if not ok:
                    continue
            # Draw: per-residue atom pick. ``rng.choice(tuple)`` and
            # ``rng.randrange(len)`` both call ``self._randbelow(len)``, so
            # the RNG state evolves identically as long as the lengths match.
            ki = int(atom_offsets[i] - atom_offsets[i - 1])  # i is 1-based
            kj = int(atom_offsets[j] - atom_offsets[j - 1])
            idx_a = rng.randrange(ki)
            idx_b = rng.randrange(kj)
            i0_start = int(atom_offsets[i - 1])
            j0_start = int(atom_offsets[j - 1])
            ai_name = _ATOM_NAMES_TUPLE[int(atom_name_id[i0_start + idx_a])]
            aj_name = _ATOM_NAMES_TUPLE[int(atom_name_id[j0_start + idx_b])]
            # Pull as Python floats (.tolist on a (3,) float64 row) so the
            # subsequent (ax - bx)**2 + … + math.sqrt sequence is identical
            # to exp34's tuple-based computation.
            ax, ay, az = atom_xyz[i0_start + idx_a].tolist()
            bx, by, bz = atom_xyz[j0_start + idx_b].tolist()
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            distance_statements.append((i, j, ai_name, aj_name, _distance_token(d)))

    # ---- assemble statements + ranks --------------------------------------
    statements: list[_Statement] = []
    for i, j in selected_long:
        statements.append(_Statement(
            rank=rng.gauss(cfg.contact_rank_mean, cfg.rank_std),
            tokens=("<long-range-contact>", f"<p{i}>", f"<p{j}>"),
        ))
    for i, j in selected_medium:
        statements.append(_Statement(
            rank=rng.gauss(cfg.contact_rank_mean, cfg.rank_std),
            tokens=("<medium-range-contact>", f"<p{i}>", f"<p{j}>"),
        ))
    for i, j in selected_short:
        statements.append(_Statement(
            rank=rng.gauss(cfg.contact_rank_mean, cfg.rank_std),
            tokens=("<short-range-contact>", f"<p{i}>", f"<p{j}>"),
        ))
    for i, j, ai, aj, dist_tok in distance_statements:
        statements.append(_Statement(
            rank=rng.gauss(cfg.distance_rank_mean, cfg.rank_std),
            tokens=("<distance>", f"<p{i}>", f"<p{j}>", f"<{ai}>", f"<{aj}>", f"<{dist_tok}>"),
        ))
    statements.sort(key=lambda s: -s.rank)

    plddt_token = _plddt_bin_token(structure.global_plddt, cfg.plddt_bin_edges)
    plddt_at_end = rng.random() < 0.5
    plddt_insert_idx = None if plddt_at_end else rng.randint(0, len(statements))

    # Additional-run slot assignment — identical to exp34. When statements
    # is empty, additional runs are dropped (no slot anchor).
    think_at_slot: dict[int, int] = {}
    if k1 > 0:
        think_at_slot[0] = k1
    if statements and additional_run_lengths:
        n_stmts = len(statements)
        for length in additional_run_lengths:
            slot = rng.randint(0, n_stmts - 1)
            think_at_slot[slot] = think_at_slot.get(slot, 0) + length

    out: list[str] = []
    out.append(f"<{NAME}>")
    out.append("<begin_sequence>")
    for name in structure.sequence:
        out.append(f"<{name}>")
    out.append("<begin_statements>")
    for idx, stmt in enumerate(statements):
        # Think tokens always come *before* any pLDDT in the same slot so
        # the "immediately after <begin_statements>" wording holds at slot 0.
        n_think = think_at_slot.get(idx, 0)
        if n_think:
            out.extend([THINK_TOKEN] * n_think)
        if plddt_insert_idx is not None and idx == plddt_insert_idx:
            out.append(f"<{plddt_token}>")
        out.extend(stmt.tokens)
    if not statements and k1 > 0:
        # No statements but the initial run still landed — emit it anyway
        # so the document captures the sampled overhead.
        out.extend([THINK_TOKEN] * k1)
    if plddt_insert_idx is not None and plddt_insert_idx >= len(statements):
        out.append(f"<{plddt_token}>")
    if plddt_at_end:
        out.append(f"<{plddt_token}>")
    out.append("<end>")
    return " ".join(out)
