# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 document generation.

Library module — the CLI surface lives in ``cli.py`` next door, which
imports :func:`generate_documents` from here.

The algorithm is a faithful port of
``timodonnell/contactdoc/contactdoc/generators/contacts_and_distances_v1.py``
which produced the published
``timodonnell/protein-docs/contacts-and-distances-v1-5x`` HF subset.

Per-doc algorithm (one doc per input structure, deterministic given
the structure's entry_id):

1. Compute CB-CB contacts at pLDDT-filtered residues (CA for GLY /
   missing CB), sorted into long/medium/short by sequence sep.
2. Sample per-mode token fractions f_long, f_medium, f_short
   uniformly from the configured range (can clamp to 0).
3. Pick that many random contacts per mode; allocate the remaining
   budget to distance statements (uniformly sampled residue pairs +
   atoms).
4. Rank-order statements (contacts skew earlier).
5. Place a pLDDT bin token either mid-statements or at the end.
6. Serialize to ``<contacts-and-distances-v1> <begin_sequence>
   <AAs> <begin_statements> [statements] [<plddt_*>] <end>``.

The "-5x" HF subset name refers to "up to 5 AFDB entries per Foldseek
structural cluster" — that's a *data-pipeline* selection concern,
not augmentation. One doc per input here.
"""

import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

from parse import (
    ParsedStructure,
    cb_or_ca_position,
    euclidean,
    iter_parsed_structures,
)
from vocab import NAME


CONTACT_TOKENS_PER_STATEMENT = 3      # <mode> <p_i> <p_j>
DISTANCE_TOKENS_PER_STATEMENT = 6     # <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for :func:`generate_documents`.

    Defaults reproduce contactdoc's ``ContactsAndDistancesV1Config``
    that produced the published HF dataset.
    """

    contact_cutoff_angstrom: float = 8.0
    long_range_sep: int = 24
    medium_range_sep: int = 12
    short_range_sep: int = 6
    # Range over which each per-mode fraction is sampled. The lower
    # bound is intentionally negative so a fraction of zero is
    # ~33% likely (clamps to 0 before scaling).
    contact_f_range: tuple[float, float] = (-0.1, 0.2)
    contact_rank_mean: float = 2.0
    distance_rank_mean: float = 0.0
    rank_std: float = 1.0
    residue_plddt_min: float = 70.0
    plddt_bin_edges: tuple[float, ...] = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)


def _plddt_bin_token(global_plddt: float, bin_edges: tuple[float, ...]) -> str:
    """Map a per-structure mean pLDDT to the matching ``plddt_*`` token."""
    for i, edge in enumerate(bin_edges):
        if global_plddt < edge:
            if i == 0:
                return f"plddt_lt{int(edge)}"
            return f"plddt_{int(bin_edges[i-1])}_{int(edge)}"
    return f"plddt_{int(bin_edges[-1])}_100"


def _distance_token(dist_angstrom: float) -> str:
    """Map a distance to a ``<d_X.X>`` bin token (64 bins, 0.5 Å width)."""
    bin_idx = int(math.ceil(dist_angstrom / 0.5))
    if bin_idx < 1:
        bin_idx = 1
    if bin_idx > 64:
        bin_idx = 64
    return f"d{bin_idx * 0.5:.1f}"


@dataclass(frozen=True)
class _Statement:
    rank: float
    tokens: tuple[str, ...]


def _generate_one(
    structure: ParsedStructure,
    *,
    context_length: int,
    cfg: GenerationConfig,
) -> str | None:
    """Build one document string for ``structure``, or None if it doesn't fit."""
    residues = structure.residues
    num_residues = len(residues)
    if num_residues < 2:
        return None

    # Deterministic seed per entry — keeps generation reproducible
    # across re-runs.
    seed = int(hashlib.sha1(structure.entry_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Contact eligibility: pLDDT-filtered CB/CA positions.
    cb_positions: dict[int, tuple[float, float, float]] = {}
    for r in residues:
        if r.plddt < cfg.residue_plddt_min:
            continue
        pos = cb_or_ca_position(r)
        if pos is not None:
            cb_positions[r.index] = pos

    contacts_long: list[tuple[int, int]] = []
    contacts_medium: list[tuple[int, int]] = []
    contacts_short: list[tuple[int, int]] = []
    cb_indices = sorted(cb_positions)
    for ii in range(len(cb_indices)):
        for jj in range(ii + 1, len(cb_indices)):
            i = cb_indices[ii]
            j = cb_indices[jj]
            sep = j - i
            if sep <= 1 or sep < cfg.short_range_sep:
                continue
            if euclidean(cb_positions[i], cb_positions[j]) > cfg.contact_cutoff_angstrom:
                continue
            if sep >= cfg.long_range_sep:
                contacts_long.append((i, j))
            elif sep >= cfg.medium_range_sep:
                contacts_medium.append((i, j))
            else:
                contacts_short.append((i, j))

    # Token budgeting: 5 overhead = <task> + <begin_sequence> +
    # <begin_statements> + <end> + <plddt_bin>; plus the residue tokens.
    fixed_overhead = 5 + num_residues
    available_tokens = context_length - fixed_overhead
    if available_tokens <= 0:
        return None

    f_long = max(0.0, rng.uniform(*cfg.contact_f_range))
    f_medium = max(0.0, rng.uniform(*cfg.contact_f_range))
    f_short = max(0.0, rng.uniform(*cfg.contact_f_range))

    n_long = min(int(available_tokens * f_long) // CONTACT_TOKENS_PER_STATEMENT, len(contacts_long))
    n_medium = min(int(available_tokens * f_medium) // CONTACT_TOKENS_PER_STATEMENT, len(contacts_medium))
    n_short = min(int(available_tokens * f_short) // CONTACT_TOKENS_PER_STATEMENT, len(contacts_short))

    contact_tokens_used = (n_long + n_medium + n_short) * CONTACT_TOKENS_PER_STATEMENT
    n_distance = (available_tokens - contact_tokens_used) // DISTANCE_TOKENS_PER_STATEMENT

    selected_long = rng.sample(contacts_long, n_long) if n_long > 0 else []
    selected_medium = rng.sample(contacts_medium, n_medium) if n_medium > 0 else []
    selected_short = rng.sample(contacts_short, n_short) if n_short > 0 else []

    distance_indices = [r.index for r in residues if r.atoms]
    distance_atoms = {r.index: r.atoms for r in residues if r.atoms}

    distance_statements: list[tuple[int, int, str, str, str]] = []
    if len(distance_indices) >= 2:
        for _ in range(n_distance):
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
            ai_name, ax, ay, az = rng.choice(distance_atoms[i])
            aj_name, bx, by, bz = rng.choice(distance_atoms[j])
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            distance_statements.append((i, j, ai_name, aj_name, _distance_token(d)))

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

    out: list[str] = []
    out.append(f"<{NAME}>")
    out.append("<begin_sequence>")
    for r in residues:
        out.append(f"<{r.name}>")
    out.append("<begin_statements>")
    for idx, stmt in enumerate(statements):
        if plddt_insert_idx is not None and idx == plddt_insert_idx:
            out.append(f"<{plddt_token}>")
        out.extend(stmt.tokens)
    if plddt_insert_idx is not None and plddt_insert_idx >= len(statements):
        out.append(f"<{plddt_token}>")
    if plddt_at_end:
        out.append(f"<{plddt_token}>")
    out.append("<end>")
    return " ".join(out)


def generate_documents(
    *,
    input_path,
    num_docs: int | None,
    context_length: int,
    config: GenerationConfig,
) -> Iterator[str]:
    """Yield one document per input structure (up to ``num_docs``).

    The driving entry point — ``cli.py`` parses args and calls this
    with the assembled ``GenerationConfig``. Structures for which the
    context budget is exhausted are silently skipped (see
    :func:`_generate_one`).
    """
    produced = 0
    for structure in iter_parsed_structures(input_path):
        doc = _generate_one(structure, context_length=context_length, cfg=config)
        if doc is None:
            continue
        yield doc
        produced += 1
        if num_docs is not None and produced >= num_docs:
            return
