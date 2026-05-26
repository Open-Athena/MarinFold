# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v2 document generation.

A serialization-only delta on top of v1: same structure parsing,
same contact + distance statement selection, same rank-ordering,
same pLDDT placement, same ``<end>``. The only addition is
``<think>`` (a.k.a. pause) tokens placed at slot boundaries between
statements. At training time these are loss-masked, so they give
the model "free" compute it can spend without committing to a
prediction (see https://arxiv.org/abs/2310.02226).

Per-doc algorithm (one doc per input structure, deterministic given
the structure's ``entry_id``):

1. Sample the think-token budget first:
   - With probability ``think_initial_prob`` (0.75), sample
     ``k1 ~ Geometric(p=think_initial_geom_p)`` (support ≥ 1)
     think tokens to place at slot 0 — i.e. right after
     ``<begin_statements>``. With probability 0.25, ``k1 = 0``.
   - Sample ``k2 ~ Uniform(-4, 4)`` and place
     ``max(int(k2), 0)`` additional think runs at random slots
     ∈ [0, N-1]. Each run length ~ ``Geometric(p=0.25)``.
2. Subtract the total think-token count from the 8192 context
   budget before allocating statements, so the doc still fits.
3. Generate statements as in v1.
4. Serialize ``<contacts-and-distances-v2> <begin_sequence> <AAs>
   <begin_statements> [slot-0 think] [maybe pLDDT] [stmt_0] [slot-1
   think] [maybe pLDDT] [stmt_1] ... <end>``. At each slot, think
   tokens come *before* any pLDDT inserted there — matching the
   issue's wording that ``<think>`` lands "immediately after
   ``<begin_statements>``".

The sampling order in ``_generate_one`` is deliberate: think
overhead → contact eligibility → per-mode fractions → statement
sampling → ranks → pLDDT placement → think-slot assignment. Any
reordering would shuffle the RNG stream and break determinism
against a previously generated dataset.
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
from vocab import NAME, THINK_TOKEN


CONTACT_TOKENS_PER_STATEMENT = 3      # <mode> <p_i> <p_j>
DISTANCE_TOKENS_PER_STATEMENT = 6     # <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for :func:`generate_documents`.

    The non-``think_*`` fields reproduce v1's ``GenerationConfig``
    defaults exactly. The ``think_*`` fields are the new knobs that
    control pause-token insertion — defaults are the values pinned
    by the issue ("0.75 / Geom(0.13) / Uniform(-4, 4) / Geom(0.25)").
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
    # ---- think-token knobs (v2-only) ---------------------------------------
    # P(any think tokens right after <begin_statements>).
    think_initial_prob: float = 0.75
    # Geometric p for k1 (length of the initial run, support ≥ 1).
    think_initial_geom_p: float = 0.13
    # Uniform range for k2; n_additional_runs = max(int(k2), 0).
    think_additional_count_range: tuple[float, float] = (-4.0, 4.0)
    # Geometric p for the length of each additional run (support ≥ 1).
    think_run_length_geom_p: float = 0.25


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


def _geometric(rng: random.Random, p: float) -> int:
    """Sample from Geometric(p) with support {1, 2, 3, ...}.

    Uses inverse-CDF on a uniform sample from ``rng`` so the result
    is deterministic in the same RNG stream as the rest of
    generation. Defined for ``p ∈ (0, 1]``; ``p = 1`` always returns
    1 (every trial succeeds immediately).
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p!r}")
    if p == 1.0:
        return 1
    # rng.random() ∈ [0, 1); 1 - U ∈ (0, 1] avoids log(0).
    u = 1.0 - rng.random()
    return int(math.ceil(math.log(u) / math.log(1.0 - p)))


def _sample_think_overhead(
    rng: random.Random, cfg: GenerationConfig,
) -> tuple[int, list[int]]:
    """Sample ``(k1, additional_run_lengths)`` for one document.

    ``k1`` is the length of the run placed right after
    ``<begin_statements>`` (0 if the 0.75 gate misses).
    ``additional_run_lengths`` are the lengths of the extra runs
    placed at random inter-statement slots. Caller then assigns each
    one to a slot in ``[0, N-1]``.
    """
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


def _generate_one(
    structure: ParsedStructure,
    *,
    context_length: int,
    cfg: GenerationConfig,
) -> str | None:
    """Build one v2 document string for ``structure``, or None if it doesn't fit."""
    residues = structure.residues
    num_residues = len(residues)
    if num_residues < 2:
        return None

    seed = int(hashlib.sha1(structure.entry_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Sample think-token overhead first so it can be subtracted from
    # the statement budget before any v1-style allocation. Keeping
    # this call ahead of the contact / distance sampling also pins
    # the RNG stream layout: callers reproducing a dataset will get
    # the same docs from the same entry_id regardless of how the
    # structure happens to be parsed downstream.
    k1, additional_run_lengths = _sample_think_overhead(rng, cfg)
    total_think_tokens = k1 + sum(additional_run_lengths)

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
    # <begin_statements> + <end> + <plddt_bin>; plus residue tokens;
    # plus the pre-sampled think-token overhead.
    fixed_overhead = 5 + num_residues + total_think_tokens
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

    # Assign each additional run to a slot in [0, N-1] uniformly at
    # random, with replacement. Slot i means "right before
    # statements[i]". Multiple runs can land in the same slot and
    # are concatenated (the issue doesn't distinguish "one long run"
    # from "two adjacent runs"). When there are no statements, the
    # only valid slot for the initial run is slot 0 in
    # ``think_at_slot``; additional runs are simply dropped since
    # there is no "before stmt i" to anchor them.
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
    for r in residues:
        out.append(f"<{r.name}>")
    out.append("<begin_statements>")
    for idx, stmt in enumerate(statements):
        # Think tokens always come *before* any pLDDT at the same
        # slot so the issue's "immediately after <begin_statements>"
        # is satisfied for slot 0.
        n_think = think_at_slot.get(idx, 0)
        if n_think:
            out.extend([THINK_TOKEN] * n_think)
        if plddt_insert_idx is not None and idx == plddt_insert_idx:
            out.append(f"<{plddt_token}>")
        out.extend(stmt.tokens)
    if not statements and k1 > 0:
        # No statements but the initial run still landed — emit it
        # so the document captures the sampled overhead. (This is a
        # rare edge case; only triggers on extremely short proteins.)
        out.extend([THINK_TOKEN] * k1)
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
    """Yield one v2 document per input structure (up to ``num_docs``).

    Structures whose context budget is exhausted by think + sequence
    overhead are silently skipped — same convention as v1.
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
