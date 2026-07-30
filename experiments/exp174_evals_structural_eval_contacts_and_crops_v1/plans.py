# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The inference plans from ``PLANS.md``, as functions from a model to coordinates.

Each plan takes a :class:`~sampler.Sampler`, one eval record, and returns a
:class:`~document_codec.CoordinateEstimate` plus per-record stats. They share
the decode contract in ``document_codec`` and differ only in what they put in
the prompt and how many times they ask.

* :func:`plan_a` — one free-running document. The in-distribution control.
* :func:`plan_e1` — teacher-force the *true contacts*, generate the coordinate
  section. The diagnostic that splits "cannot predict the contact map" from
  "cannot turn a contact map into 3D".
* :func:`plan_e2` — teacher-force the *true* Pass-1 boxes, generate only crops.
  The oracle that gates C and F: can the model refine a box it was handed?
* :func:`plan_c` — one forced crop sweep off a shared Pass-1 prefix, no
  neighbour context, one sample per box. Plan F's ablation.
* :func:`plan_f` — neighbour-conditioned iterative refinement. The plan of
  record: a spatially coherent sweep that conditions each crop on its
  already-refined neighbours and its own earlier visits, K samples per box,
  repeated until the coordinates stop moving.

**Why every plan stays in one frame.** A document's coordinates live in a
random rotated + translated frame chosen when the document is generated. Plans
A, C, E2 and F all hang off a *single* generated Pass-1 section, so every later
crop is in that same frame and nothing needs registering. Only a plan that
merged across independently-generated documents (Plan B) would have to estimate
a rigid transform, which is why B is not implemented here.
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from document_codec import (
    CROP_ATOM_TOKENS,
    CROP_HEADER_TOKENS,
    CoordinateEstimate,
    count_contacts,
    crop_header,
    estimate_to_atom_array,
    pass1_budget,
    parse_observations,
    place_in_cube,
    render_crop,
    sequence_prefix,
    start_index,
    synthesize_pass1,
)
from marinfold.document_structures.contacts_and_crops_v1 import (
    CONTEXT_LENGTH,
    GenerationConfig,
)
from marinfold.document_structures.contacts_and_crops_v1.vocab import (
    CONTACT_TOKEN,
    position_token,
)
from marinfold.document_structures.contacts_and_crops_v1 import NUM_POSITION_INDICES
from sampler import SamplingConfig

# How many neighbouring crops Plan F puts in context around a target box. The
# format's own Pass-2 shows ~20 crops in a document's fine reserve, so staying
# under that keeps the prompt the shape of a real training document. Six is the
# face-neighbourhood; the 26-neighbourhood would blow the budget.
DEFAULT_NEIGHBOR_CROPS = 6

# How many of a box's own earlier visits go back into its prompt. Two is enough
# to put the visit index at 2 (σ = 1/9 Å on the training schedule) without
# spending the whole fine reserve on one box's history.
MAX_OWN_VISITS_IN_CONTEXT = 2


@dataclass
class PlanResult:
    """What a plan produced for one record."""

    estimate: CoordinateEstimate
    stats: dict = field(default_factory=dict)


def _valid_atoms_from_gt(gt, length: int) -> list[frozenset]:
    """Per-residue atom names the ground truth has, for decode-time filtering.

    Dropping a mention of an atom the residue does not have costs nothing — it
    could not be scored anyway — and keeps hallucinated atoms out of the
    coverage count.
    """
    valid: list[set] = [set() for _ in range(length)]
    for res_id, atom_name in zip(gt.res_id.tolist(), gt.atom_name.tolist()):
        if 1 <= res_id <= length:
            valid[res_id - 1].add(atom_name)
    return [frozenset(names) for names in valid]


def _neighbors(cell):
    """The 26-neighbourhood of a cell, in-bounds on the 100³ grid."""
    cx, cy, cz = cell
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                n = (cx + dx, cy + dy, cz + dz)
                if all(0 <= v < 100 for v in n):
                    out.append(n)
    return out


def spatial_order(cells, rng: random.Random):
    """A spatially coherent walk over occupied cells — the "flashlight" path.

    Breadth-first from a random occupied seed, following the 26-neighbourhood,
    restarting at another unvisited cell when a component is exhausted. This is
    what makes a Plan-F sweep look like the format's own Pass-2, which picks
    45 % of its boxes from the frontier of already-shown ones.
    """
    remaining = set(cells)
    order = []
    while remaining:
        seed = sorted(remaining)[rng.randrange(len(remaining))]
        queue = deque([seed])
        remaining.discard(seed)
        while queue:
            cell = queue.popleft()
            order.append(cell)
            for neighbor in _neighbors(cell):
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    queue.append(neighbor)
    return order


def plan_a(sampler, record, *, config: SamplingConfig, gt=None, seed: int = 0) -> PlanResult:
    """One free-running document: condition on the sequence, generate the rest."""
    sequence = record["input_seq"]
    entry_id = record["record_id"]
    prefix = sequence_prefix(entry_id, sequence)
    if prefix is None:
        return PlanResult(CoordinateEstimate(), {"status": "unserializable"})
    start = start_index(entry_id, sequence)
    prompt_ids = sampler.encode(prefix)

    import torch

    generator = torch.Generator(device=sampler.device).manual_seed(seed)
    started = time.time()
    sampled = sampler.sample(
        prompt_ids,
        n_samples=1,
        config=config,
        max_new_tokens=CONTEXT_LENGTH - len(prompt_ids),
        generator=generator,
    )[0]
    elapsed = time.time() - started

    tokens = sampler.decode(sampled)
    valid_atoms = _valid_atoms_from_gt(gt, len(sequence)) if gt is not None else None
    estimate = CoordinateEstimate()
    n_pass1 = n_crop = 0
    for observation in parse_observations(
        tokens, start=start, length=len(sequence), valid_atoms=valid_atoms
    ):
        estimate.add(observation)
        n_pass1 += observation.source == "pass1"
        n_crop += observation.source == "crop"

    return PlanResult(
        estimate,
        {
            "status": "ok",
            "plan": "A",
            "generated_tokens": len(sampled),
            "decoded_pass1": n_pass1,
            "decoded_crop": n_crop,
            # Nothing is teacher-forced in Plan A, so any contacts here are the
            # model's own — worth recording, since they are what it conditioned
            # its coordinate section on.
            "self_emitted_contacts": count_contacts(tokens),
            "decoded_atoms": len(estimate),
            "refined_atoms": len(estimate.refined_keys()),
            "elapsed_seconds": elapsed,
        },
    )


def plan_e1(
    sampler, record, gt=None, *, config: SamplingConfig, seed: int = 0,
    contacts: list | None = None, n_contacts: int | None = None,
) -> PlanResult:
    """Teacher-force the true contacts, then generate the coordinate section.

    E2 showed that the model refines a box it is handed about as well as ground
    truth does, so the deficit is Pass 1 — the fold. E1 asks whether the fold
    fails because the model cannot infer the contact map, or because it cannot
    turn one into 3D coordinates: it writes real contacts into the contacts
    section and lets the model generate everything downstream.

    **How many contacts.** The format never shows more than
    ``n_contacts_max`` (50) in a document, and it samples them *uniformly* from
    the eligible pool rather than strongest-first (SPEC → Contacts). So
    "teacher-force the true contacts" can only mean "the 50-ish the format is
    able to show", sampled the same way — anything more would be a prompt the
    model has never seen. The contacts also cost 3 tokens each, which the
    Pass-1 budget accounts for.

    Args:
        contacts: ``(i, j, degree)`` triples in 0-based input-sequence
            coordinates (the ``gt_contacts.jsonl`` payload, which is *unfiltered*
            — this function applies the format's eligibility rule itself).
            Empty or ``None`` degrades to Plan A.
        n_contacts: how many to force; defaults to the format's own maximum.
    """
    import torch

    sequence = record["input_seq"]
    entry_id = record["record_id"]
    prefix = sequence_prefix(entry_id, sequence)
    if prefix is None:
        return PlanResult(CoordinateEstimate(), {"status": "unserializable"})
    start = start_index(entry_id, sequence)
    limit = n_contacts if n_contacts is not None else GenerationConfig().n_contacts_max

    rng = random.Random(seed)
    # Filter to the contacts a real contacts section can contain, THEN sample.
    # The gt bundle carries every degree>0 pair, but the format emits only
    # separation >= min_seq_separation and degree >= min_contact_degree — and
    # only ~39 % of the raw list clears that bar. Sampling the raw list wastes
    # most of the 50-contact budget on short-range pairs the model never saw in
    # a contacts section and which are nearly implied by the chain anyway, which
    # makes the diagnostic far weaker than it looks.
    config = GenerationConfig()
    pool = [
        c
        for c in (contacts or [])
        if (c[1] - c[0]) >= config.min_seq_separation
        and c[2] >= config.min_contact_degree
    ]
    n_raw = len(contacts or [])
    chosen = pool if len(pool) <= limit else rng.sample(pool, limit)
    contact_tokens: list[str] = []
    for i, j, _degree in chosen:
        # The format coin-flips each pair's order; do the same.
        a, b = (j, i) if rng.random() < 0.5 else (i, j)
        contact_tokens += [
            CONTACT_TOKEN,
            position_token((start + a) % NUM_POSITION_INDICES),
            position_token((start + b) % NUM_POSITION_INDICES),
        ]

    prompt = prefix + contact_tokens
    prompt_ids = sampler.encode(prompt)
    generator = torch.Generator(device=sampler.device).manual_seed(seed)
    started = time.time()
    sampled = sampler.sample(
        prompt_ids,
        n_samples=1,
        config=config,
        max_new_tokens=max(0, CONTEXT_LENGTH - len(prompt_ids)),
        generator=generator,
    )[0]
    elapsed = time.time() - started

    tokens = sampler.decode(sampled)
    valid_atoms = _valid_atoms_from_gt(gt, len(sequence)) if gt is not None else None
    estimate = CoordinateEstimate()
    for observation in parse_observations(
        tokens, start=start, length=len(sequence), valid_atoms=valid_atoms
    ):
        estimate.add(observation)

    return PlanResult(
        estimate,
        {
            "status": "ok",
            "plan": "E1",
            "contacts_raw": n_raw,
            "contacts_eligible": len(pool),
            "contacts_forced": len(chosen),
            "generated_tokens": len(sampled),
            "decoded_atoms": len(estimate),
            "refined_atoms": len(estimate.refined_keys()),
            "elapsed_seconds": elapsed,
        },
    )


def plan_e2(sampler, record, gt, *, config: SamplingConfig, seed: int = 0) -> PlanResult:
    """Teacher-force the true Pass-1 boxes; the model generates only crops.

    The oracle gate on Plans C and F. The Pass-1 section is synthesized from the
    **ground-truth** coordinates (with the format's own σ=2 Å box noise, so the
    prompt is in-distribution), which pins the frame to the ground truth's own
    frame and hands the model every atom's correct 10 Å cell. Whatever the
    resulting structure scores above the ``box10`` baseline (lDDT 0.323 at full
    coverage) is what the model's crops added.
    """
    sequence = record["input_seq"]
    entry_id = record["record_id"]
    prefix = sequence_prefix(entry_id, sequence)
    if prefix is None:
        return PlanResult(CoordinateEstimate(), {"status": "unserializable"})
    start = start_index(entry_id, sequence)

    # Ground truth as an estimate, so synthesize_pass1 can draw from it exactly
    # the way the generator draws from real coordinates.
    truth = CoordinateEstimate()
    from document_codec import Observation

    for res_id, atom_name, coord in zip(
        gt.res_id.tolist(), gt.atom_name.tolist(), gt.coord
    ):
        truth.add(
            Observation(
                seq_index=res_id - 1,
                atom_name=atom_name,
                position=np.asarray(coord, dtype=np.float64),
                variance=1.0,
                source="crop",
                visit_index=0,
            )
        )

    cap, structure_budget = pass1_budget(len(sequence))
    rng = random.Random(seed)
    # Ground-truth frames routinely have negative coordinates; the format's
    # cube is [0, 1000). Place before boxing, or every negative coordinate
    # clamps to 0 and the "correct boxes" are a heap at the origin.
    truth = place_in_cube(truth, rng)
    pass1_tokens = synthesize_pass1(truth, start=start, cap_tokens=cap, rng=rng)
    prompt = prefix + pass1_tokens
    prompt_ids = sampler.encode(prompt)

    import torch

    generator = torch.Generator(device=sampler.device).manual_seed(seed)
    started = time.time()
    sampled = sampler.sample(
        prompt_ids,
        n_samples=1,
        config=config,
        max_new_tokens=max(0, CONTEXT_LENGTH - len(prompt_ids)),
        generator=generator,
    )[0]
    elapsed = time.time() - started

    valid_atoms = _valid_atoms_from_gt(gt, len(sequence))
    estimate = CoordinateEstimate()
    # The teacher-forced boxes are part of the prediction — they are what the
    # model was conditioned on, and excluding them would score a structure the
    # plan never claimed. They are oracle information, which is why E2 is
    # reported as an upper bound and never next to a real predictor.
    for observation in parse_observations(
        pass1_tokens + sampler.decode(sampled),
        start=start,
        length=len(sequence),
        valid_atoms=valid_atoms,
    ):
        estimate.add(observation)

    return PlanResult(
        estimate,
        {
            "status": "ok",
            "plan": "E2",
            "pass1_tokens": len(pass1_tokens),
            "generated_tokens": len(sampled),
            "decoded_atoms": len(estimate),
            "refined_atoms": len(estimate.refined_keys()),
            "elapsed_seconds": elapsed,
        },
    )


def _generate_pass1(sampler, prompt_ids, cap, config, generator):
    """Sample a Pass-1 section only, stopping at the Pass-1 budget."""
    return sampler.sample(
        prompt_ids,
        n_samples=1,
        config=config,
        max_new_tokens=cap,
        generator=generator,
    )[0]


def plan_c(
    sampler, record, *, config: SamplingConfig, gt=None, seed: int = 0
) -> PlanResult:
    """One forced crop sweep off a shared Pass-1 prefix — Plan F's ablation.

    Generates Pass 1 once, reads the occupied boxes off it, then for each box
    continues from the *same* cached prefix with a forced ``<crop>`` header.
    No neighbour context, no revisits, one sample per box: exactly Plan F with
    everything that distinguishes it turned off.
    """
    return _sweep_plan(
        sampler,
        record,
        config=config,
        gt=gt,
        seed=seed,
        n_sweeps=1,
        n_samples=1,
        n_neighbor_crops=0,
        label="C",
    )


def plan_f(
    sampler,
    record,
    *,
    config: SamplingConfig,
    gt=None,
    seed: int = 0,
    n_sweeps: int = 3,
    n_samples: int = 4,
    n_neighbor_crops: int = DEFAULT_NEIGHBOR_CROPS,
    pass1_feedback_sigma: float | None = None,
    convergence_a: float = 0.1,
) -> PlanResult:
    """Neighbour-conditioned iterative refinement — the plan of record."""
    return _sweep_plan(
        sampler,
        record,
        config=config,
        gt=gt,
        seed=seed,
        n_sweeps=n_sweeps,
        n_samples=n_samples,
        n_neighbor_crops=n_neighbor_crops,
        pass1_feedback_sigma=pass1_feedback_sigma,
        convergence_a=convergence_a,
        label="F",
    )


def _sweep_plan(
    sampler,
    record,
    *,
    config: SamplingConfig,
    gt,
    seed: int,
    n_sweeps: int,
    n_samples: int,
    n_neighbor_crops: int,
    label: str,
    pass1_feedback_sigma: float | None = None,
    convergence_a: float = 0.1,
) -> PlanResult:
    """Shared machinery for Plans C and F.

    The loop, per ``PLANS.md`` §7:

    1. Generate Pass 1 once. This fixes the frame; every later crop lives in it.
    2. Per sweep, synthesize a Pass-1 section from the current estimate and hold
       it **byte-identical for the whole sweep**, so one prefill serves every
       crop in that sweep. This is the 10× in the cost model.
    3. Walk the occupied cells in a spatially coherent order. For each, force a
       prompt suffix of (a few neighbouring crops, already refined) + (this
       box's own earlier visits, which set the visit index the model sees) +
       this box's header, then sample ``n_samples`` bodies.
    4. Fold every decoded observation into the running precision-weighted
       estimate.
    5. Stop when the mean per-atom displacement between sweeps drops below
       ``convergence_a``.
    """
    import torch

    sequence = record["input_seq"]
    entry_id = record["record_id"]
    prefix = sequence_prefix(entry_id, sequence)
    if prefix is None:
        return PlanResult(CoordinateEstimate(), {"status": "unserializable"})
    start = start_index(entry_id, sequence)
    valid_atoms = _valid_atoms_from_gt(gt, len(sequence)) if gt is not None else None
    cap, structure_budget = pass1_budget(len(sequence))

    rng = random.Random(seed)
    generator = torch.Generator(device=sampler.device).manual_seed(seed)
    started = time.time()

    # (1) Pass 1, once. This is the only free-running generation in the plan.
    pass1_ids = _generate_pass1(sampler, sampler.encode(prefix), cap, config, generator)
    pass1_tokens = sampler.decode(pass1_ids)
    estimate = CoordinateEstimate()
    for observation in parse_observations(
        pass1_tokens, start=start, length=len(sequence), valid_atoms=valid_atoms
    ):
        estimate.add(observation)
    if len(estimate) == 0:
        return PlanResult(
            estimate,
            {"status": "empty_pass1", "plan": label, "elapsed_seconds": time.time() - started},
        )

    sweep_stats = []
    previous = {key: estimate.position(key).copy() for key in estimate.keys()}
    n_crop_statements = 0
    # Rendered crops per cell, **persisting across sweeps**. This is what lets a
    # box see its own earlier visits: within a sweep the spatial walk touches
    # each cell once, so a per-sweep history would leave every visit index at 0
    # and the format's σ=1/(i+1)² sharpening would never be requested.
    cell_history: dict[tuple[int, int, int], list[list[str]]] = {}

    for sweep in range(n_sweeps):
        # (2) One Pass-1 section for the whole sweep -> one prefill.
        if sweep == 0:
            sweep_pass1 = pass1_tokens
        else:
            sigma = (
                pass1_feedback_sigma
                if pass1_feedback_sigma is not None
                else None  # default: the format's own σ, applied inside synthesize_pass1
            )
            kwargs = {} if sigma is None else {"noise_sigma": sigma}
            sweep_pass1 = synthesize_pass1(
                estimate, start=start, cap_tokens=cap, rng=rng, **kwargs
            )
        base_prompt = prefix + sweep_pass1
        base_ids = sampler.encode(base_prompt)
        cache, last_logits = sampler.prefill(base_ids)
        budget_left_base = CONTEXT_LENGTH - len(base_ids)

        cells = estimate.occupied_cells()
        order = spatial_order(list(cells), rng)

        for cell in order:
            # (3) Build the forced suffix: neighbouring crops, then this box's
            #     own earlier visits, then its header.
            forced_tokens: list[str] = []
            if n_neighbor_crops:
                for neighbor in _neighbors(cell):
                    rendered = cell_history.get(neighbor)
                    if rendered:
                        forced_tokens += rendered[-1]
                    if len(forced_tokens) >= n_neighbor_crops * CROP_ATOM_TOKENS * 8:
                        break
            # The box's own prior visits raise the visit index the model sees,
            # which is how the format's σ=1/(i+1)² sharpening is *requested*
            # rather than hoped for. These accumulate across sweeps.
            for rendered in cell_history.get(cell, [])[-MAX_OWN_VISITS_IN_CONTEXT:]:
                forced_tokens += rendered
            forced_tokens += crop_header(cell)

            room = budget_left_base - len(forced_tokens)
            if room < CROP_ATOM_TOKENS:
                # No space for even one atom: drop neighbour context and retry
                # with the bare header, which always fits.
                forced_tokens = crop_header(cell)
                room = budget_left_base - len(forced_tokens)
                if room < CROP_ATOM_TOKENS:
                    continue

            bodies = sampler.sample_from_cache(
                cache,
                last_logits,
                len(base_ids),
                n_samples=n_samples,
                config=config,
                # An occupied box holds ~23 atoms on average and 69 at the
                # observed maximum (SPEC → Coverage), so 4 x 80 tokens is a
                # generous cap; the `<crop>` stop is what normally ends a body.
                max_new_tokens=min(room, 4 * 80),
                forced_ids=sampler.encode(forced_tokens),
                stop_token_ids=[sampler.crop_id],
                generator=generator,
            )

            for body in bodies:
                stream = forced_tokens + sampler.decode(body)
                for observation in parse_observations(
                    stream, start=start, length=len(sequence), valid_atoms=valid_atoms
                ):
                    if observation.source != "crop":
                        continue
                    estimate.add(observation)
                    n_crop_statements += 1

            # Record what this box now looks like, for its neighbours' context
            # and for its own next visit.
            members = [k for k in cells.get(cell, ()) if k in estimate]
            if members:
                rng.shuffle(members)
                cell_history.setdefault(cell, []).append(
                    render_crop(cell, members, estimate, start=start)
                )

        # (5) Convergence: how far did atoms move this sweep?
        moved = [
            float(np.linalg.norm(estimate.position(key) - previous[key]))
            for key in estimate.keys()
            if key in previous
        ]
        displacement = float(np.mean(moved)) if moved else float("nan")
        sweep_stats.append(
            {
                "sweep": sweep,
                "n_cells": len(order),
                "mean_displacement_a": displacement,
                "atoms": len(estimate),
                "refined_atoms": len(estimate.refined_keys()),
            }
        )
        previous = {key: estimate.position(key).copy() for key in estimate.keys()}
        if displacement == displacement and displacement < convergence_a:
            break

    return PlanResult(
        estimate,
        {
            "status": "ok",
            "plan": label,
            "n_sweeps_run": len(sweep_stats),
            "pass1_tokens": len(pass1_tokens),
            "self_emitted_contacts": count_contacts(pass1_tokens),
            "crop_statements": n_crop_statements,
            "decoded_atoms": len(estimate),
            "refined_atoms": len(estimate.refined_keys()),
            "sweeps": sweep_stats,
            "elapsed_seconds": time.time() - started,
        },
    )


PLANS = {"A": plan_a, "C": plan_c, "F": plan_f, "E1": plan_e1, "E2": plan_e2}
