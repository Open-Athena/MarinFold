# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference-free geometric self-consistency of a contacts-v1 contact set (issue #211).

Given a contact set on ``L`` residues — and *nothing else*, no ground truth — how
close does it come to being realizable as a single 3D structure?

Three tiers, cheapest first. Only T3 is the headline metric; see the issue's
Phase 0 for why.

**T1 — packing** (:func:`packing_score`, O(L)). The largest number of contact
partners any one residue claims, against the ceiling measured on real structures.
Catches gross impossibilities for free.

**T2 — triangle-inequality bound smoothing** (:func:`smooth_upper_bounds`,
``scipy`` all-pairs shortest path). Upper bounds from the contacts (``U_CONTACT``)
and the CA(i)-CA(i+1) virtual bond (``BOND``) propagate through the triangle
inequality to give the tightest triangle-consistent bound on every pair.

*This tier does not discriminate and is not the metric.* Phase 0 measured 0
violations for a true contact set **and** 0 for a separation-matched random one, at
four different bound pairs. The reason is structural: a violation needs a path of
upper bounds summing below some lower bound, and any 2-hop contact path is
``2 * U_CONTACT`` while a contact plus *k* backbone steps is ``U_CONTACT + BOND*k``
— nothing reaches under a ~10 A lower bound once ``min_seq_separation=6`` has
excluded the close-in-sequence pairs. Triangle smoothing tests feasibility in an
arbitrary **metric space**, which a contact graph satisfies nearly for free; it
does not test feasibility in **R^3**. It is kept because (a) the null result is
worth having on the record and (b) it is the correct *preconditioner* for T3 —
this is exactly the bound-smoothing step of the classic Crippen-Havel EMBED
distance-geometry algorithm, and initializing the embedder from smoothed bounds
beats a random start.

**T3 — 3D embeddability residual** (:func:`embed_residual`). The metric. Minimize a
bound-violation energy over ``x in R^{L x 3}``

    bonds        ||x_i - x_{i+1}||  = BOND
    contacts     ||x_i - x_j||     <= U_CONTACT
    non-contacts ||x_i - x_j||     >= L_NONCONTACT   (|i-j| >= MIN_SEP only)
    steric       ||x_i - x_j||     >= D_MIN          (all pairs)

and report the residual contact violation. Because this asks whether an embedding
*exists*, the score is the **minimum over ``n_restarts``** independent runs — which
is also what stops an unlucky optimizer run from masquerading as an inconsistent
contact set.

Batched: every contact set for one protein has the same ``L`` and solves in
lockstep, so a protein's ~600 arm/rollout sets go through as one ``(B, L, 3)``
tensor. On CPU this is ~3.5 s per set at L=92; batched on an H100 it is the only
way the 1.3M-embedding budget closes.

**The bounds are statistical, not physical.** Phase 0 measured, on a real
structure, contact CA-CA distances spanning 4.0-13.7 A and non-contact CA-CA
distances starting at 4.1 A — 17.5% of non-contact pairs sit closer than the
contact p99. pyconfind contacts are *side-chain* contacts, so CA-CA distance is
only a proxy and no threshold pair cleanly separates the two populations.
Consequently a nonzero residual means "less geometrically consistent", **never**
"provably unrealizable". Defaults here are placeholders pinned by
``calibrate_bounds.py`` against the 554 ground-truth structures; the comparison
between arms is valid regardless, because every arm is scored under identical
bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Geometry constants
# --------------------------------------------------------------------------

# CA(i)-CA(i+1) virtual bond. Phase 0 on 1QYS: 3.798 +/- 0.017 A (min 3.760, max
# 3.851) — tight enough to treat as an equality constraint rather than a range.
BOND = 3.80

# Upper bound on the CA-CA distance of a pyconfind contact, and lower bound for a
# declared non-contact. Placeholders; calibrate_bounds.py pins them as quantiles
# over the 554-protein GT set. See the module docstring on why these overlap.
U_CONTACT = 12.0
L_NONCONTACT = 6.0

# Steric floor on every CA-CA pair. Phase 0: the closest non-bonded CA pair in
# 1QYS was 4.01 A.
D_MIN = 4.0

# contacts-v1's definitional minimum primary-sequence separation (SPEC: a pair is
# only ever a contact when |i - j| >= 6). Pairs closer than this are neither
# contacts nor non-contacts — they carry no lower-bound constraint at all.
MIN_SEP = 6


@dataclass(frozen=True)
class Bounds:
    """The distance-bound system a contact set is scored against."""

    bond: float = BOND
    u_contact: float = U_CONTACT
    l_noncontact: float = L_NONCONTACT
    d_min: float = D_MIN
    min_sep: int = MIN_SEP


# --------------------------------------------------------------------------
# Contact-set representation
# --------------------------------------------------------------------------


def contact_matrix(pairs, length: int) -> np.ndarray:
    """Build the symmetric boolean contact matrix for ``pairs`` on ``length`` residues.

    ``pairs`` is any iterable of ``(i, j)`` 0-based sequence indices in either
    order; the matrix is symmetric and the diagonal is False. Pairs outside
    ``[0, length)`` raise — a rollout that emits an out-of-range position is a
    parsing bug upstream, not something to silently drop here.
    """
    m = np.zeros((length, length), dtype=bool)
    for i, j in pairs:
        if not (0 <= i < length and 0 <= j < length):
            raise ValueError(f"contact ({i}, {j}) out of range for L={length}")
        if i == j:
            continue
        m[i, j] = m[j, i] = True
    return m


def separation(length: int) -> np.ndarray:
    """``|i - j|`` for every residue pair."""
    idx = np.arange(length)
    return np.abs(idx[:, None] - idx[None, :])


# --------------------------------------------------------------------------
# T1 — packing
# --------------------------------------------------------------------------


def packing_score(mask: np.ndarray) -> dict[str, float]:
    """Contact-degree statistics: how many partners does each residue claim?

    Returns the max and mean number of contact partners per residue. A contact set
    whose max exceeds what real structures ever reach is impossible on packing
    grounds alone, with no geometry computed. ``calibrate_bounds.py`` measures the
    empirical ceiling; this function just reports the statistic.
    """
    deg = mask.sum(axis=1)
    return {
        "max_degree": float(deg.max()) if deg.size else 0.0,
        "mean_degree": float(deg.mean()) if deg.size else 0.0,
        "n_contacts": float(np.triu(mask, 1).sum()),
    }


# --------------------------------------------------------------------------
# T2 — triangle-inequality bound smoothing (Floyd-Warshall / EMBED step 1)
# --------------------------------------------------------------------------


def smooth_upper_bounds(mask: np.ndarray, bounds: Bounds = Bounds()) -> np.ndarray:
    """Tightest triangle-consistent upper bounds implied by the contacts + chain.

    Builds the sparse upper-bound graph — ``bond`` on every ``(i, i+1)`` backbone
    edge, ``u_contact`` on every contact — and takes all-pairs shortest paths. The
    result ``U_hat[i, j]`` is the smallest distance the triangle inequality lets
    ``(i, j)`` be forced *below*; no embedding can exceed it.

    Uses Dijkstra rather than literal Floyd-Warshall: the graph is sparse
    (``O(L)`` backbone + ``O(L)`` contact edges), so ``O(V E log V)`` beats
    ``O(V^3)`` at every ``L`` in the eval set, and both compute the same closure.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    length = mask.shape[0]
    rows, cols, vals = [], [], []
    idx = np.arange(length - 1)
    rows += idx.tolist()
    cols += (idx + 1).tolist()
    vals += [bounds.bond] * (length - 1)
    ci, cj = np.nonzero(np.triu(mask, 1))
    rows += ci.tolist()
    cols += cj.tolist()
    vals += [bounds.u_contact] * len(ci)
    g = csr_matrix((vals, (rows, cols)), shape=(length, length))
    return shortest_path(g, method="D", directed=False)


def triangle_violations(mask: np.ndarray, bounds: Bounds = Bounds()) -> dict[str, float]:
    """Pairs whose smoothed upper bound falls below their declared lower bound.

    A nonzero count is a *proof* of infeasibility under the given bounds (in any
    metric space, let alone R^3). Phase 0 found this fires for neither real nor
    random contact sets — see the module docstring — so expect zeros and report
    them.
    """
    length = mask.shape[0]
    upper = smooth_upper_bounds(mask, bounds)
    sep = separation(length)
    triu = np.triu(np.ones((length, length), dtype=bool), 1)
    # Only declared non-contacts at or beyond min_sep carry a lower bound.
    scored = triu & (sep >= bounds.min_sep) & ~mask
    viol = scored & (upper < bounds.l_noncontact)
    return {
        "n_triangle_violations": float(viol.sum()),
        "n_scored_pairs": float(scored.sum()),
        "max_implied_diameter": float(upper[np.isfinite(upper)].max()),
    }


# --------------------------------------------------------------------------
# T3 — 3D embeddability residual (the metric)
# --------------------------------------------------------------------------


def _pair_index(masks: np.ndarray, bounds: Bounds):
    """Per-set index arrays for the contact and non-contact constraint groups.

    ``masks`` is ``(B, L, L)``. Sets in a batch have different contact counts, so
    the constraint lists are ragged; they are returned flattened with a parallel
    batch index, which is what ``index_add_`` wants anyway.
    """
    b, length, _ = masks.shape
    sep = separation(length)
    triu = np.triu(np.ones((length, length), dtype=bool), 1)
    far = triu & (sep >= bounds.min_sep)

    cb, ci, cj = np.nonzero(masks & triu[None])
    nb, ni, nj = np.nonzero(far[None] & ~masks)
    return (cb, ci, cj), (nb, ni, nj)


def embed_residual(
    masks: np.ndarray,
    bounds: Bounds = Bounds(),
    *,
    n_restarts: int = 4,
    iters: int = 3000,
    lr: float = 0.25,
    seed: int = 0,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Residual bound violation after embedding each contact set into R^3.

    Args:
        masks: ``(B, L, L)`` boolean contact matrices, or a single ``(L, L)``.
            Every set in a batch must share ``L`` — they are one protein's arms.
        bounds: the distance-bound system.
        n_restarts: independent random initializations. The reported score is the
            **minimum** over restarts: the question is whether an embedding
            exists, so a single bad optimizer run must not count as evidence of
            inconsistency.
        iters: Adam steps per restart.
        lr: Adam learning rate, in Angstroms of coordinate step.
        seed: base RNG seed; restart ``r`` uses ``seed + r``. **A batch shares one
            RNG stream**, so a set's initial coordinates depend on its position in
            the batch and a score is reproducible only for a fixed batch
            composition. That is deliberate: batches are one protein's arms, so
            every arm faces the same draw of optimization landscapes, which is
            what the paired comparison wants. Do not read across two runs that
            grouped sets differently.
        device: torch device. Defaults to CUDA when available.

    Returns:
        One dict per input set, each with the min-over-restarts residuals:
        ``contact_excess`` (total Angstroms by which contacts exceed
        ``u_contact`` — the headline), ``contact_excess_per_contact``,
        ``unsat_frac`` (fraction of contacts still over by >0.5 A),
        ``noncontact_violation``, ``bond_err``, and ``rg`` (radius of gyration,
        reported so a compactness artifact would be visible).
    """
    import torch

    single = masks.ndim == 2
    if single:
        masks = masks[None]
    masks = np.ascontiguousarray(masks.astype(bool))
    b, length, _ = masks.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    (cb, ci, cj), (nb, ni, nj) = _pair_index(masks, bounds)
    t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.long)  # noqa: E731
    cb_, ci_, cj_ = t(cb), t(ci), t(cj)
    nb_, ni_, nj_ = t(nb), t(ni), t(nj)
    n_contacts = torch.as_tensor(
        np.triu(masks, 1).sum(axis=(1, 2)), device=dev, dtype=torch.float32
    )

    bi = torch.arange(length - 1, device=dev)
    au, av = torch.triu_indices(length, length, offset=1, device=dev)

    # Initial coordinate scale. A compact globule of L residues has
    # Rg ~ 2.2 * L^0.38 A, and an isotropic gaussian cloud with per-axis sigma s
    # has Rg = sqrt(3) * s — so seed with s = Rg_target / sqrt(3). Getting this
    # wrong matters: an initial cloud 2x too spread leaves the ground-truth set
    # short of a zero residual at fixed iteration count (measured: 1.79 vs 0.09
    # contact excess on 1QYS), because the optimizer spends its budget
    # collapsing the globule instead of satisfying bounds.
    scale = 2.2 * length**0.38 / np.sqrt(3.0)

    best: dict[str, torch.Tensor] | None = None
    for r in range(n_restarts):
        g = torch.Generator(device=dev).manual_seed(seed + r)
        x = torch.randn(b, length, 3, generator=g, device=dev) * scale
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            db = (x[:, bi] - x[:, bi + 1]).norm(dim=-1)
            dc = (x[cb_, ci_] - x[cb_, cj_]).norm(dim=-1)
            dn = (x[nb_, ni_] - x[nb_, nj_]).norm(dim=-1)
            da = (x[:, au] - x[:, av]).norm(dim=-1)
            loss = (
                ((db - bounds.bond) ** 2).sum()
                + (torch.clamp(dc - bounds.u_contact, min=0) ** 2).sum()
                + (torch.clamp(bounds.l_noncontact - dn, min=0) ** 2).sum()
                + (torch.clamp(bounds.d_min - da, min=0) ** 2).sum()
            )
            loss.backward()
            opt.step()

        with torch.no_grad():
            db = (x[:, bi] - x[:, bi + 1]).norm(dim=-1)
            dc = (x[cb_, ci_] - x[cb_, cj_]).norm(dim=-1)
            dn = (x[nb_, ni_] - x[nb_, nj_]).norm(dim=-1)
            zeros = lambda: torch.zeros(b, device=dev)  # noqa: E731

            excess = zeros().index_add_(0, cb_, torch.clamp(dc - bounds.u_contact, min=0))
            unsat = zeros().index_add_(
                0, cb_, ((dc - bounds.u_contact) > 0.5).float()
            ) / n_contacts.clamp(min=1)
            ncviol = zeros().index_add_(0, nb_, torch.clamp(bounds.l_noncontact - dn, min=0))
            bond_err = (db - bounds.bond).abs().mean(dim=1)
            rg = (x - x.mean(dim=1, keepdim=True)).pow(2).sum(-1).mean(-1).sqrt()

            cand = {
                "contact_excess": excess,
                "unsat_frac": unsat,
                "noncontact_violation": ncviol,
                "bond_err": bond_err,
                "rg": rg,
            }
            if best is None:
                best = cand
            else:
                # Keep, per set, the restart with the lowest contact excess —
                # every other field follows that same restart so the row stays a
                # coherent description of one embedding.
                take = cand["contact_excess"] < best["contact_excess"]
                best = {k: torch.where(take, cand[k], best[k]) for k in cand}

    assert best is not None
    n_c = n_contacts.clamp(min=1)
    out = {k: v.cpu().numpy() for k, v in best.items()}
    out["contact_excess_per_contact"] = out["contact_excess"] / n_c.cpu().numpy()
    rows = [{k: float(v[i]) for k, v in out.items()} for i in range(b)]
    return rows[0] if single else rows


def score_contact_set(
    pairs, length: int, bounds: Bounds = Bounds(), **embed_kwargs
) -> dict[str, float]:
    """All three tiers for one contact set. Convenience wrapper; batch for real runs."""
    mask = contact_matrix(pairs, length)
    return {
        **packing_score(mask),
        **triangle_violations(mask, bounds),
        **embed_residual(mask, bounds, **embed_kwargs),
    }
