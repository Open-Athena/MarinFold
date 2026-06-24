# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-v1 inference + evaluation.

Library module — the CLI surface lives in ``cli.py`` next door, which
imports :func:`predict` and :func:`evaluate` from here, and the top-level
``marinfold infer`` / ``marinfold evaluate`` dispatch into them.

Unlike contacts-and-distances-v1 (which reads an *expected distance* off a
``<d_X.X>`` next-token distribution), a contacts-v1 model predicts
**contacts**: which residue pairs are in side-chain contact. We score it
with the *pairwise* readout exp82/exp89 found best (rollout / iterative did
not beat it):

1. Build the contacts-v1 **sequence-section prefix** from the input
   sequence — the official deterministic :func:`build_document` with an
   empty contact list, truncated to ``… <begin_statements>``. This is
   exactly what the model conditions on at the start of the structure
   section.
2. For every residue pair (i, j) read the autoregressive probability of the
   contact statement ``<contact> <p_i> <p_j>`` — two conditional
   distributions over the protein's position tokens, obtained from the
   backend's :meth:`~marinfold.Backend.next_token_probs` primitive:

       lp1[i]    = log P(<p_i> | prefix, <contact>)
       lp2[i, j] = log P(<p_j> | prefix, <contact>, <p_i>)

   and combine them into an **unordered contact score**

       P(contact)[i, j] = exp(lp1[i] + lp2[i, j]) + exp(lp1[j] + lp2[j, i])

   i.e. P(emit i then j) + P(emit j then i) — symmetric because the training
   documents randomize each pair's orientation. This is the same quantity
   the exp89 heatmaps call ``P(contact)``; ranking by it reproduces the
   exp89 contact-prediction numbers within backend noise.

Two top-level entry points, mirroring the contacts-and-distances-v1 impl:

- :func:`predict` yields one record per input structure: the candidate
  pairs and their ``P(contact)`` score. No ground truth is consulted, so a
  bare sequence (``--input-sequence``) is valid input.

- :func:`evaluate` additionally scores the prediction against pyconfind
  ground-truth contacts (the input file *is* the ground truth): precision @
  {L, L/2, L/5, R} and ranking AUC, per sequence-separation range
  (all / short / medium / long), macro-averaged across structures.

**Test-time augmentation.** contacts-v1 documents carry two random
nuisances — the N-terminal start index and the order of the ``<p> <AA>``
statements. ``ensemble_k`` > 1 resamples the sequence definition that many
times (a different deterministic seed per draw) and averages the per-pair
``P(contact)`` — the Monte-Carlo marginal exp89 used for its headline
"×10 ens" model. It costs ``ensemble_k`` forward passes per structure and
needs no ground truth.

The model forward pass goes through :mod:`marinfold.inference`; this module
is backend-agnostic (vLLM / transformers / MLX).
"""

import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marinfold import Backend, EvalResult, load_backend

from .generate import GenerationConfig, build_document
from .parse import (
    RawContact,
    ResidueInfo,
    iter_analyzed_structures,
    residues_from_sequence,
)
from .vocab import (
    BEGIN_STRUCTURE_TOKEN,
    CONTACT_TOKEN,
    NAME,
    NUM_POSITION_INDICES,
    position_token,
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


# GT contact definition (matches the contacts-v1 training documents and the
# exp89 eval): a pyconfind side-chain contact counts as ground truth when its
# degree clears this floor and the residues are at least ``min_seq_separation``
# apart in the primary sequence.
_MIN_CONTACT_DEGREE = 0.001

# Sequence-separation ranges (CASP convention) used to bin contact-prediction
# metrics. ``(lo, hi)`` is inclusive; ``hi=None`` means unbounded.
_RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None),
    "short": (6, 11),
    "medium": (12, 23),
    "long": (24, None),
}

# Probability floor before taking a log. Backends with full-vocab logits
# (transformers, MLX) return real softmax mass, so this never bites; vLLM
# returns 0 for target tokens outside its top-k logprobs, and this keeps the
# log finite (such pairs simply rank last).
_PROB_FLOOR = 1e-30


# --------------------------------------------------------------------------
# Inference config + structure container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    """Shared inputs to :func:`predict` and :func:`evaluate`.

    The CLI assembles one of these from the parsed args. ``model`` /
    ``input_path`` / ``backend`` / ``batch_size`` / ``dtype`` are the common
    surface the top-level ``marinfold`` CLI fills in; the rest are
    contacts-v1 knobs with defaults.

    ``min_seq_separation`` is the smallest primary-sequence gap |i - j| that
    can be a contact (6 — matching the data), so closer pairs are never
    scored or counted. ``ensemble_k`` is the test-time-augmentation draw
    count (1 = a single deterministic realization). ``top_k_logprobs`` and
    ``gpu_memory_utilization`` are vLLM-only; other backends ignore them.
    ``batch_size`` is the per-prefix tail fan-out the backend batches
    internally. ``keep_matrix`` adds the dense per-structure ``P(contact)``
    matrix to each ``predict`` record.
    """

    model: str | None
    input_path: Path | None = None
    backend: str = "vllm"
    batch_size: int = 64
    dtype: str = "bfloat16"
    min_seq_separation: int = 6
    ensemble_k: int = 1
    top_k_logprobs: int = 256
    gpu_memory_utilization: float = 0.85
    keep_matrix: bool = False


@dataclass(frozen=True)
class ContactStructure:
    """One protein ready for contacts-v1 inference / evaluation.

    ``residues`` is the chain in sequence order (the only thing inference
    needs — contacts-v1 conditions on the sequence). ``gt_contacts`` carries
    the pyconfind ground-truth contacts for :func:`evaluate`; it is ``None``
    for the sequence-only / predict path.
    """

    entry_id: str
    residues: tuple[ResidueInfo, ...]
    gt_contacts: tuple[RawContact, ...] | None = None
    global_plddt: float = math.nan


def structure_from_sequence(
    aa_string: str, *, entry_id: str = "sequence"
) -> ContactStructure:
    """Build a :class:`ContactStructure` from a one-letter AA sequence.

    The entry point for ``marinfold infer --input-sequence ...``: the model
    is queried over residue pairs but no ground truth is consulted, so no
    structure / pyconfind is needed. One-letter codes outside the standard
    20 map to ``<UNK>`` (the same fallback the contacts-v1 generator uses).

    Raises:
        ValueError: fewer than 2 residues after whitespace stripping.
    """
    residues = residues_from_sequence(aa_string)
    if len(residues) < 2:
        raise ValueError(
            f"contacts-v1 inference needs at least 2 residues; got {len(residues)}."
        )
    return ContactStructure(entry_id=entry_id, residues=residues)


# --------------------------------------------------------------------------
# Prompt construction + scoring
# --------------------------------------------------------------------------


def _prefix_and_positions(
    structure: ContactStructure, *, entry_id: str
) -> tuple[str, list[int], int] | None:
    """Deterministic contacts-v1 sequence prefix + per-seq-index position ids.

    Returns ``(prefix, seq_positions, L)`` where ``prefix`` is the document
    text up to and including ``<begin_statements>``, ``seq_positions[k]`` is
    the position index assigned to residue ``k`` (0-based, sequence order),
    and ``L`` is the residue count. ``None`` when the chain can't be
    serialized (fewer than 2 residues, or more than the position-token
    space — see :func:`build_document`). ``entry_id`` is the deterministic
    RNG seed: changing it resamples the start index + statement order
    (test-time augmentation).
    """
    result = build_document(
        entry_id, structure.residues, [], config=GenerationConfig()
    )
    if result is None:
        return None
    doc = result.document
    prefix = doc[: doc.index(BEGIN_STRUCTURE_TOKEN) + len(BEGIN_STRUCTURE_TOKEN)]
    nterm = result.n_term_index
    seq_positions = [(nterm + k) % NUM_POSITION_INDICES for k in range(result.seq_len)]
    return prefix, seq_positions, result.seq_len


def _token_id(tokenizer, token: str) -> int:
    """Resolve one domain token to its id; fail loudly on an UNK collapse.

    A wrong / missing contacts-v1 tokenizer maps every ``<pX>`` / ``<contact>``
    to the UNK id, which would otherwise silently produce a garbage contact
    map. Catch it here, the way the contacts-and-distances-v1 distance-bin
    resolver does.
    """
    tid = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk_id is not None and tid == unk_id):
        raise ValueError(
            f"Tokenizer has no dedicated id for {token!r} (got {tid}). The "
            f"tokenizer is missing the contacts-v1 vocabulary — make sure it "
            f"is co-located with the model."
        )
    return int(tid)


def _pcontact_matrix(
    backend: Backend, prefix: str, seq_positions: list[int]
) -> np.ndarray:
    """``P(contact)`` over all residue pairs for one sequence realization.

    Returns a symmetric ``[L, L]`` array (sequence coordinates) where entry
    (i, j) is ``exp(lp1[i] + lp2[i, j]) + exp(lp1[j] + lp2[j, i])`` — the
    model's probability of emitting {i, j} as its next contact statement,
    unordered. The diagonal / near-diagonal band is meaningless here; callers
    mask it.
    """
    tokenizer = backend.tokenizer
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    contact_id = _token_id(tokenizer, CONTACT_TOKEN)
    pos_ids = [_token_id(tokenizer, position_token(p)) for p in seq_positions]

    # lp1[i] = log P(<p_i> | prefix, <contact>): one forward, the distribution
    # right after `prefix + <contact>`, restricted to this protein's position
    # tokens.
    p1 = backend.next_token_probs(prefix_ids, [[contact_id]], pos_ids)  # (1, L)
    lp1 = np.log(np.clip(np.asarray(p1[0], dtype=np.float64), _PROB_FLOOR, None))

    # lp2[i, j] = log P(<p_j> | prefix, <contact>, <p_i>): one tail per i.
    tails = [[contact_id, pid] for pid in pos_ids]
    p2 = backend.next_token_probs(prefix_ids, tails, pos_ids)  # (L, L)
    lp2 = np.log(np.clip(np.asarray(p2, dtype=np.float64), _PROB_FLOOR, None))

    fwd = lp1[:, None] + lp2  # log P(i)·P(j|i)
    return np.exp(fwd) + np.exp(fwd.T)  # unordered contact probability, symmetric


def _ensemble_pcontact(
    backend: Backend, structure: ContactStructure, *, ensemble_k: int
) -> tuple[np.ndarray, int] | None:
    """Mean ``P(contact)`` over ``ensemble_k`` sequence-definition resamples.

    Each draw uses a different deterministic ``entry_id`` salt, so the
    contacts-v1 start index + statement order differ; the resulting matrices
    are all in sequence coordinates, so averaging is element-wise. Returns
    ``(P_mean, L)`` or ``None`` if the structure can't be serialized.
    """
    acc: np.ndarray | None = None
    seq_len = 0
    for k in range(ensemble_k):
        entry_id = (
            structure.entry_id
            if ensemble_k == 1
            else f"{structure.entry_id}#cv1ens{k}"
        )
        built = _prefix_and_positions(structure, entry_id=entry_id)
        if built is None:
            return None
        prefix, seq_positions, seq_len = built
        matrix = _pcontact_matrix(backend, prefix, seq_positions)
        acc = matrix if acc is None else acc + matrix
    assert acc is not None  # ensemble_k >= 1
    return acc / ensemble_k, seq_len


def _candidate_pairs(seq_len: int, min_seq_separation: int) -> list[tuple[int, int]]:
    """0-based (i, j), i < j, with sequence separation >= ``min_seq_separation``."""
    return [
        (i, j)
        for i in range(seq_len)
        for j in range(i + min_seq_separation, seq_len)
    ]


def _make_backend(cfg: InferenceConfig) -> Backend:
    """Construct the requested backend, passing through backend-specific kwargs."""
    if cfg.backend == "vllm":
        return load_backend(
            "vllm",
            model=cfg.model,
            dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            top_k_logprobs=cfg.top_k_logprobs,
            tail_batch_size=cfg.batch_size,
        )
    if cfg.backend == "transformers":
        return load_backend(
            "transformers",
            model=cfg.model,
            dtype=cfg.dtype,
            tail_batch_size=cfg.batch_size,
        )
    if cfg.backend == "mlx":
        return load_backend("mlx", model=cfg.model, tail_batch_size=cfg.batch_size)
    raise ValueError(
        f"Unknown backend {cfg.backend!r}. Expected one of: 'vllm', "
        f"'transformers', 'mlx'."
    )


# --------------------------------------------------------------------------
# Structure resolution (predict vs evaluate)
# --------------------------------------------------------------------------


def _structures_for_predict(
    cfg: InferenceConfig,
    structures: Iterable[ContactStructure] | None,
) -> list[ContactStructure]:
    """Structures for :func:`predict` — sequences only, no ground truth.

    Caller-supplied ``structures`` (e.g. from :func:`structure_from_sequence`)
    win. Otherwise structures are read from ``cfg.input_path`` with the
    gemmi-only contacts-and-distances-v1 parser, so plain inference on a PDB /
    mmCIF does **not** require pyconfind (only the sequence is needed).
    """
    if structures is not None:
        return list(structures)
    if cfg.input_path is None:
        raise ValueError("predict requires cfg.input_path or structures=")
    from marinfold.document_structures.contacts_and_distances_v1.parse import (
        iter_parsed_structures,
    )

    out: list[ContactStructure] = []
    for parsed in iter_parsed_structures(Path(cfg.input_path)):
        residues = tuple(
            ResidueInfo(seq_index=i, resname=r.name, resnum=r.index, chain="A")
            for i, r in enumerate(parsed.residues)
        )
        out.append(
            ContactStructure(
                entry_id=parsed.entry_id,
                residues=residues,
                global_plddt=parsed.global_plddt,
            )
        )
    return out


def _structures_for_evaluate(
    cfg: InferenceConfig,
    structures: Iterable[ContactStructure] | None,
) -> list[ContactStructure]:
    """Structures for :func:`evaluate` — with pyconfind ground-truth contacts.

    Caller-supplied ``structures`` (which must carry ``gt_contacts``) win;
    otherwise structures are analyzed from ``cfg.input_path`` with pyconfind
    (the ``contacts-v1`` extra), exactly as the training documents were built.
    """
    if structures is not None:
        return list(structures)
    if cfg.input_path is None:
        raise ValueError("evaluate requires cfg.input_path or structures=")
    return [
        ContactStructure(
            entry_id=analyzed.entry_id,
            residues=analyzed.residues,
            gt_contacts=analyzed.contacts,
            global_plddt=analyzed.global_plddt,
        )
        for analyzed in iter_analyzed_structures(Path(cfg.input_path))
    ]


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------


def predict(
    cfg: InferenceConfig,
    *,
    structures: Iterable[ContactStructure] | None = None,
) -> Iterator[dict]:
    """Yield one ``P(contact)`` record per input structure.

    Each record carries the entry id, residue count, the candidate pairs
    (1-based ``[i, j]``, ``i < j``, separation >= ``cfg.min_seq_separation``)
    and the model's ``P(contact)`` score per pair. With ``cfg.keep_matrix``
    the dense, band-masked ``[L, L]`` matrix is included too.

    No ground truth is consulted. ``structures`` may be passed directly (e.g.
    from :func:`structure_from_sequence`); otherwise they are read from
    ``cfg.input_path``.
    """
    resolved = _structures_for_predict(cfg, structures)
    if not resolved:
        return
    backend = _make_backend(cfg)
    for structure in resolved:
        built = _ensemble_pcontact(backend, structure, ensemble_k=cfg.ensemble_k)
        if built is None:
            warnings.warn(
                f"skipping {structure.entry_id}: cannot serialize "
                f"({len(structure.residues)} residues outside "
                f"[2, {NUM_POSITION_INDICES}])",
                stacklevel=2,
            )
            continue
        pcontact, seq_len = built
        pairs = _candidate_pairs(seq_len, cfg.min_seq_separation)
        record: dict[str, Any] = {
            "entry_id": structure.entry_id,
            "n_residues": seq_len,
            "min_seq_separation": cfg.min_seq_separation,
            "ensemble_k": cfg.ensemble_k,
            "pairs": [[i + 1, j + 1] for (i, j) in pairs],
            "p_contact": [float(pcontact[i, j]) for (i, j) in pairs],
        }
        if cfg.keep_matrix:
            record["p_contact_matrix"] = _band_masked(
                pcontact, seq_len, cfg.min_seq_separation
            ).tolist()
        yield record


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def _gt_contact_matrix(
    gt_contacts: Iterable[RawContact], seq_len: int, min_seq_separation: int
) -> np.ndarray:
    """Boolean ``[L, L]`` GT matrix: degree >= floor and separation in range."""
    matrix = np.zeros((seq_len, seq_len), dtype=bool)
    for contact in gt_contacts:
        i, j = contact.seq_i, contact.seq_j
        if (
            contact.degree >= _MIN_CONTACT_DEGREE
            and (j - i) >= min_seq_separation
            and 0 <= i < j < seq_len
        ):
            matrix[i, j] = matrix[j, i] = True
    return matrix


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann–Whitney rank statistic (ties get average ranks).

    Avoids a scikit-learn dependency. Returns NaN when one class is absent.
    """
    n = labels.size
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average ranks within tied score groups.
    sorted_scores = scores[order]
    start = 0
    for end in range(1, n + 1):
        if end == n or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _metric_rows(
    pcontact: np.ndarray, gt: np.ndarray, seq_len: int, min_seq_separation: int
) -> dict[str, dict[str, float]]:
    """Per-range precision @ {L, L/2, L/5, R} + AUC for one structure.

    Mirrors exp89's ``compute_metrics.metric_rows``: candidate pairs are
    every i < j with separation in the range; ``R`` is the number of true
    contacts in the range; precision is over the top-K ranked candidates.
    """
    rows = np.array([i for (i, j) in _candidate_pairs(seq_len, min_seq_separation)])
    cols = np.array([j for (i, j) in _candidate_pairs(seq_len, min_seq_separation)])
    if rows.size == 0:
        return {rng: {} for rng in _RANGES}
    sep = cols - rows
    scores_all = pcontact[rows, cols]
    labels_all = gt[rows, cols]

    out: dict[str, dict[str, float]] = {}
    for rng, (lo, hi) in _RANGES.items():
        mask = sep >= lo
        if hi is not None:
            mask = mask & (sep <= hi)
        scores = scores_all[mask]
        labels = labels_all[mask]
        n_candidate = scores.size
        n_true = int(labels.sum())
        metrics: dict[str, float] = {"auc": _rank_auc(scores, labels)}
        order = np.argsort(-scores, kind="mergesort") if n_candidate else None
        ranked = labels[order] if n_candidate else None
        cuts = (
            ("precision_at_L", seq_len),
            ("precision_at_L2", max(1, seq_len // 2)),
            ("precision_at_L5", max(1, seq_len // 5)),
            ("r_precision", n_true),
        )
        for key, target in cuts:
            if n_candidate == 0 or target <= 0:
                metrics[key] = float("nan")
            else:
                top = min(int(target), n_candidate)
                metrics[key] = float(ranked[:top].sum()) / top
        out[rng] = metrics
    return out


def evaluate(
    cfg: InferenceConfig,
    *,
    structures: Iterable[ContactStructure] | None = None,
) -> EvalResult:
    """Score the model's contacts against pyconfind ground truth.

    Headline metrics are macro-means across structures, keyed
    ``<metric>_<range>`` (e.g. ``auc_long``, ``r_precision_long``,
    ``precision_at_L_long``). ``per_example`` holds every scored candidate
    pair (entry id, 1-based i / j, ``p_contact``, ``gt``) so
    :func:`plots.plot_evaluate_pdf` can redraw GT-vs-model contact maps.

    Ground truth is the input itself — pyconfind side-chain contacts on the
    structure (degree >= 0.001, separation >= ``cfg.min_seq_separation``),
    the same definition as the contacts-v1 training documents. ``structures``
    may be passed directly (carrying ``gt_contacts``); otherwise they are
    analyzed from ``cfg.input_path`` (needs the ``contacts-v1`` extra).
    """
    resolved = _structures_for_evaluate(cfg, structures)
    if not resolved:
        return EvalResult(
            metrics={},
            per_example=[],
            extras={"structure": NAME, "warning": "no input structures", "model": cfg.model},
        )
    backend = _make_backend(cfg)

    agg: dict[str, list[float]] = defaultdict(list)
    per_example: list[dict] = []
    per_structure_n_residues: dict[str, int] = {}
    n_scored = 0

    for structure in resolved:
        if structure.gt_contacts is None:
            warnings.warn(
                f"skipping {structure.entry_id}: no ground-truth contacts",
                stacklevel=2,
            )
            continue
        built = _ensemble_pcontact(backend, structure, ensemble_k=cfg.ensemble_k)
        if built is None:
            warnings.warn(
                f"skipping {structure.entry_id}: cannot serialize "
                f"({len(structure.residues)} residues)",
                stacklevel=2,
            )
            continue
        pcontact, seq_len = built
        gt = _gt_contact_matrix(structure.gt_contacts, seq_len, cfg.min_seq_separation)
        per_structure_n_residues[structure.entry_id] = seq_len

        rows = _metric_rows(pcontact, gt, seq_len, cfg.min_seq_separation)
        for rng, metrics in rows.items():
            for key, value in metrics.items():
                if math.isfinite(value):
                    agg[f"{key}_{rng}"].append(value)

        for (i, j) in _candidate_pairs(seq_len, cfg.min_seq_separation):
            per_example.append({
                "entry_id": structure.entry_id,
                "i": i + 1,
                "j": j + 1,
                "p_contact": float(pcontact[i, j]),
                "gt": int(bool(gt[i, j])),
            })
        n_scored += 1

    metrics = {key: float(np.mean(values)) for key, values in agg.items() if values}
    return EvalResult(
        metrics=metrics,
        per_example=per_example,
        extras={
            "structure": NAME,
            "model": cfg.model,
            "backend": cfg.backend,
            "min_seq_separation": cfg.min_seq_separation,
            "ensemble_k": cfg.ensemble_k,
            "n_structures": n_scored,
            "per_structure_n_residues": per_structure_n_residues,
        },
    )


# --------------------------------------------------------------------------
# Shared helper (also used by plots.py)
# --------------------------------------------------------------------------


def _band_masked(
    pcontact: np.ndarray, seq_len: int, min_seq_separation: int
) -> np.ndarray:
    """Copy of ``pcontact`` with the diagonal + near-diagonal band set to NaN.

    Pairs with |i - j| < ``min_seq_separation`` are never contacts in
    contacts-v1, so they're blanked for display rather than shown as a
    sequence-locality gradient.
    """
    out = np.array(pcontact, dtype=np.float64, copy=True)
    idx = np.arange(seq_len)
    band = np.abs(np.subtract.outer(idx, idx)) < min_seq_separation
    out[band] = np.nan
    return out
