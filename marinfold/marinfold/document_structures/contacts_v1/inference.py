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
import re
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
    CONTEXT_LENGTH,
    END_TOKEN,
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

# Parses `<contact> <p_i> <p_j>` triples out of a sampled rollout completion
# (one contacts-v1 contact statement). Mirrors exp82's CONTACT_RE.
_CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

# Rollout generation budget: 4·L + 64 tokens. A contacts-v1 protein emits well
# under L contacts (sep >= 6), each a 3-token statement, so this is generous
# headroom while bounding a runaway sample. Sized from L (which the model sees
# in the prefix), never the GT contact count — no oracle dependence (exp82).
_ROLLOUT_TOKENS_PER_RESIDUE = 4
_ROLLOUT_TOKENS_CONSTANT = 64


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

    ``method`` selects the contact readout:

    - ``"pairwise"`` (default, fast ~0.3 s/protein) — the P(contact) readout
      above, optionally averaged over ``ensemble_k`` resampled realizations.
    - ``"rollout"`` (~50 s/protein) — exp82's settled best LM-only recipe:
      ``n_rollouts`` sampled contact-section completions, each from a fresh
      document realization, voted by pair-occurrence frequency and tie-broken
      by the pairwise log-prob. Needs a sampling backend (vLLM / transformers;
      not MLX). ``temperature`` / ``top_p`` / ``top_k`` are its sampling knobs.

    ``min_seq_separation`` is the smallest primary-sequence gap |i - j| that
    can be a contact (6 — matching the data), so closer pairs are never
    scored or counted. ``top_k_logprobs`` and ``gpu_memory_utilization`` are
    vLLM-only; other backends ignore them. ``batch_size`` is the per-prefix
    fan-out backends batch internally (pairwise tails / rollout completions).
    ``keep_matrix`` adds the dense per-structure score matrix to each
    ``predict`` record.
    """

    model: str | None
    input_path: Path | None = None
    backend: str = "vllm"
    batch_size: int = 64
    dtype: str = "bfloat16"
    method: str = "pairwise"
    min_seq_separation: int = 6
    ensemble_k: int = 1
    top_k_logprobs: int = 256
    gpu_memory_utilization: float = 0.85
    keep_matrix: bool = False
    # rollout-only sampling knobs (method == "rollout"):
    n_rollouts: int = 100
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50


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


def _fwd_matrix(
    backend: Backend, prefix: str, seq_positions: list[int]
) -> np.ndarray:
    """``log P(i)·P(j|i)`` over all residue pairs for one sequence realization.

    Returns ``fwd[i, j] = lp1[i] + lp2[i, j]`` with
    ``lp1[i] = log P(<p_i> | prefix, <contact>)`` and
    ``lp2[i, j] = log P(<p_j> | prefix, <contact>, <p_i>)`` — the two
    conditional distributions the pairwise readout needs, restricted to this
    protein's position tokens. Not symmetric; callers symmetrize.
    """
    tokenizer = backend.tokenizer
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    contact_id = _token_id(tokenizer, CONTACT_TOKEN)
    pos_ids = [_token_id(tokenizer, position_token(p)) for p in seq_positions]

    # lp1: one forward, the distribution right after `prefix + <contact>`.
    p1 = backend.next_token_probs(prefix_ids, [[contact_id]], pos_ids)  # (1, L)
    lp1 = np.log(np.clip(np.asarray(p1[0], dtype=np.float64), _PROB_FLOOR, None))
    # lp2: one tail (`<contact> <p_i>`) per i.
    tails = [[contact_id, pid] for pid in pos_ids]
    p2 = backend.next_token_probs(prefix_ids, tails, pos_ids)  # (L, L)
    lp2 = np.log(np.clip(np.asarray(p2, dtype=np.float64), _PROB_FLOOR, None))
    return lp1[:, None] + lp2  # log P(i)·P(j|i)


def _pcontact_from_fwd(fwd: np.ndarray) -> np.ndarray:
    """Unordered ``P(contact)``: ``exp(fwd) + exp(fwd.T)`` (symmetric)."""
    return np.exp(fwd) + np.exp(fwd.T)


def _sym_from_fwd(fwd: np.ndarray) -> np.ndarray:
    """Symmetrized geo-mean log-score ``0.5·(fwd + fwd.T)``.

    The pairwise *ranking* score, and the key that breaks rollout's vote ties.
    """
    return 0.5 * (fwd + fwd.T)


def _pcontact_matrix(
    backend: Backend, prefix: str, seq_positions: list[int]
) -> np.ndarray:
    """``P(contact)`` over all residue pairs for one sequence realization.

    Entry (i, j) is the model's probability of emitting {i, j} as its next
    contact statement, unordered. The diagonal / near-diagonal band is
    meaningless here; callers mask it.
    """
    return _pcontact_from_fwd(_fwd_matrix(backend, prefix, seq_positions))


# --------------------------------------------------------------------------
# Per-structure score matrices (pairwise / rollout)
# --------------------------------------------------------------------------


def _pairwise_score_matrix(
    backend: Backend, structure: ContactStructure, cfg: "InferenceConfig"
) -> tuple[np.ndarray, int] | None:
    """Mean ``P(contact)`` over ``cfg.ensemble_k`` sequence-definition resamples.

    Each draw uses a different deterministic ``entry_id`` salt, so the
    contacts-v1 start index + statement order differ; the resulting matrices
    are all in sequence coordinates, so averaging is element-wise. Returns
    ``(score, L)`` or ``None`` if the structure can't be serialized.
    """
    acc: np.ndarray | None = None
    seq_len = 0
    for k in range(cfg.ensemble_k):
        entry_id = (
            structure.entry_id
            if cfg.ensemble_k == 1
            else f"{structure.entry_id}#cv1ens{k}"
        )
        built = _prefix_and_positions(structure, entry_id=entry_id)
        if built is None:
            return None
        prefix, seq_positions, seq_len = built
        matrix = _pcontact_from_fwd(_fwd_matrix(backend, prefix, seq_positions))
        acc = matrix if acc is None else acc + matrix
    assert acc is not None  # ensemble_k >= 1
    return acc / cfg.ensemble_k, seq_len


def _adaptive_sample_batch(seq_len: int, cap: int, *, budget: int = 20000) -> int:
    """Bigger generation batch for short proteins, smaller for long.

    Keeps the per-batch KV-cache roughly constant (~``budget`` tokens). vLLM
    ignores this (it schedules its own batching); transformers honours it.
    """
    return max(1, min(cap, budget // max(seq_len, 1)))


def _tiebreak(votes: np.ndarray, pairwise_sym: np.ndarray) -> np.ndarray:
    """``votes + (pairwise − lo)/(hi − lo)·0.5`` — votes rank, pairwise breaks ties.

    Vote counts are integers (gaps >= 1), so adding a pairwise term bounded to
    [0, 0.5) only reorders pairs *tied* on votes; it never crosses a count
    boundary. min-max over the upper triangle is monotonic, so within a tie
    group this is exactly ranking by the pairwise score (exp82's tie-break).
    """
    upper = np.triu_indices(votes.shape[0], k=1)
    scores = pairwise_sym[upper]
    lo, hi = float(scores.min()), float(scores.max())
    return votes + (pairwise_sym - lo) / (hi - lo + 1e-9) * 0.5


def _rollout_score_matrix(
    backend: Backend, structure: ContactStructure, cfg: "InferenceConfig"
) -> tuple[np.ndarray, int] | None:
    """exp82 ``rollout + resample + tiebreak`` score matrix.

    Draw ``cfg.n_rollouts`` sampled contact-section completions — each from a
    *fresh* document realization (resampled N-terminus + statement order) — and
    accumulate the per-pair occurrence frequency into a symmetric ``[L, L]``
    vote matrix (input-sequence coordinates). The big 0-vote tie mass is then
    broken by the pairwise log-prob from one canonical realization. Returns
    ``(combined, L)`` or ``None`` if the structure can't be serialized.

    Needs a sampling backend (vLLM / transformers); MLX raises
    ``NotImplementedError`` from :meth:`Backend.sample_completions`.
    """
    tokenizer = backend.tokenizer
    stop_id = _token_id(tokenizer, END_TOKEN)

    prefixes: list[list[int]] = []
    position_maps: list[dict[int, int]] = []
    seq_len = 0
    for r in range(cfg.n_rollouts):
        built = _prefix_and_positions(structure, entry_id=f"{structure.entry_id}:r{r}")
        if built is None:
            return None
        prefix, seq_positions, seq_len = built
        prefixes.append(list(tokenizer.encode(prefix, add_special_tokens=False)))
        position_maps.append({pos: i for i, pos in enumerate(seq_positions)})

    max_new = min(
        CONTEXT_LENGTH - len(prefixes[0]),
        _ROLLOUT_TOKENS_PER_RESIDUE * seq_len + _ROLLOUT_TOKENS_CONSTANT,
    )
    completions = backend.sample_completions(
        prefixes,
        max_new_tokens=max_new,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        stop_token_id=stop_id,
        batch_size=_adaptive_sample_batch(seq_len, cfg.batch_size),
    )

    votes = np.zeros((seq_len, seq_len), dtype=np.float64)
    for token_ids, pos_to_seq in zip(completions, position_maps, strict=True):
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        seen: set[tuple[int, int]] = set()
        for a, b in _CONTACT_RE.findall(text):
            ia, ib = pos_to_seq.get(int(a)), pos_to_seq.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            lo, hi = (ia, ib) if ia < ib else (ib, ia)
            if (hi - lo) >= cfg.min_seq_separation and (lo, hi) not in seen:
                seen.add((lo, hi))
                votes[lo, hi] += 1.0
                votes[hi, lo] += 1.0

    # Pairwise log-prob (one canonical realization) breaks the vote ties.
    built = _prefix_and_positions(structure, entry_id=structure.entry_id)
    assert built is not None  # the rollout realizations already serialized
    prefix, seq_positions, _ = built
    pairwise_sym = _sym_from_fwd(_fwd_matrix(backend, prefix, seq_positions))
    return _tiebreak(votes, pairwise_sym), seq_len


def _score_matrix(
    backend: Backend, structure: ContactStructure, cfg: "InferenceConfig"
) -> tuple[np.ndarray, int] | None:
    """Dispatch to the configured readout; returns ``(score, L)`` or ``None``.

    Higher score ⇒ more likely contact, for both methods (so ranking, plots,
    and metrics are method-agnostic).
    """
    if cfg.method == "rollout":
        return _rollout_score_matrix(backend, structure, cfg)
    if cfg.method == "pairwise":
        return _pairwise_score_matrix(backend, structure, cfg)
    raise ValueError(
        f"Unknown method {cfg.method!r}. Expected 'pairwise' or 'rollout'."
    )


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
    """Yield one contact-score record per input structure.

    Each record carries the entry id, residue count, the ``method``, the
    candidate pairs (1-based ``[i, j]``, ``i < j``, separation >=
    ``cfg.min_seq_separation``) and the per-pair ranking ``score`` (higher ⇒
    more likely contact: ``P(contact)`` for ``method="pairwise"``, the
    tie-broken vote score for ``"rollout"``). With ``cfg.keep_matrix`` the
    dense, band-masked ``[L, L]`` score matrix is included too.

    No ground truth is consulted. ``structures`` may be passed directly (e.g.
    from :func:`structure_from_sequence`); otherwise they are read from
    ``cfg.input_path``.
    """
    resolved = _structures_for_predict(cfg, structures)
    if not resolved:
        return
    backend = _make_backend(cfg)
    for structure in resolved:
        built = _score_matrix(backend, structure, cfg)
        if built is None:
            warnings.warn(
                f"skipping {structure.entry_id}: cannot serialize "
                f"({len(structure.residues)} residues outside "
                f"[2, {NUM_POSITION_INDICES}])",
                stacklevel=2,
            )
            continue
        score, seq_len = built
        pairs = _candidate_pairs(seq_len, cfg.min_seq_separation)
        record: dict[str, Any] = {
            "entry_id": structure.entry_id,
            "n_residues": seq_len,
            "min_seq_separation": cfg.min_seq_separation,
            "method": cfg.method,
            "pairs": [[i + 1, j + 1] for (i, j) in pairs],
            "score": [float(score[i, j]) for (i, j) in pairs],
        }
        record["n_rollouts" if cfg.method == "rollout" else "ensemble_k"] = (
            cfg.n_rollouts if cfg.method == "rollout" else cfg.ensemble_k
        )
        if cfg.keep_matrix:
            record["score_matrix"] = _band_masked(
                score, seq_len, cfg.min_seq_separation
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
        built = _score_matrix(backend, structure, cfg)
        if built is None:
            warnings.warn(
                f"skipping {structure.entry_id}: cannot serialize "
                f"({len(structure.residues)} residues)",
                stacklevel=2,
            )
            continue
        score, seq_len = built
        gt = _gt_contact_matrix(structure.gt_contacts, seq_len, cfg.min_seq_separation)
        per_structure_n_residues[structure.entry_id] = seq_len

        rows = _metric_rows(score, gt, seq_len, cfg.min_seq_separation)
        for rng, rng_metrics in rows.items():
            for key, value in rng_metrics.items():
                if math.isfinite(value):
                    agg[f"{key}_{rng}"].append(value)

        for (i, j) in _candidate_pairs(seq_len, cfg.min_seq_separation):
            per_example.append({
                "entry_id": structure.entry_id,
                "i": i + 1,
                "j": j + 1,
                "score": float(score[i, j]),
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
            "method": cfg.method,
            "min_seq_separation": cfg.min_seq_separation,
            "ensemble_k": cfg.ensemble_k,
            "n_rollouts": cfg.n_rollouts,
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
