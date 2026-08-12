# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense per-contact reward for exp200 RL post-training — issue #200.

Operates on **token ids**, not text. exp163's ``rollout_metrics.parse_sections``
regex-scans a decoded string, which is fine for scoring but cannot tell you
*which token position* a contact was emitted at — and a dense policy-gradient
signal has to land on specific positions.

TOKEN ALIGNMENT. marin.rl computes ``logprobs[t] = log pi(token_t | tokens_<t>)``
(``rl_losses.compute_logprobs``: ``-loss[:, :-1]`` with a zero column prepended)
and indexes ``loss_weights`` the same way (``train_batch.py``: prompt zeros then
one entry per response token). So the weight for a token lands **on that token,
with no shift**. This is the OPPOSITE of exp163's ``loss_mask.py`` convention,
where ``weight[i]`` supervises predicting ``tokens[i+1]``. Do not port that
convention here.

THE REWARD. For contact *t* in section *k*, with ``x = 1`` iff its (i, j) pair is
in ground truth (seq-index, separation >= 6, not already emitted in this
section)::

    r = (1 - p_bar)            if x == 1
        -p_bar * decay**e      if x == 0     (e = # earlier errors in section k)

spread evenly over the triple's three tokens. ``p_bar`` is the policy's own
recent per-contact precision, so ``E[r] ~ 0`` at current performance and the
gradient says only "beat yourself". This matters more than any other constant
here: precision is ~0.3, so a *fixed* penalty would make "emit nothing" the
optimal policy and the run would collapse to empty sections. ``decay`` < 1
implements the observation that once a section has gone wrong, later contacts in
it may be logical consequences of the first mistake rather than independent
errors (``decay=1`` recovers flat penalties, ``decay=0`` penalizes only the
first error in each section).

The document-level best-of-N term is deliberately NOT applied here: it needs a
baseline over the whole group of rollouts for a protein, which only the loss
module sees. This function returns the stepwise term plus ``episode_reward``
(the document-level *raw* return), and ``dense_loss.ContactsDenseLoss`` combines
them.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# Ids are fully determined by ``marinfold.document_structures.core.build_tokenizer``
# (``<pad>``=0, ``<eos>``=1, then ``all_domain_tokens()`` at 2+). Baked in as a
# convenience; ``resolve_contact_vocab`` re-derives them from a live tokenizer and
# refuses to run on drift, the same guard ``loss_mask.resolve_span_ids`` uses.
CONTACT_ID = 5           # <contact>
BEGIN_STATEMENTS_ID = 9  # <begin_statements>
END_ID = 10              # <end>
P0_ID = 143              # <p0>; <p0>..<p1999> are contiguous
NUM_POSITIONS = 2000

PLAIN_DOC_ID = 2   # <contacts-v1>
# NB: exp200 also carried a multi-draft sentinel (id 7) for exp163's RENAMED
# tokenizer. exp208 is plain-mode only, and exp199 ships the STANDARD
# contacts-v1 tokenizer where id 7 is <contacts-and-distances-v1> -- a
# different document structure this model was never trained to emit. Keeping
# the constant would invite a 'just set mode=multi' that silently builds an
# out-of-distribution prompt. Verified against exp199's tokenizer 2026-08-10:
# ids 5 / 9 / 10 / 143 and the <p0>..<p1999> contiguity all hold, so
# resolve_contact_vocab passes unchanged.

MIN_SEP = 6

Pair = tuple[int, int]


@dataclass(frozen=True)
class ContactVocab:
    """The handful of token ids the reward walk needs."""

    contact_id: int = CONTACT_ID
    begin_id: int = BEGIN_STATEMENTS_ID
    end_id: int = END_ID
    p0_id: int = P0_ID
    n_positions: int = NUM_POSITIONS

    def position_of(self, token_id: int) -> int | None:
        """Residue-position index for a ``<pN>`` token id, else None."""
        offset = token_id - self.p0_id
        return offset if 0 <= offset < self.n_positions else None


DEFAULT_VOCAB = ContactVocab()


def resolve_contact_vocab(tokenizer) -> ContactVocab:
    """Resolve the contact-walk ids from a live contacts-v1 tokenizer.

    Raises if a token is unknown or if any id drifts from the baked-in constant.
    A silent drift would reward the wrong token positions, which is exactly the
    class of bug that produces a plausible-looking run with no signal.

    Args:
        tokenizer: A HuggingFace tokenizer built over the contacts-v1 vocab.

    Returns:
        The resolved :class:`ContactVocab`.
    """
    names = {
        "contact_id": "<contact>",
        "begin_id": "<begin_statements>",
        "end_id": "<end>",
        "p0_id": "<p0>",
    }
    unk = getattr(tokenizer, "unk_token_id", None)
    resolved: dict[str, int] = {}
    for field_name, token in names.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or (unk is not None and tid == unk):
            raise ValueError(f"tokenizer does not know {token!r} — is this the contacts-v1 tokenizer?")
        resolved[field_name] = int(tid)

    baked = {"contact_id": CONTACT_ID, "begin_id": BEGIN_STATEMENTS_ID, "end_id": END_ID, "p0_id": P0_ID}
    if resolved != baked:
        raise ValueError(
            f"contacts-v1 vocab drift: resolved {resolved} != baked-in {baked}. "
            "Update contact_rewards.py's constants (and re-check every place that "
            "documents them) before training on this tokenizer."
        )

    last = tokenizer.convert_tokens_to_ids(f"<p{NUM_POSITIONS - 1}>")
    if last != resolved["p0_id"] + NUM_POSITIONS - 1:
        raise ValueError(
            f"position tokens are not contiguous: <p0>={resolved['p0_id']} but "
            f"<p{NUM_POSITIONS - 1}>={last}"
        )
    return ContactVocab(**resolved)


@dataclass
class EmittedContact:
    """One ``<contact> <pI> <pJ>`` triple found in the response."""

    start: int            # response index of the <contact> token
    section: int          # 0-based section index
    pair: Pair | None     # (min, max) seq-index pair, or None if unusable
    correct: bool
    reason: str           # "ok" | "duplicate" | "malformed" | "unmapped" | "too_close" | "truncated"


@dataclass
class DenseReward:
    """Stepwise token rewards plus the document-level return for one rollout."""

    token_rewards: np.ndarray       # float32, len == len(response_ids)
    episode_reward: float           # R_doc: best-section F1 (multi) or section-1 F1 (plain)
    section_f1: list[float]
    scored_sections: int
    diagnostics: dict[str, float] = field(default_factory=dict)


def in_band(pair: Pair, lo: int = MIN_SEP, hi: int | None = None) -> bool:
    """Whether a seq-index pair falls in an inclusive separation band."""
    sep = abs(pair[0] - pair[1])
    return sep >= lo and (hi is None or sep <= hi)


def f1_all_band(pred: set[Pair], gt: set[Pair]) -> float:
    """``all``-band F1, matching exp163 ``rollout_metrics.score_rollout``.

    Both sets are assumed already filtered to separation >= :data:`MIN_SEP`.
    Returns NaN when ground truth is empty (recall undefined), 0.0 when the
    prediction is empty but ground truth is not — the same conventions the
    published exp163 numbers were computed under, so the two are comparable.
    """
    if not gt:
        return math.nan
    tp = len(pred & gt)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt)
    if prec + rec <= 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def walk_contacts(
    response_ids: Sequence[int],
    pos_to_seq: Mapping[int, int],
    gt: set[Pair],
    *,
    starts_in_section: bool = True,
    vocab: ContactVocab = DEFAULT_VOCAB,
) -> list[EmittedContact]:
    """Decode every contact triple in a response, in emission order.

    Args:
        response_ids: Generated token ids (the completion only, no prompt).
        pos_to_seq: This realization's ``<pN>`` index -> sequence index map.
        gt: Ground-truth contact set as (min, max) sequence-index pairs.
        starts_in_section: True when the prompt ended on ``<begin_statements>``
            (exp163's prompt does), so response index 0 is already inside
            section 0.
        vocab: Resolved token ids.

    Returns:
        One :class:`EmittedContact` per ``<contact>`` token, tagged with the
        section it belongs to and why it was or was not counted correct.
        Duplicates are scoped per section: a pair repeated inside one section is
        a duplicate, but the same pair in a later section is a fresh prediction
        (sections are independent candidate structures).
    """
    out: list[EmittedContact] = []
    seen_per_section: dict[int, set[Pair]] = {}
    section = 0 if starts_in_section else -1
    i = 0
    n = len(response_ids)
    while i < n:
        tid = response_ids[i]
        if tid == vocab.begin_id:
            section += 1
            i += 1
            continue
        if tid == vocab.end_id:
            break
        if tid != vocab.contact_id:
            i += 1
            continue

        # A contact triple. Anything that is not two position tokens is malformed.
        if i + 2 >= n:
            out.append(EmittedContact(i, max(section, 0), None, False, "truncated"))
            break
        pa = vocab.position_of(response_ids[i + 1])
        pb = vocab.position_of(response_ids[i + 2])
        if pa is None or pb is None:
            out.append(EmittedContact(i, max(section, 0), None, False, "malformed"))
            i += 1
            continue

        sec = max(section, 0)
        ia, ib = pos_to_seq.get(pa), pos_to_seq.get(pb)
        if ia is None or ib is None:
            out.append(EmittedContact(i, sec, None, False, "unmapped"))
        elif ia == ib or abs(ia - ib) < MIN_SEP:
            out.append(EmittedContact(i, sec, None, False, "too_close"))
        else:
            pair = (min(ia, ib), max(ia, ib))
            seen = seen_per_section.setdefault(sec, set())
            if pair in seen:
                out.append(EmittedContact(i, sec, pair, False, "duplicate"))
            else:
                seen.add(pair)
                out.append(EmittedContact(i, sec, pair, pair in gt, "ok"))
        i += 3
    return out


def dense_rewards(
    response_ids: Sequence[int],
    pos_to_seq: Mapping[int, int],
    gt: set[Pair],
    *,
    mode: str,
    precision_baseline: float,
    err_decay: float = 0.5,
    max_sections: int | None = None,
    starts_in_section: bool = True,
    vocab: ContactVocab = DEFAULT_VOCAB,
) -> DenseReward:
    """Dense per-token stepwise reward and the document-level return.

    Args:
        response_ids: Generated token ids (completion only).
        pos_to_seq: ``<pN>`` index -> sequence index for this realization.
        gt: Ground-truth contacts as (min, max) sequence-index pairs, sep >= 6.
        mode: ``"multi"`` scores every section up to ``max_sections`` and returns
            best-of-N as the document return; ``"plain"`` scores only the first
            section (the base contacts-v1 task) and leaves any further sections
            unrewarded rather than shaping termination — #200 reports the
            last-vs-best gap but does not optimize it.
        precision_baseline: ``p_bar``, the policy's recent per-contact precision.
            Centering on it is what keeps "emit nothing" from being optimal.
        err_decay: Geometric decay applied to the penalty for the 2nd, 3rd, ...
            error within a section.
        max_sections: Hard cap on scored sections, mirroring the sampler's
            ``--max-sections``. Sections past the cap get no stepwise reward.
        starts_in_section: See :func:`walk_contacts`.
        vocab: Resolved token ids.

    Returns:
        A :class:`DenseReward`. ``token_rewards`` carries the RAW stepwise term —
        ``dense_loss.ContactsDenseLoss`` owns both the ``lam_step`` and ``lam_doc``
        scales, so they live in one place;
        the document-level best-of-N advantage is added by the loss module, which
        is the only place the per-protein group baseline is available.
    """
    if not 0.0 <= precision_baseline <= 1.0:
        raise ValueError(f"precision_baseline must be a probability, got {precision_baseline}")
    if not 0.0 <= err_decay <= 1.0:
        raise ValueError(f"err_decay must be in [0, 1], got {err_decay}")
    if mode not in ("plain", "multi"):
        raise ValueError(f"mode must be 'plain' or 'multi', got {mode!r}")

    contacts = walk_contacts(
        response_ids, pos_to_seq, gt, starts_in_section=starts_in_section, vocab=vocab
    )

    n_raw_sections = 1 + max((c.section for c in contacts), default=0)
    if mode == "plain":
        scored_sections = 1
    elif max_sections is None:
        scored_sections = n_raw_sections
    else:
        scored_sections = min(n_raw_sections, max_sections)

    rewards = np.zeros(len(response_ids), dtype=np.float32)
    errors_in_section: dict[int, int] = {}
    n_scored = n_correct = 0
    for c in contacts:
        if c.section >= scored_sections or c.reason == "truncated":
            continue
        n_scored += 1
        if c.correct:
            n_correct += 1
            r = 1.0 - precision_baseline
        else:
            e = errors_in_section.get(c.section, 0)
            errors_in_section[c.section] = e + 1
            r = -precision_baseline * (err_decay**e)
        # Spread over <contact> <pI> <pJ> so a contact's total credit does not
        # depend on how many tokens happened to encode it.
        span = rewards[c.start : c.start + 3]
        span += np.float32(r / 3.0)

    preds: list[set[Pair]] = [set() for _ in range(scored_sections)]
    for c in contacts:
        if c.pair is not None and c.reason == "ok" and c.section < scored_sections:
            preds[c.section].add(c.pair)
    section_f1 = [f1_all_band(p, gt) for p in preds]
    finite = [v for v in section_f1 if not math.isnan(v)]
    episode_reward = max(finite) if finite else 0.0

    jac = [
        len(a & b) / max(1, len(a | b))
        for a, b in zip(preds, preds[1:])
    ]
    diagnostics = {
        "n_sections": float(scored_sections),
        "n_sections_raw": float(n_raw_sections),
        "n_contacts_scored": float(n_scored),
        "n_contacts_correct": float(n_correct),
        "precision": (n_correct / n_scored) if n_scored else math.nan,
        "n_pred": float(sum(len(p) for p in preds)),
        "n_gt": float(len(gt)),
        "best_f1": episode_reward,
        "first_f1": finite[0] if finite else math.nan,
        "last_f1": finite[-1] if finite else math.nan,
        "mean_f1": float(np.mean(finite)) if finite else math.nan,
        "mean_jaccard": float(np.mean(jac)) if jac else math.nan,
        "frac_improving": (
            float(np.mean([b > a for a, b in zip(finite, finite[1:])])) if len(finite) > 1 else math.nan
        ),
        "n_duplicate": float(sum(c.reason == "duplicate" for c in contacts)),
        "n_malformed": float(sum(c.reason == "malformed" for c in contacts)),
        "n_unmapped": float(sum(c.reason == "unmapped" for c in contacts)),
        "n_too_close": float(sum(c.reason == "too_close" for c in contacts)),
        "n_truncated": float(sum(c.reason == "truncated" for c in contacts)),
    }
    return DenseReward(
        token_rewards=rewards,
        episode_reward=float(episode_reward),
        section_f1=section_f1,
        scored_sections=scored_sections,
        diagnostics=diagnostics,
    )


__all__ = [
    "BEGIN_STATEMENTS_ID",
    "CONTACT_ID",
    "DEFAULT_VOCAB",
    "END_ID",
    "MIN_SEP",
    "NUM_POSITIONS",
    "P0_ID",
    "PLAIN_DOC_ID",
    "ContactVocab",
    "DenseReward",
    "EmittedContact",
    "dense_rewards",
    "f1_all_band",
    "in_band",
    "resolve_contact_vocab",
    "walk_contacts",
]
