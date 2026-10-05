# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the exp254 seeded-rollout evaluation.

Everything here is deliberately duplicated from exp82's published worker
(``score_rollout_worker.py`` / ``score_rollout_vllm.py``) rather than imported:
this experiment's whole claim rests on its ``iid`` arm reproducing #245's
published m2-p06 eval-val number, so the realization construction, the contact
regex, the ``sep >= 6`` filter and the within-rollout dedup have to be the same
bytes, not a refactor of them.

The one thing that is *not* duplicated is the eval set: exp254 scores **only
eval-val** (97 natural FoldBench monomers, #245). eval-test is not read.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: exp82's document-frame constants. ``NUM_POS`` is the contacts-v1 position-token
#: ring, ``MIN_SEP`` the minimum primary-sequence separation a contact can have.
BEGIN, NUM_POS, MIN_SEP = "<begin_statements>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

#: #245's published eval inputs, vendored into this repo by that experiment.
EXP245_DATA = (
    Path(__file__).resolve().parents[1]
    / "exp245_evals_foldbench_held_out_monomers"
    / "data"
)
TARGETS_PARQUET = EXP245_DATA / "eval_targets_foldbench_monomers.parquet"
EVAL_SETS_CSV = EXP245_DATA / "eval_sets.csv"
GROUND_TRUTH_JSONL = EXP245_DATA / "gt_universe_scored.jsonl"

#: The only eval set this experiment is allowed to read, and its scored size.
EVAL_SET = "eval-val"
EXPECTED_UNITS = 97


@dataclass(frozen=True)
class Target:
    """One eval protein: its identity, length and input sequence."""

    dataset: str
    stem: str
    L: int
    input_seq: str

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.stem}"


def load_targets(eval_set: str = EVAL_SET) -> list[Target]:
    """The eval-set's targets, sorted short -> long.

    Joins #245's target parquet (which carries the input sequences) to its
    ``eval_sets.csv`` on ``stem``. The FoldBench monomer universe is the one
    place where units and stems are in bijection, so a stem join is safe here
    and only here -- the legacy 554 and eval2 universes repeat stems across
    datasets and must never be joined or deduplicated this way.
    """
    targets = pd.read_parquet(TARGETS_PARQUET)
    sets = pd.read_csv(EVAL_SETS_CSV, usecols=["stem", "eval_set"])
    assert targets["stem"].is_unique, "target stems are not unique"
    merged = targets.merge(sets, on="stem", how="inner", validate="one_to_one")
    subset = merged[merged["eval_set"] == eval_set]
    recs = [
        Target(r.dataset, r.stem, int(r.L), r.input_seq)
        for r in subset.sort_values("L").itertuples(index=False)
    ]
    assert recs, f"no targets for eval set {eval_set!r}"
    return recs


def load_ground_truth() -> dict[tuple[str, str], dict]:
    """#245's scored ground-truth universe, keyed by ``(dataset, stem)``."""
    with GROUND_TRUTH_JSONL.open() as fh:
        return {(r["dataset"], r["stem"]): r for r in (json.loads(line) for line in fh)}


def realization(stem: str, residues, tag: str):
    """One contacts-v1 realization: the prompt prefix and its position ring.

    Byte-identical to exp82's ``realization`` helper. ``seq_positions[k]`` is the
    position index the realization assigned to sequence index ``k``; the prefix
    is the document truncated just after ``<begin_statements>``, which is exactly
    what the model conditions on at the start of the structure section.
    """
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
    )

    doc = build_document(f"{stem}:{tag}", residues, [], config=GenerationConfig())
    assert doc is not None, f"{stem}:{tag} could not be serialized as contacts-v1"
    seq_positions = [(doc.n_term_index + k) % NUM_POS for k in range(doc.seq_len)]
    prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
    return prefix, seq_positions


def seed_statement(pos_i: int, pos_j: int, rng: random.Random) -> str:
    """A single ``<contact> <pX> <pY>`` statement with coin-flipped orientation.

    contacts-v1 training documents flip each pair's orientation at random, so a
    seed written in one fixed orientation would sit off the training manifold in
    a way that has nothing to do with the hypothesis under test.
    """
    a, b = (pos_i, pos_j) if rng.random() < 0.5 else (pos_j, pos_i)
    return f" <contact> <p{a}> <p{b}>"


def parse_rollout(text: str, pos_to_seq: dict[int, int]) -> list[tuple[int, int]]:
    """Contacts emitted by one rollout, in emission order, deduped, ``sep >= 6``.

    Returns 0-based ``(lo, hi)`` sequence-index pairs. The order matters: the
    oracle best-of-N readout ranks a rollout's own contacts by it, and only the
    first R survive the cut.
    """
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for x, y in CONTACT_RE.findall(text):
        ia, ib = pos_to_seq.get(int(x)), pos_to_seq.get(int(y))
        if ia is None or ib is None or ia == ib:
            continue
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        if (hi - lo) >= MIN_SEP and (lo, hi) not in seen:
            seen.add((lo, hi))
            out.append((lo, hi))
    return out
