# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire `contacts-v1-exp120-1.5B` into the backtracking engine's callables (#159).

The engine (``backtrack_engine.py``) is pure and space-agnostic; this adapter
supplies the two model-backed callables and the seq<->position bookkeeping.

**Coordinate spaces.** The engine runs in **sequence-index** space (0..L-1),
which is where ground-truth contacts live and where the pairwise score matrix
is indexed. The base model, however, speaks **position** tokens (``<pX>``, the
randomized wrap-around index). This adapter owns the one fixed realization of
the sequence section per protein — ``seq_positions[k]`` = the position token
for residue ``k`` — and maps both ways: proposals decoded from the model
(positions) are mapped back to seq indices, and the final edit list is
rendered to a real contacts-v1 document (positions). The map is a bijection
(distinct residues get distinct positions), so folding the rendered document
back with ``read.live_contacts`` recovers exactly the engine's final live set.

- **Proposer.** Prompt = the fixed sequence prefix + the live contacts rendered
  as clean ``<contact> <pI> <pJ>`` statements (no ``<retract>`` — the base
  model never saw one). Sample a few tokens, take the first valid contact,
  map to a seq pair. ``<end>`` first (or nothing valid) → ``None``.
- **Scorer.** Prompt = the fixed prefix + the *committed* contacts; one
  ``inference._fwd_matrix`` pass yields ``s(c)`` for every pair at once
  (``_pcontact_from_fwd``); return the requested targets' entries.

Efficiency note: each call re-prefills the growing prompt (no KV reuse) — fine
for the pilot (whose point is to *measure* per-document wall-clock); a caching
backend is the scale-up lever.
"""

from __future__ import annotations

from collections.abc import Sequence

from marinfold.document_structures.contacts_v1 import inference as inf
from marinfold.document_structures.contacts_v1.read import (
    iter_structure_statements,
    live_contacts,
)
from marinfold.document_structures.contacts_v1.vocab import (
    CONTACT_TOKEN,
    END_TOKEN,
    position_token,
)

from backtrack_engine import Pair, canon


class ModelAdapter:
    """Proposer + Scorer backed by a marinfold transformers/vLLM ``Backend``."""

    def __init__(
        self,
        backend,
        structure: inf.ContactStructure,
        *,
        entry_id: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        propose_tokens: int = 12,
    ):
        built = inf._prefix_and_positions(structure, entry_id=entry_id)
        if built is None:
            raise ValueError(f"cannot serialize {entry_id}")
        self.prefix, self.seq_positions, self.L = built
        self.pos_to_seq = {p: k for k, p in enumerate(self.seq_positions)}
        self.backend = backend
        self.tok = backend.tokenizer
        self.prefix_ids = list(self.tok.encode(self.prefix, add_special_tokens=False))
        self.end_id = inf._token_id(self.tok, END_TOKEN)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.propose_tokens = propose_tokens
        # Metrics the pilot reads back.
        self.n_propose_calls = 0
        self.n_score_calls = 0

    # -- rendering -------------------------------------------------------
    def _render_contacts(self, pairs: Sequence[Pair]) -> str:
        """Seq pairs -> a clean ``<contact> <pI> <pJ> ...`` position string."""
        toks: list[str] = []
        for i, j in pairs:
            toks += ["<contact>", position_token(self.seq_positions[i]),
                     position_token(self.seq_positions[j])]
        return " ".join(toks)

    def _prompt_ids(self, live: Sequence[Pair]) -> list[int]:
        if not live:
            return list(self.prefix_ids)
        extra = self.tok.encode(
            " " + self._render_contacts(live), add_special_tokens=False
        )
        return self.prefix_ids + list(extra)

    # -- Proposer --------------------------------------------------------
    def propose(self, live: Sequence[Pair]) -> Pair | None:
        self.n_propose_calls += 1
        ids = self._prompt_ids(live)
        completion = self.backend.sample_completions(
            [ids],
            max_new_tokens=self.propose_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_token_id=self.end_id,
        )[0]
        text = self.tok.decode(completion, skip_special_tokens=False)
        live_set = set(live)
        for kind, a, b in iter_structure_statements(text):
            if kind != "contact":
                continue
            ia, ib = self.pos_to_seq.get(a), self.pos_to_seq.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            pair = canon(ia, ib)
            if pair not in live_set:   # skip a repeat; try the next statement
                return pair
        return None

    # -- Scorer ----------------------------------------------------------
    def score(
        self, committed: Sequence[Pair], targets: Sequence[Pair]
    ) -> dict[Pair, float]:
        """``s(c)`` for each target pair, conditioned on the committed set.

        Mathematically identical to ``_pcontact_from_fwd(_fwd_matrix(...))``
        read at the target entries, but computed with **only the tails the
        targets need**. ``_fwd_matrix`` runs one tail per residue (L of them)
        to fill the whole [L, L] map; we only ever read a handful of entries,
        so we run tails just for the residues appearing in ``targets``. With a
        queue of ~10 pairs on an L~100 protein that is ~20 tails instead of
        100 — score was the scheduler's serial bottleneck once propose was
        batched.

            s(c) = exp(lp1[i] + lp2[i, j]) + exp(lp1[j] + lp2[j, i])
        """
        import numpy as np

        self.n_score_calls += 1
        pairs = [canon(*t) for t in targets]
        if not pairs:
            return {}
        prompt = self.prefix
        if committed:
            prompt = self.prefix + " " + self._render_contacts(committed)

        tokenizer = self.backend.tokenizer
        prefix_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        contact_id = inf._token_id(tokenizer, CONTACT_TOKEN)
        pos_ids = [inf._token_id(tokenizer, position_token(p)) for p in self.seq_positions]

        # lp1 over every residue: a single forward pass after `<contact>`.
        p1 = self.backend.next_token_probs(prefix_ids, [[contact_id]], pos_ids)
        lp1 = np.log(np.clip(np.asarray(p1[0], dtype=np.float64), inf._PROB_FLOOR, None))

        # lp2 rows only for residues that appear in a target pair.
        needed = sorted({i for i, _ in pairs} | {j for _, j in pairs})
        row_of = {k: r for r, k in enumerate(needed)}
        tails = [[contact_id, pos_ids[k]] for k in needed]
        p2 = self.backend.next_token_probs(prefix_ids, tails, pos_ids)
        lp2 = np.log(np.clip(np.asarray(p2, dtype=np.float64), inf._PROB_FLOOR, None))

        out: dict[Pair, float] = {}
        for i, j in pairs:
            fwd_ij = lp1[i] + lp2[row_of[i], j]
            fwd_ji = lp1[j] + lp2[row_of[j], i]
            out[(i, j)] = float(np.exp(fwd_ij) + np.exp(fwd_ji))
        return out

    # -- document assembly + validation ---------------------------------
    def assemble_document(self, statements: Sequence[tuple[str, int, int]]) -> str:
        """Render the engine's seq-space edit list to a real contacts-v1 doc."""
        body: list[str] = []
        for kind, i, j in statements:
            body += [f"<{kind}>", position_token(self.seq_positions[i]),
                     position_token(self.seq_positions[j])]
        return " ".join([self.prefix, *body, END_TOKEN])

    def gt_positions(self, gt_seq: frozenset[Pair]) -> frozenset[Pair]:
        """Map a seq-index GT set to canonical position pairs (fold space)."""
        return frozenset(
            canon(self.seq_positions[i], self.seq_positions[j]) for (i, j) in gt_seq
        )

    def document_folds_to_gt(self, document: str, gt_seq: frozenset[Pair]) -> bool:
        """The corpus's hard invariant, checked on the *rendered* document."""
        return live_contacts(document) == self.gt_positions(gt_seq)
