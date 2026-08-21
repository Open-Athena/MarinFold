# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL environment for contacts-v1 — issue #208, SkyRL port.

DELIBERATELY THIN, AND THAT IS THE DESIGN. `BaseTextEnv.step()` is text-in
(`action: str`) and returns a single float. exp208's reward walks **token ids**,
not decoded text, because a dense policy-gradient signal has to land on specific
positions — that was exp200's whole reason for scanning ids, and re-deriving
positions by re-tokenizing decoded text is precisely the class of off-by-one that
`loss_mask.py`'s convention warning exists about.

So the scoring does NOT happen here. This environment carries the per-protein
state the scorer needs — ground truth contacts and this realization's
`<pN> -> sequence index` map — and `DenseContactsGenerator` (see
`dense_generator.py`) computes the reward where `response_ids` actually exist.
`step()` reports only a coarse scalar so that SkyRL's own logging and any
turn-level machinery still see something sane.

The prompt realizations are resampled per rollout, exactly as on the marin.rl
path and in exp82's eval worker: each rollout gets a different `<pN>` rotation
and a different sequence-statement order, seeded by `f"{entry_id}:r{k}"`. That
resampling is what the consensus vote is built out of, so it is a property of the
task rather than a detail of any one framework.
"""

from typing import Any, Dict, Optional

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

import contact_rewards as cr


class ContactsV1Env(BaseTextEnv):
    """One protein realization: ground truth, position map, and a coarse score.

    Args:
        extras: SkyRL passes the dataset row through here. Expected keys:
            ``entry_id``, ``L``, ``gt_contacts`` (list of ``[i, j]`` PAIRS in
            sequence-index space — NOT a flat ``[i0, j0, i1, ...]`` vector, which
            is the shape used for predictions; confusing the two silently halves
            the ground truth and pairs unrelated residues), and ``seq_positions``
            (this realization's ``<pN>`` order).
    """

    def __init__(self, env_config: Optional[Dict[str, Any]] = None, extras: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.max_turns = 1                    # single-shot generation, no tools
        extras = extras or {}
        # Accept a JSON string as well as a dict: parquet round-trips nested
        # structures either way depending on how the dataset was written, and a
        # silent AttributeError here would surface as an empty ground truth
        # (every contact scored wrong) rather than as a load failure.
        if isinstance(extras, str):
            import json as _json
            extras = _json.loads(extras)
        self.entry_id: str = extras.get("entry_id", "")
        self.length: int = int(extras.get("L", 0))
        pairs = extras.get("gt_contacts", []) or []
        gt = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs}
        self.gt = {p for p in gt if cr.in_band(p)}
        self.pos_to_seq = {int(p): i for i, p in enumerate(extras.get("seq_positions", []) or [])}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Coarse scalar only — the dense reward is computed in the generator.

        Returning 0.0 here would make SkyRL's own reward logging useless, so this
        reports the fraction of ground-truth contacts recovered, parsed from text.
        It is never the training signal.
        """
        self.turns += 1
        hits = 0
        if self.gt and self.pos_to_seq:
            seen = set()
            for raw_i, raw_j in cr.CONTACT_TEXT_RE.findall(action) if hasattr(cr, "CONTACT_TEXT_RE") else []:
                a, b = self.pos_to_seq.get(int(raw_i)), self.pos_to_seq.get(int(raw_j))
                if a is None or b is None:
                    continue
                pair = (min(a, b), max(a, b))
                if cr.in_band(pair) and pair not in seen:
                    seen.add(pair)
                    hits += pair in self.gt
        return BaseTextEnvStepOutput(
            observations=[],
            reward=float(hits) / len(self.gt) if self.gt else 0.0,
            done=True,
            metadata={
                # The generator reads these back to score token ids properly.
                "entry_id": self.entry_id,
                "gt": sorted(self.gt),
                "pos_to_seq": self.pos_to_seq,
            },
        )


__all__ = ["ContactsV1Env"]
