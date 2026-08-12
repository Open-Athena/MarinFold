# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dense per-contact rewards inside SkyRL's generator — issue #208, SkyRL port.

WHY HERE AND NOT IN THE ENVIRONMENT. `BaseTextEnv.step()` is text-in and returns
one float. exp208's stepwise term must land on the three tokens of each
`<contact> <pI> <pJ>` triple, which means it has to be computed against
**response token ids**. Re-deriving positions by re-tokenizing decoded text is
the exact off-by-one class that exp163's `loss_mask.py` convention warning is
about, and exp200 walked ids specifically to avoid it.

`SkyRLGymGenerator.agent_loop` is the first place that both exists: it holds
`response_ids`, and it holds `agent_loop_output.step_outputs`, whose `metadata`
carries the ground truth and position map this environment put there. So the
override lands on `_build_per_token_rewards`, with the per-protein state passed
through a small side-channel populated in the same loop.

WHAT SKYRL GIVES US THAT MARIN.RL DID NOT. The generator already builds
`token_level_rewards = [0.0] * len(response_ids)` and hands a
`[batch, response_len]` tensor to the advantage estimator. A dense vector is the
*documented* return type of this method — `Union[float, List[float]]`. On
marin.rl the same capability existed only because `np.full` happens to broadcast
an array `fill_value`, undocumented and pinned by a test because a silent
regression there degrades to constant advantages and reads as "RL didn't help".

The **document** term (the leave-one-out consensus marginal) needs the whole
rollout group, which the generator identifies via
`trajectory_id.instance_id` / `repetition_id` — protein identity and
rollout-within-protein. It is accumulated per instance and applied by the
registered advantage estimator, which receives SkyRL's own `index` grouping.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

import contact_rewards as cr


class DenseContactsGenerator(SkyRLGymGenerator):
    """`SkyRLGymGenerator` that emits exp208's dense per-contact reward.

    Attributes:
        p_bar: The policy's own EMA per-contact precision. Centering the stepwise
            reward on it is what keeps "emit nothing" from being optimal —
            precision is ~0.2 on the training pool, so a FIXED penalty makes
            silence the best policy and the run collapses to empty sections.
        err_decay: Geometric decay on the penalty for repeat errors within a
            rollout; a wrong contact may be the consequence of an earlier one.
    """

    def __init__(self, *args, p_bar: float = 0.45, err_decay: float = 0.5,
                 precision_ema_decay: float = 0.9, vocab_size: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_bar = float(p_bar)
        self.err_decay = float(err_decay)
        self.precision_ema_decay = float(precision_ema_decay)
        self.vocab_size = vocab_size
        # instance_id -> {repetition_id: dedup'd pair set}, for the consensus term.
        self._group_pairs: Dict[str, Dict[str, set]] = {}
        self._pending: Optional[Dict] = None

    def _build_per_token_rewards(
        self, per_step_rewards: List[Tuple[float, Optional[int]]],
        response_ids: List[int], appended_eos_token: bool,
    ):
        """Dense stepwise reward over `response_ids`, or SkyRL's default.

        Falls back to the base implementation whenever the per-protein state is
        missing, so a misconfigured run degrades to ordinary turn-level rewards
        rather than to silently-zero ones.
        """
        state = self._pending
        if not state or not state.get("gt"):
            return super()._build_per_token_rewards(per_step_rewards, response_ids, appended_eos_token)

        if self.vocab_size:
            # The exp208 NaN: vLLM samples its own vocab padding (2845 -> 2848),
            # those ids do not exist in the embedding, and the trainer dies on
            # step 1. Engine-agnostic, so the guard travels with the port.
            worst = max(response_ids) if response_ids else -1
            if worst >= self.vocab_size:
                n_oov = sum(t >= self.vocab_size for t in response_ids)
                raise ValueError(
                    f"sampled {n_oov}/{len(response_ids)} token ids outside the model "
                    f"vocabulary (max {worst}, vocab_size {self.vocab_size}); constrain the "
                    "sampler to [0, vocab_size) before training on these."
                )

        reward = cr.dense_rewards(
            response_ids, state["pos_to_seq"], state["gt"],
            mode="plain", precision_baseline=self.p_bar, err_decay=self.err_decay,
        )
        scored = reward.diagnostics.get("n_contacts_scored", 0.0)
        correct = reward.diagnostics.get("n_contacts_correct", 0.0)
        if scored > 0:      # EMA update AFTER scoring, so one step shares a baseline
            observed = correct / scored
            self.p_bar = self.precision_ema_decay * self.p_bar + (1 - self.precision_ema_decay) * observed

        pairs = {c.pair for c in cr.walk_contacts(response_ids, state["pos_to_seq"], state["gt"])
                 if c.pair is not None and c.reason == "ok"}
        self._group_pairs.setdefault(state["instance_id"], {})[state["repetition_id"]] = pairs

        return [float(x) for x in reward.token_rewards]

    def apply_consensus_term(self, rewards_by_key: Dict[str, List[float]],
                             gt_by_instance: Dict[str, set], lengths: Dict[str, int],
                             lam_step: float = 1.0, lam_doc: float = 4.5) -> Dict[str, List[float]]:
        """Fold the leave-one-out consensus marginal into the per-token rewards.

        MUST run once the whole group exists. A rollout's marginal is
        ``C(all) - C(all \\ {i})``, so it is undefined until every sibling has
        been generated — which is why this is a group-level pass rather than
        something `_build_per_token_rewards` could do inline.

        The marginal is spread evenly over the rollout's response tokens, because
        that is how a per-sequence advantage reaches a per-token loss; `lam_doc`
        is calibrated so the two terms contribute comparably (measured on the
        marin.rl path: the document term must be integrated over the response
        before comparing, or it reads ~400x too small).
        """
        import consensus as cs

        for instance_id, per_rep in self._group_pairs.items():
            gt = gt_by_instance.get(instance_id)
            length = lengths.get(instance_id)
            if not gt or not length:
                continue
            reps = sorted(per_rep)
            pairs, position = cs.candidate_index(length)
            is_true = cs.truth_mask(pairs, gt)
            votes = cs.vote_counts([per_rep[r] for r in reps], position, len(pairs))
            _, marginals = cs.loo_marginals(votes, is_true, int(is_true.sum()))
            marginals = np.nan_to_num(marginals, nan=0.0)
            marginals = marginals - marginals.mean()      # centre across the group
            for rep, marg in zip(reps, marginals):
                key = f"{instance_id}:{rep}"
                row = rewards_by_key.get(key)
                if not row:
                    continue
                share = float(lam_doc) * float(marg) / max(len(row), 1)
                rewards_by_key[key] = [lam_step * v + share for v in row]
        return rewards_by_key

    def group_pairs(self) -> Dict[str, Dict[str, set]]:
        """Per-protein rollout pair sets, for the consensus-marginal term."""
        return self._group_pairs

    def reset_groups(self) -> None:
        self._group_pairs.clear()


__all__ = ["DenseContactsGenerator"]
