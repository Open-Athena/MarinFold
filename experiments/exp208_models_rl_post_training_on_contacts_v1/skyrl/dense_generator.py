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

import contextvars
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

import contact_rewards as cr

logger = logging.getLogger(__name__)


def _unwrap_extras(env_extras: Any) -> Dict[str, Any]:
    """Get to exp208's payload inside whatever SkyRL hands the generator.

    SkyRL passes the dataset row's NON-STANDARD COLUMNS as `env_extras`, so a
    dataset with an `extras` column arrives as ``{"extras": <payload>, "split":
    ...}`` — the payload nested one level down, and still a JSON string because
    that is how it was written to parquet. Measured directly: the override saw
    ``extras_keys=['extras', 'split']`` while looking for `gt_contacts`, found
    nothing, and silently fell back to SkyRL's sparse rewards.

    Handles both shapes so the generator does not care how the dataset was
    written, and returns {} rather than raising — the caller checks for empty
    ground truth and falls back, and the advantage estimator's constant-advantage
    guard catches it downstream either way.
    """
    import json as _json

    if isinstance(env_extras, str):
        try:
            env_extras = _json.loads(env_extras)
        except (ValueError, TypeError):
            return {}
    if not isinstance(env_extras, dict):
        return {}
    if "gt_contacts" not in env_extras and "extras" in env_extras:
        inner = env_extras["extras"]
        if isinstance(inner, str):
            try:
                inner = _json.loads(inner)
            except (ValueError, TypeError):
                return {}
        if isinstance(inner, dict):
            return inner
    return env_extras

# Per-trajectory state, carried from `agent_loop` to `_build_per_token_rewards`.
#
# A ContextVar rather than an attribute on `self`: SkyRL runs trajectories
# CONCURRENTLY under `asyncio.gather`, so a single `self._pending` would be
# overwritten by whichever coroutine ran last and every rollout would be scored
# against another protein's ground truth — quietly, since a mismatched pair set
# just scores as wrong rather than raising. ContextVars are per-task, which is
# exactly the isolation needed here.
_TRAJECTORY: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "exp208_trajectory", default=None
)


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
                 precision_ema_decay: float = 0.9, vocab_size: Optional[int] = None,
                 doc_term: str = "none", lam_step: float = 1.0, lam_doc: float = 0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_term = doc_term
        self.lam_step = float(lam_step)
        self.lam_doc = float(lam_doc)
        self.p_bar = float(p_bar)
        self.err_decay = float(err_decay)
        self.precision_ema_decay = float(precision_ema_decay)
        self.vocab_size = vocab_size
        # instance_id -> {repetition_id: dedup'd pair set}, for the consensus term.
        self._group_pairs: Dict[str, Dict[str, set]] = {}
        self._announced = False

    async def agent_loop(self, prompt, env_class, env_extras, max_tokens, max_input_length,
                         sampling_params=None, trajectory_id=None, cache_salt=None):
        """Publish this trajectory's per-protein state, then run SkyRL's loop.

        `_build_per_token_rewards` is called deep inside `agent_loop` and receives
        only `(per_step_rewards, response_ids, appended_eos_token)` — no env, no
        ground truth, no trajectory id. This is the narrowest place that has all
        of them, so the state is published here and read there.
        """
        extras = _unwrap_extras(env_extras)
        gt = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in extras.get("gt_contacts", [])}
        token = _TRAJECTORY.set({
            "gt": {p for p in gt if cr.in_band(p)},
            "pos_to_seq": {int(p): i for i, p in enumerate(extras.get("seq_positions", []) or [])},
            "instance_id": str(getattr(trajectory_id, "instance_id", extras.get("entry_id", ""))),
            "repetition_id": str(getattr(trajectory_id, "repetition_id", "0")),
            "L": int(extras.get("L", 0)),
        })
        try:
            return await super().agent_loop(
                prompt, env_class, env_extras, max_tokens, max_input_length,
                sampling_params=sampling_params, trajectory_id=trajectory_id,
                cache_salt=cache_salt,
            )
        finally:
            _TRAJECTORY.reset(token)

    def _build_per_token_rewards(
        self, per_step_rewards: List[Tuple[float, Optional[int]]],
        response_ids: List[int], appended_eos_token: bool,
    ):
        """Dense stepwise reward over `response_ids`, or SkyRL's default.

        Falls back to the base implementation whenever the per-protein state is
        missing, so a misconfigured run degrades to ordinary turn-level rewards
        rather than to silently-zero ones.
        """
        state = _TRAJECTORY.get()
        if not self._announced:
            # ONE line, once. Getting here with empty ground truth is the failure
            # that cost this port an afternoon: SkyRL nests the dataset's extra
            # columns, so `gt_contacts` was not where it was looked for, the
            # generator fell back to sparse rewards, and every advantage came out
            # constant. Silent at every layer until the estimator refused it.
            self._announced = True
            logger.warning(
                "[exp208] dense reward active: state=%s gt=%d positions=%d response=%d",
                state is not None, len(state.get("gt", ())) if state else -1,
                len(state.get("pos_to_seq", ())) if state else -1, len(response_ids))
            if not state or not state.get("gt"):
                logger.error(
                    "[exp208] NO GROUND TRUTH -- falling back to SkyRL's sparse rewards, so "
                    "the dense per-contact signal is absent. Check that the dataset's extras "
                    "reach the generator (see _unwrap_extras).")
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

    async def generate(self, input_batch):
        """Run SkyRL's batch generation, then fold in the consensus term.

        A rollout's marginal is ``C(all) - C(all \\ {i})``, so it is undefined
        until every sibling exists. `agent_loop` sees one trajectory at a time;
        this is the first place the whole group is available, which is why the
        document term is applied here rather than alongside the stepwise term.

        With ``doc_term="none"`` (arm S) this is a no-op beyond clearing state,
        so the stepwise-only arm runs through exactly the same code path.
        """
        self.reset_groups()
        out = await super().generate(input_batch)
        if self.doc_term == "none" or self.lam_doc == 0.0:
            return out
        try:
            out = self._fold_document_term(out)
        except Exception:
            # Never let the document term take down a run that has already paid
            # for its rollouts; the stepwise signal is still valid without it.
            logger.exception("[exp208] consensus term failed; stepwise reward stands")
        return out

    def _fold_document_term(self, out):
        """Add the group-centred consensus marginal to each rollout's rewards.

        NOT YET IMPLEMENTED, and deliberately loud rather than a silent no-op.

        The maths is done and tested (`apply_consensus_term` above, and
        `consensus.loo_marginals`, which is pinned equal to the published metric
        implementation). What is missing is the mapping from SkyRL's
        `GeneratorOutput` rows back to `(instance_id, repetition_id)` so each
        rollout's marginal lands on the right reward vector. Getting that mapping
        wrong would attribute one protein's consensus contribution to another —
        silently, because a misattributed marginal is still a plausible number.

        Arm S (`doc_term="none"`) never reaches here, so the stepwise-only arm is
        unaffected. Arms B and F must not run until this is written.
        """
        raise NotImplementedError(
            "the consensus/own-F1 document term is not wired into GeneratorOutput yet; "
            "run arm S (doc_term=none) or implement the rollout->reward mapping first"
        )

    def group_pairs(self) -> Dict[str, Dict[str, set]]:
        """Per-protein rollout pair sets, for the consensus-marginal term."""
        return self._group_pairs

    def reset_groups(self) -> None:
        self._group_pairs.clear()


__all__ = ["DenseContactsGenerator"]
