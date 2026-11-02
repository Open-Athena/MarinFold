# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL generator that rewards SECTIONS of one ``<contacts-v1.multi>`` rollout — issue #237.

Structurally this is #208's `dense_generator.DenseContactsGenerator` with one
substitution: everywhere that file grouped **rollouts of a prompt**, this one
groups **sections of a rollout**. The plumbing it inherits — the ContextVar
side-channel, `_unwrap_extras`, the vocab-padding guard, the explicit
`trajectory_ids` row mapping — is unchanged because every one of those was paid
for by a silent failure that would recur here identically.

What is *not* inherited is the reward. #208's per-contact stepwise term is
deliberately absent: its own conclusion is that a ``p̄``-centred per-contact
reward is a sharpening operator to first order, and that no second-order
redistribution overcomes it. Issue #237 states the consequence as a rule — **no
arm here should be per-contact-only** — so this generator emits only
section-level quantities.

Three reward modes, one per arm:

``section_consensus`` (arm M-C)
    Per-section, dense. Section *k*'s reward is its leave-one-out marginal
    contribution to its **own rollout's** consensus R-precision, centred and
    scaled against every section of every rollout for the same prompt. Needs
    ``advantage_estimator=contacts_section``.

``final_f1`` (arm M-F) and ``best_f1`` (arm M-B)
    One scalar per rollout — the last section's F1, or the best section's F1
    (ORACLE). Needs a group estimator (``grpo``), which supplies the baseline.

Why the group pass exists at all. A section's marginal is defined against its
siblings, so it is computable inside `_build_per_token_rewards`; but the
*baseline* it is centred on is the whole prompt group's, which is not available
until every rollout of that prompt has been generated. `generate` is the first
place both exist, exactly as it was for #208's consensus term.
"""

import contextvars
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

import contact_rewards as cr
import section_rewards as sr

logger = logging.getLogger(__name__)

#: Batches a diversity gate may be violated before the run is stopped. The gates
#: are #237's preregistered kill criteria, so tripping them IS the result; the
#: patience only keeps a noisy window from ending a run early.
_GATE_PATIENCE = 3
#: Batches whose MEDIAN forms the reference the coverage gate compares against.
#: The gates arm after these.
_GATE_WARMUP = 6
#: Batches in the rolling window the gates are evaluated on.
#:
#: A batch is 8 proteins, and the diversity statistics are dominated by which 8.
#: Measured on the first six batches of arm M-C at lr 1e-5, with the policy
#: barely moved (KL 0.0012): union pairs per rollout ranged 440-892, sections
#: 16.4-28.1, Jaccard 0.074-0.279. A gate reading single batches against a single
#: baseline batch would have fired on the protein draw, killed a healthy run, and
#: reported it as #237's preregistered diversity collapse -- the most expensive
#: kind of wrong answer this experiment could produce. Medians over a window,
#: both sides, are what make the comparison about the policy.
_GATE_WINDOW = 6


def _unwrap_extras(env_extras: Any) -> Dict[str, Any]:
    """Get to the experiment's payload inside whatever SkyRL hands the generator.

    Verbatim from #208, and for the reason recorded there: SkyRL passes the
    dataset row's non-standard columns as `env_extras`, so a dataset with an
    ``extras`` column arrives as ``{"extras": <json>, "split": ...}`` — the
    payload nested one level down and still a JSON string. #208's generator
    silently fell back to sparse rewards for an afternoon because of it.
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
# A ContextVar rather than an attribute on `self`: SkyRL runs trajectories
# concurrently under `asyncio.gather`, and an attribute would be overwritten by
# whichever coroutine ran last, scoring every rollout against another protein's
# ground truth — silently, since a mismatched pair set merely scores as wrong.
_TRAJECTORY: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "exp237_trajectory", default=None
)


class MultiSectionGenerator(SkyRLGymGenerator):
    """`SkyRLGymGenerator` whose reward is computed over a rollout's sections.

    Args:
        reward_mode: One of :data:`section_rewards.REWARD_MODES`.
        vocab_size: Constrains the sampled ids to the real vocabulary. vLLM pads
            2845 -> 2848 with zero rows that emit logit 0.0, and #208 measured
            those rows taking 12.4 % of sampled tokens across 256 of 256 rollouts
            before NaN-ing the trainer on step 1. The trap belongs to the
            inference engine, so it travels with any port.
        min_sections / max_jaccard / min_union_over_r: the diversity kill
            criteria, checked every batch.
        min_union_ratio: #237's preregistered coverage criterion — union pairs
            per rollout against the run's own warmup, kill below 0.80. **Default
            0, i.e. off, and that is a deliberate correction.** It stopped all
            three arms, and the evaluation then showed the mechanism it stands
            for was never approached: R-precision cuts at R = |gt|, so zero-vote
            pairs pad the top-R only once the union falls BELOW R, and every arm
            held union/R between 2.8 and 4.0. Arm M-B was stopped at step 36
            while improving *every* aggregation mode, consensus included.
            `min_union_over_r` is the same criterion written in the units the
            mechanism is actually in.
        gates_fatal: Stop the run when a gate is violated `_GATE_PATIENCE`
            batches running. Off turns the gates into warnings.
        count_penalty_beta / count_penalty_floor: `beta * min(0, K - floor)`,
            added to the RAW rollout scalar for the `final_f1` / `best_f1` arms.
            **0.0 (off) everywhere by default.** Arm M-B's reward is `max_k F1`,
            which does not depend on K at all, so the arm has no first-order
            opinion about its own candidate count — and it declined 20.2 -> 19.3
            -> 15.9 -> 11.0 sections before the gate stopped it. This is the term
            that gives it one. See `section_rewards.count_penalty` for why it is
            added raw rather than standardised, and why the floor has to sit
            ABOVE the count at which the run dies rather than at it.
    """

    def __init__(self, *args, reward_mode: str = "section_consensus",
                 vocab_size: Optional[int] = None, lam: float = 1.0,
                 min_sections: float = 12.0, max_jaccard: float = 0.45,
                 min_union_ratio: float = 0.0, min_union_over_r: float = 1.25,
                 lam_consensus: float = 1.0, max_sections: float = 60.0,
                 min_precision: float = 0.15, gates_fatal: bool = True,
                 collapse_ratio: float = 0.2, count_penalty_beta: float = 0.0,
                 count_penalty_floor: float = 18.0, beta_shape: float = 0.0,
                 positional_shape: bool = True, shape_signal: str = "prefix",
                 lam_false: float = 1.0, **kwargs):
        # BEFORE super().__init__: a bad mode should fail on the config, not after
        # a tokenizer, an engine client and a Ray actor have been constructed.
        if reward_mode not in sr.REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {sr.REWARD_MODES}, got {reward_mode!r}")
        super().__init__(*args, **kwargs)
        self.reward_mode = reward_mode
        self.vocab_size = vocab_size
        self.lam = float(lam)
        self.lam_consensus = float(lam_consensus)
        self.max_sections = float(max_sections)
        self.min_precision = float(min_precision)
        self.min_sections = float(min_sections)
        self.max_jaccard = float(max_jaccard)
        self.min_union_ratio = float(min_union_ratio)
        self.min_union_over_r = float(min_union_over_r)
        self.gates_fatal = bool(gates_fatal)
        self.collapse_ratio = float(collapse_ratio)
        self.count_penalty_beta = float(count_penalty_beta)
        self.count_penalty_floor = float(count_penalty_floor)
        self.beta_shape = float(beta_shape)
        self.positional_shape = bool(positional_shape)
        if shape_signal not in ("prefix", "novelty", "pair"):
            raise ValueError(
                f"shape_signal must be 'prefix', 'novelty' or 'pair', got {shape_signal!r}")
        self.shape_signal = shape_signal
        self.lam_false = float(lam_false)
        # instance_id -> {repetition_id: (marginals, bounds, n_response_tokens)}
        self._group: Dict[str, Dict[str, Any]] = {}
        # instance_id -> {repetition_id: (best_f1, consensus, n_response_tokens)}
        # for arm M-BC, whose two terms are both ROLLOUT-level scalars.
        self._rollout_scores: Dict[str, Dict[str, Any]] = {}
        self._diag: Dict[str, float] = {}
        self._announced = False
        self._batches = 0
        self._gate_strikes: Dict[str, int] = {}
        #: Per-batch gate metrics, newest last; only the last _GATE_WINDOW matter.
        self._history: List[Dict[str, float]] = []
        #: Median over the warmup window, the reference the coverage gate uses.
        self._union_baseline: Optional[float] = None

    # ------------------------------------------------------------------ hooks

    async def agent_loop(self, prompt, env_class, env_extras, max_tokens, max_input_length,
                         sampling_params=None, trajectory_id=None, cache_salt=None):
        """Publish this trajectory's per-protein state, then run SkyRL's loop.

        `_build_per_token_rewards` is called deep inside `agent_loop` and receives
        only ``(per_step_rewards, response_ids, appended_eos_token)`` — no env, no
        ground truth, no trajectory id. This is the narrowest place that has all
        of them.
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
        """Section-level reward for one rollout.

        Returns a per-token vector in ``section_consensus`` mode (rewritten by the
        group pass in `generate`, which owns the baseline) and a single float in
        the two scalar modes. Falls back to SkyRL's own implementation when the
        per-protein state is missing, so a misconfigured run degrades to ordinary
        turn-level rewards rather than to silently-zero ones — and says so.
        """
        state = _TRAJECTORY.get()
        if not self._announced:
            self._announced = True
            logger.warning(
                "[exp237] section reward active: mode=%s state=%s gt=%d positions=%d response=%d",
                self.reward_mode, state is not None,
                len(state.get("gt", ())) if state else -1,
                len(state.get("pos_to_seq", ())) if state else -1, len(response_ids))
            if not state or not state.get("gt"):
                logger.error(
                    "[exp237] NO GROUND TRUTH -- falling back to SkyRL's sparse rewards, so the "
                    "section signal is absent. Check that the dataset's extras reach the "
                    "generator (see _unwrap_extras).")
        if not state or not state.get("gt"):
            return super()._build_per_token_rewards(per_step_rewards, response_ids, appended_eos_token)

        if self.vocab_size:
            worst = max(response_ids) if response_ids else -1
            if worst >= self.vocab_size:
                n_oov = sum(t >= self.vocab_size for t in response_ids)
                raise ValueError(
                    f"sampled {n_oov}/{len(response_ids)} token ids outside the model vocabulary "
                    f"(max {worst}, vocab_size {self.vocab_size}); constrain the sampler to "
                    "[0, vocab_size) before training on these.")

        walk = sr.walk_rollout(response_ids, state["pos_to_seq"], state["gt"])
        consensus, marginals = sr.consensus_and_marginals(
            walk.sections, state["gt"], state["L"])
        self._accumulate(walk, consensus, state["gt"])

        if self.reward_mode == "consensus_only":
            # Arm M-K: the rollout's OWN consensus R-precision, one scalar, GRPO
            # baseline. This is the deployed metric computed on the object the
            # model emits, and it is the only reward here that is scale-correct
            # in the section count BY CONSTRUCTION -- dropping sections lowers
            # your own consensus (0.543 at 22 sections, 0.341 at one), so the
            # direction that destroyed arm M-C is penalised rather than paid.
            # NaN (no scoreable ground truth) becomes 0.0; every rollout of such
            # a prompt gets it, so the group is constant and GRPO returns 0.
            return 0.0 if consensus != consensus else float(consensus)

        if self.reward_mode in ("final_f1", "best_f1"):
            # The penalty is added to the RAW scalar, before GRPO's group
            # standardisation. Because GRPO subtracts the group mean, a term
            # that is the same constant across the group cancels exactly -- so
            # a batch in which every rollout clears the floor is untouched, not
            # merely nudged. That exactness is the whole point of the deadband.
            return float(sr.scalar_reward(self.reward_mode, walk, state["gt"])
                         + sr.count_penalty(walk.diagnostics["n_sections"],
                                            self.count_penalty_beta,
                                            self.count_penalty_floor))

        if self.reward_mode == "consensus_shaped":
            cons = 0.0 if consensus != consensus else float(consensus)
            self._rollout_scores.setdefault(state["instance_id"], {})[state["repetition_id"]] = (
                cons,
                (sr.pair_token_advantages(response_ids, state["pos_to_seq"], state["gt"],
                                          lam_false=self.lam_false)
                 if self.shape_signal == "pair"
                 else sr.novelty_marginals(walk.sections, state["gt"], state["L"])
                 if self.shape_signal == "novelty"
                 else sr.prefix_marginals(walk.sections, state["gt"], state["L"])),
                walk.bounds,
                walk.n_response_tokens,
            )
            # Right SHAPE only; `generate` overwrites it once the group exists.
            return [cons] * walk.n_response_tokens

        if self.reward_mode in ("best_plus_consensus", "final_plus_consensus"):
            which = "best_f1" if self.reward_mode == "best_plus_consensus" else "final_f1"
            best = float(sr.scalar_reward(which, walk, state["gt"]))
            # NaN consensus (no scoreable ground truth) becomes 0.0. Every rollout
            # of such a prompt gets it, so the group is constant and standardising
            # returns exactly 0 -- the prompt contributes nothing, which is right.
            cons = 0.0 if consensus != consensus else float(consensus)
            self._rollout_scores.setdefault(state["instance_id"], {})[state["repetition_id"]] = (
                best, cons, walk.n_response_tokens,
            )
            # Placeholder of the right SHAPE; `generate` overwrites it with the
            # blended advantage once the whole group exists.
            return [best] * walk.n_response_tokens

        self._group.setdefault(state["instance_id"], {})[state["repetition_id"]] = (
            marginals, walk.bounds, walk.n_response_tokens,
        )
        # A provisional vector of the right SHAPE. `generate` overwrites it with
        # the group-centred advantage; returning the raw marginals here (rather
        # than zeros) means a run whose group pass somehow never fires produces
        # an obviously-uncentred reward instead of a silently dead one.
        return [float(x) for x in sr.token_advantages(marginals, walk.bounds, walk.n_response_tokens)]

    async def generate(self, input_batch):
        """Run SkyRL's batch generation, then apply the group-level baseline."""
        self.reset_groups()
        out = await super().generate(input_batch)
        self._batches += 1
        if self.reward_mode == "section_consensus":
            out = self._apply_group_baseline(out, input_batch)
        elif self.reward_mode in ("best_plus_consensus", "final_plus_consensus"):
            out = self._apply_rollout_blend(out, input_batch)
        elif self.reward_mode == "consensus_shaped":
            out = self._apply_shaped_consensus(out, input_batch)
        return self._emit_metrics(out)

    # -------------------------------------------------------------- internals

    def _accumulate(self, walk: sr.RolloutSections, consensus: float, gt: set) -> None:
        d = self._diag
        d["n_rollouts"] = d.get("n_rollouts", 0.0) + 1.0
        d["sections"] = d.get("sections", 0.0) + walk.diagnostics["n_sections"]
        d["union"] = d.get("union", 0.0) + walk.diagnostics["union_pairs"]
        d["votes"] = d.get("votes", 0.0) + walk.diagnostics["total_votes"]
        d["empty"] = d.get("empty", 0.0) + walk.diagnostics["n_empty_sections"]
        if gt:
            # union / R -- the quantity #208's coverage mechanism is actually
            # about. R-precision cuts a ranking at R = |gt|, so zero-vote pairs
            # start padding the top-R only once the union falls BELOW R. Measured
            # at eval, every arm here held union/R between 2.8 and 4.0 while a
            # gate defined relative to each run's own opening stopped all three.
            d["union_over_r"] = d.get("union_over_r", 0.0) + \
                walk.diagnostics["union_pairs"] / len(gt)
            d["n_union_over_r"] = d.get("n_union_over_r", 0.0) + 1.0
        d["finished"] = d.get("finished", 0.0) + float(walk.finished)
        d["resp_tokens"] = d.get("resp_tokens", 0.0) + float(walk.n_response_tokens)
        d["scored"] = d.get("scored", 0.0) + float(walk.n_scored)
        d["correct"] = d.get("correct", 0.0) + float(walk.n_correct)
        d["gt"] = d.get("gt", 0.0) + float(len(gt))
        jac = walk.diagnostics["mean_jaccard"]
        if jac == jac:      # not NaN
            d["jaccard"] = d.get("jaccard", 0.0) + jac
            d["n_jaccard"] = d.get("n_jaccard", 0.0) + 1.0
        if consensus == consensus:
            d["consensus"] = d.get("consensus", 0.0) + consensus
            d["n_consensus"] = d.get("n_consensus", 0.0) + 1.0
        f1s = sr.section_f1s(walk.sections, gt)
        if f1s:
            d["best_f1"] = d.get("best_f1", 0.0) + max(f1s)
            d["last_f1"] = d.get("last_f1", 0.0) + f1s[-1]
            d["n_f1"] = d.get("n_f1", 0.0) + 1.0

    def _apply_group_baseline(self, out, input_batch=None):
        """Centre every section marginal against its whole prompt group.

        The row -> rollout mapping is read from ``out["trajectory_ids"]``, which
        carries ``(instance_id, repetition_id)`` per row. SkyRL documents that
        `generate` returns rows in input order, but relying on that would make a
        misattributed advantage — one protein's baseline landing on another's
        tokens — a silent, plausible-looking number.
        """
        rewards = out.get("rewards")
        if not rewards:
            raise RuntimeError("[exp237] generator output has no rewards to centre")
        traj = out.get("trajectory_ids") or (input_batch or {}).get("trajectory_ids")
        if traj is None:
            raise RuntimeError(
                "[exp237] no trajectory_ids on the generator output or input batch; refusing to "
                "map section advantages by position alone")
        if len(traj) != len(rewards):
            raise RuntimeError(
                f"[exp237] trajectory_ids ({len(traj)}) and rewards ({len(rewards)}) disagree")

        row_of: Dict[str, int] = {}
        for i, tid in enumerate(traj):
            key = f"{getattr(tid, 'instance_id', '')}:{getattr(tid, 'repetition_id', '')}"
            if key in row_of:
                raise RuntimeError(f"[exp237] duplicate trajectory id in one batch: {key}")
            row_of[key] = i

        n_written = n_dead = 0
        for instance_id, per_rep in self._group.items():
            marginals = {rep: v[0] for rep, v in per_rep.items()}
            centred = sr.centred_section_advantages(marginals)
            if all(not np.any(v) for v in centred.values()):
                n_dead += 1
            for rep, adv in centred.items():
                key = f"{instance_id}:{rep}"
                row = row_of.get(key)
                if row is None:
                    continue
                _, bounds, n_tok = per_rep[rep]
                vec = self.lam * sr.token_advantages(adv, bounds, n_tok)
                if len(vec) != len(rewards[row]):
                    raise RuntimeError(
                        f"[exp237] advantage vector {len(vec)} != reward row "
                        f"{len(rewards[row])} for {key}")
                rewards[row] = [float(x) for x in vec]
                n_written += 1
        if not n_written:
            raise RuntimeError(
                "[exp237] the group baseline wrote no rewards -- every row's section state is "
                "missing, so the reward is whatever _build_per_token_rewards left behind. "
                "Check the trajectory_id mapping.")
        self._diag["dead_prompts"] = float(n_dead)
        self._diag["n_prompts"] = float(len(self._group))
        logger.info("[exp237] group baseline applied to %d/%d rollouts; %d/%d prompts had zero "
                    "marginal spread", n_written, len(rewards), n_dead, len(self._group))
        out["rewards"] = rewards
        return out

    def _rows(self, out, input_batch=None):
        """``(rewards, {"<instance>:<rep>": row_index})`` for a group pass.

        Factored out of the three group passes rather than repeated: SkyRL
        documents that `generate` returns rows in input order, but relying on
        that would make a misattributed advantage — one protein's baseline
        landing on another's tokens — a silent, plausible-looking number.
        """
        rewards = out.get("rewards")
        if not rewards:
            raise RuntimeError("[exp237] generator output has no rewards to write")
        traj = out.get("trajectory_ids") or (input_batch or {}).get("trajectory_ids")
        if traj is None:
            raise RuntimeError(
                "[exp237] no trajectory_ids on the generator output or input batch; refusing to "
                "map advantages by position alone")
        if len(traj) != len(rewards):
            raise RuntimeError(
                f"[exp237] trajectory_ids ({len(traj)}) and rewards ({len(rewards)}) disagree")
        row_of: Dict[str, int] = {}
        for i, tid in enumerate(traj):
            key = f"{getattr(tid, 'instance_id', '')}:{getattr(tid, 'repetition_id', '')}"
            if key in row_of:
                raise RuntimeError(f"[exp237] duplicate trajectory id in one batch: {key}")
            row_of[key] = i
        return rewards, row_of

    def _apply_shaped_consensus(self, out, input_batch=None):
        """Arm M-KS: arm M-K's base, redistributed within each rollout.

            A_i,k  =  GRPO_group( C_i(all) )_i  +  beta * ( m_k - mean_k m )

        The base is arm M-K exactly — the best consensus measured in #237 —
        and the shaping term is **zero-sum within the rollout**, so it cannot
        move the level the base sets. It only decides *which* sections of a
        good rollout are reinforced: those that added something their
        predecessors had not, rather than every section equally.

        Arm M-K reinforces all ~22 sections of a good rollout identically,
        including the ones that merely repeat their siblings — which is the
        mechanism most consistent with its Jaccard climbing 0.23 -> 0.39 as its
        score turned over. This is the term that separates them.
        """
        rewards, row_of = self._rows(out, input_batch)
        n_written = 0
        for instance_id, per_rep in self._rollout_scores.items():
            reps = sorted(per_rep)
            base = sr.grpo_standardise([per_rep[r][0] for r in reps])
            # The positional baseline is a GROUP quantity, so it can only be
            # formed here. Without it the shaping term is a "stop early" signal:
            # the prefix marginal decays in k by construction, and centring
            # within the rollout removes the level but not the shape.
            positional = (sr.positional_baseline({r: per_rep[r][1] for r in reps})
                          if self.positional_shape and self.shape_signal != "pair" else None)
            for b, rep in zip(base, reps):
                row = row_of.get(f"{instance_id}:{rep}")
                if row is None:
                    continue
                _, marginals, bounds, n_tok = per_rep[rep]
                if self.shape_signal == "pair":
                    # Already a per-TOKEN vector, zero-sum over the pair tokens and
                    # exactly zero on every structural token. No section machinery
                    # touches it -- which is the point of the arm.
                    vec = float(b) + self.beta_shape * np.asarray(marginals, dtype=np.float64)
                else:
                    adv = sr.shaped_section_advantages(float(b), marginals, self.beta_shape,
                                                       positional=positional)
                    vec = sr.token_advantages(adv, bounds, n_tok)
                if len(vec) != len(rewards[row]):
                    raise RuntimeError(
                        f"[exp237] advantage vector {len(vec)} != reward row "
                        f"{len(rewards[row])} for {instance_id}:{rep}")
                rewards[row] = [float(x) for x in vec]
                n_written += 1
        if not n_written:
            raise RuntimeError(
                "[exp237] the shaped-consensus pass wrote no rewards; check the "
                "trajectory_id mapping and that MultiSectionGenerator is configured.")
        logger.info("[exp237] shaped consensus applied to %d/%d rollouts "
                    "(beta_shape=%.3g positional=%s signal=%s)",
                    n_written, len(rewards), self.beta_shape, self.positional_shape,
                    self.shape_signal)
        out["rewards"] = rewards
        return out

    def _apply_rollout_blend(self, out, input_batch=None):
        """Arm M-BC: two rollout-level scalars, each GRPO-standardised, then summed.

            A_i  =  GRPO( max_k F1(section k) )_i  +  lam * GRPO( C_i(all) )_i

        **Standardised SEPARATELY, and that is the design choice.** Because each
        term is divided by its own within-group standard deviation, ``lam`` is a
        ratio of standardised quantities: ``lam = 1`` means "these two objectives
        get equal weight, in units of within-group spread". Standardising the
        *sum* instead — ``GRPO(best + lam * C)`` — would make ``lam`` depend on
        the raw scales of two quantities that are not commensurable, which is
        exactly the calibration #208 got wrong twice with ``lam_doc``.

        Neither term can be gamed by section count: ``max_k F1`` does not depend
        on how many sections there are (only on the best one), and ``C_i(all)``
        *falls* when sections are dropped (0.543 at 22 sections, 0.341 at one).
        That is the whole reason this arm exists rather than blending M-B with
        M-C's per-section marginal, whose magnitude diverges as sections vanish.
        """
        rewards = out.get("rewards")
        if not rewards:
            raise RuntimeError("[exp237] generator output has no rewards to blend")
        traj = out.get("trajectory_ids") or (input_batch or {}).get("trajectory_ids")
        if traj is None:
            raise RuntimeError("[exp237] no trajectory_ids; refusing to map by position")
        row_of = {f"{getattr(t, 'instance_id', '')}:{getattr(t, 'repetition_id', '')}": i
                  for i, t in enumerate(traj)}

        n_written = 0
        diag_best, diag_cons = [], []
        for instance_id, per_rep in self._rollout_scores.items():
            reps = sorted(per_rep)
            a_best = sr.grpo_standardise([per_rep[r][0] for r in reps])
            a_cons = sr.grpo_standardise([per_rep[r][1] for r in reps])
            diag_best.append(float(np.mean([per_rep[r][0] for r in reps])))
            diag_cons.append(float(np.mean([per_rep[r][1] for r in reps])))
            for rep, ab, ac in zip(reps, a_best, a_cons):
                row = row_of.get(f"{instance_id}:{rep}")
                if row is None:
                    continue
                n_tok = per_rep[rep][2]
                if n_tok != len(rewards[row]):
                    raise RuntimeError(
                        f"[exp237] rollout length {n_tok} != reward row {len(rewards[row])}")
                adv = float(ab) + self.lam_consensus * float(ac)
                rewards[row] = [adv] * n_tok
                n_written += 1
        if not n_written:
            raise RuntimeError(
                "[exp237] the rollout blend wrote no rewards -- the trajectory_id mapping is "
                "wrong, and the reward is whatever _build_per_token_rewards left behind.")
        self._diag["blend_best"] = float(np.mean(diag_best)) if diag_best else 0.0
        self._diag["blend_cons"] = float(np.mean(diag_cons)) if diag_cons else 0.0
        logger.info("[exp237] blend applied to %d/%d rollouts (lam_consensus=%.3g)",
                    n_written, len(rewards), self.lam_consensus)
        out["rewards"] = rewards
        return out

    def _emit_metrics(self, out):
        """Attach the section and diversity tallies to ``rollout_metrics``.

        These are the columns #237 asks for every run — ``union pairs``, ``total
        votes`` and ``votes/pair`` — because #208 showed they separate the two
        failure modes (volume collapse vs diversity collapse) where reward and
        accuracy alone cannot.
        """
        d = self._diag
        n = d.get("n_rollouts", 0.0)
        if not n:
            return out
        union = d.get("union", 0.0)
        metrics = out.setdefault("rollout_metrics", {}) or {}
        m = {
            "multi/sections_per_rollout": d.get("sections", 0.0) / n,
            "multi/union_pairs": union / n,
            "multi/total_votes": d.get("votes", 0.0) / n,
            "multi/votes_per_pair": (d.get("votes", 0.0) / union) if union else 0.0,
            "multi/mean_jaccard": (d.get("jaccard", 0.0) / d["n_jaccard"]) if d.get("n_jaccard") else float("nan"),
            "multi/consensus_rprec": (d.get("consensus", 0.0) / d["n_consensus"]) if d.get("n_consensus") else float("nan"),
            "multi/best_f1": (d.get("best_f1", 0.0) / d["n_f1"]) if d.get("n_f1") else float("nan"),
            "multi/last_f1": (d.get("last_f1", 0.0) / d["n_f1"]) if d.get("n_f1") else float("nan"),
            "multi/empty_sections": d.get("empty", 0.0) / n,
            "multi/finished": d.get("finished", 0.0) / n,
            "multi/response_tokens": d.get("resp_tokens", 0.0) / n,
            "contacts/precision": (d.get("correct", 0.0) / d["scored"]) if d.get("scored") else 0.0,
            "contacts/pred_per_gt": (d.get("scored", 0.0) / d["gt"]) if d.get("gt") else 0.0,
            "multi/union_over_r": (d.get("union_over_r", 0.0) / d["n_union_over_r"]) if d.get("n_union_over_r") else float("nan"),
            "multi/dead_prompts": (d.get("dead_prompts", 0.0) / d["n_prompts"]) if d.get("n_prompts") else 0.0,
        }
        metrics.update(m)
        out["rollout_metrics"] = metrics
        # One parseable line per batch. `logger: console` is the configured SkyRL
        # logger, so this is what the results table is reconstructed from.
        logger.warning(
            "[exp237-metrics] batch=%d " % self._batches
            + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in m.items()))
        self._check_gates(m)
        return out

    def _check_gates(self, m: Dict[str, float]) -> None:
        """#237's preregistered kill criteria, evaluated on the training rollouts.

        Every quantity here is a MEDIAN over the last `_GATE_WINDOW` batches, and
        the coverage reference is the median over the warmup window -- not the
        opening batch. See `_GATE_WINDOW` for the measurement that forced that:
        at 8 proteins a batch, the protein draw moves union coverage by 2x while
        the policy has barely moved, so a single-batch gate fires on the data and
        reports it as diversity collapse.

        The eval-time versions of these gates are computed against #230's own
        658 union pairs and 22.0 sections; here the reference is the run's own
        warmup, so the gate measures what RL did rather than what the harness's
        sampling settings did.
        """
        self._history.append(m)
        window = self._history[-_GATE_WINDOW:]

        def med(key: str, rows=None) -> float:
            vals = [r[key] for r in (rows or window) if r.get(key) == r.get(key)]
            return float(np.median(vals)) if vals else float("nan")

        if self._batches == _GATE_WARMUP:
            self._union_baseline = med("multi/union_pairs", self._history)
            logger.warning(
                "[exp237] coverage baseline set to %.1f union pairs/rollout "
                "(median of the first %d batches)", self._union_baseline, _GATE_WARMUP)
        if self._batches < _GATE_WARMUP + _GATE_WINDOW:
            return

        violations = []
        sections = med("multi/sections_per_rollout")
        if sections < self.min_sections:
            violations.append(
                f"sections/rollout median {sections:.2f} < {self.min_sections:g} "
                "(the multi format is collapsing back toward a single document)")
        # Arm M-F failed in a direction none of the gates below could see: 146-259
        # sections carrying 1.4 contacts each. Every existing criterion is
        # one-sided toward a failure already observed, so that run pushed all of
        # them AWAY from their thresholds -- Jaccard 0.003 and 259 sections read
        # as maximally healthy. These two are the instruments that did see it,
        # promoted from diagnostics to gates.
        secs_hi = med("multi/sections_per_rollout")
        if secs_hi > self.max_sections:
            violations.append(
                f"sections/rollout median {secs_hi:.1f} > {self.max_sections:g} (the policy is "
                "spamming section markers; check contacts per section)")
        prec = med("contacts/precision")
        if prec == prec and prec < self.min_precision:
            violations.append(
                f"per-contact precision median {prec:.3f} < {self.min_precision:g} "
                "(the contacts themselves have stopped being right)")
        jac = med("multi/mean_jaccard")
        if jac == jac and jac > self.max_jaccard:
            violations.append(
                f"mean pairwise Jaccard median {jac:.3f} > {self.max_jaccard:g} "
                "(diversity collapse)")
        if self._union_baseline and self.min_union_ratio > 0:
            ratio = med("multi/union_pairs") / self._union_baseline
            if ratio < self.min_union_ratio:
                violations.append(
                    f"union pairs/rollout median fell to {100 * ratio:.0f}% of the warmup "
                    f"{self._union_baseline:.0f} (< {100 * self.min_union_ratio:.0f}%)")
        uor = med("multi/union_over_r")
        if uor == uor and uor < self.min_union_over_r:
            violations.append(
                f"union/R median {uor:.2f} < {self.min_union_over_r:g} (the vote no longer "
                "covers the top-R slots, so they are padded with zero-vote pairs)")

        for key in list(self._gate_strikes):
            if not any(key in v for v in violations):
                self._gate_strikes.pop(key, None)
        for v in violations:
            key = v.split()[0]
            self._gate_strikes[key] = self._gate_strikes.get(key, 0) + 1
            logger.warning("[exp237] GATE (%d/%d): %s", self._gate_strikes[key], _GATE_PATIENCE, v)

        tripped = [k for k, c in self._gate_strikes.items() if c >= _GATE_PATIENCE]
        if tripped and self.gates_fatal:
            raise RuntimeError(
                f"[exp237] KILL CRITERION met for {tripped} on {_GATE_PATIENCE} consecutive "
                f"batches: {violations}. This is #237's preregistered stop, not a crash -- the "
                f"last exported checkpoint is the result. Re-run with gates_fatal=false to "
                f"observe the collapse instead of stopping at it.")

    def reset_groups(self) -> None:
        self._group.clear()
        self._rollout_scores.clear()
        self._diag.clear()


__all__ = ["MultiSectionGenerator"]
