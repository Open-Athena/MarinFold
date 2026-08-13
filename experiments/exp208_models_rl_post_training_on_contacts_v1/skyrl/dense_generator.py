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

# Consecutive document-term failures tolerated before the run is declared invalid.
_MAX_DOC_TERM_FAILURES = 3


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
                 collapse_ratio: float = 0.2, reward_mode: str = "dense",
                 p_bar_count_weighted: bool = True, **kwargs):
        # BEFORE super().__init__: a bad mode should fail on the config, not after
        # a tokenizer, an engine client and a Ray actor have been constructed.
        if reward_mode not in ("dense", "document_f1"):
            raise ValueError(f"reward_mode must be 'dense' or 'document_f1', got {reward_mode!r}")
        super().__init__(*args, **kwargs)
        self.reward_mode = reward_mode
        self.p_bar_count_weighted = bool(p_bar_count_weighted)
        self.collapse_ratio = float(collapse_ratio)
        self.doc_term = doc_term
        self.lam_step = float(lam_step)
        self.lam_doc = float(lam_doc)
        self.p_bar = float(p_bar)
        self.err_decay = float(err_decay)
        self.precision_ema_decay = float(precision_ema_decay)
        self.vocab_size = vocab_size
        # instance_id -> {repetition_id: dedup'd pair set}, for the consensus term.
        self._group_pairs: Dict[str, Dict[str, set]] = {}
        # instance_id -> {"gt", "L"}. Captured in `agent_loop` rather than re-read
        # from the batch in `_fold_document_term`, so the consensus term is scored
        # against exactly the ground truth the stepwise reward used.
        self._group_meta: Dict[str, Dict[str, Any]] = {}
        self._doc_term_failures = 0
        # First observed pred/gt, the reference for the collapse tripwire below.
        self._pred_per_gt_baseline: Optional[float] = None
        self._announced = False
        # Per-batch contact tallies. Without these the run reports only reward and
        # response length, and those two cannot distinguish "the policy got better
        # at contacts" from "the policy learned to emit more of them" -- the
        # failure mode this reward shape is most exposed to (see `err_decay`).
        self._diag: Dict[str, float] = {}

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
        in_band_gt = {p for p in gt if cr.in_band(p)}
        instance_id = str(getattr(trajectory_id, "instance_id", extras.get("entry_id", "")))
        length = int(extras.get("L", 0))
        token = _TRAJECTORY.set({
            "gt": in_band_gt,
            "pos_to_seq": {int(p): i for i, p in enumerate(extras.get("seq_positions", []) or [])},
            "instance_id": instance_id,
            "repetition_id": str(getattr(trajectory_id, "repetition_id", "0")),
            "L": length,
        })
        self._group_meta[instance_id] = {"gt": in_band_gt, "L": length}
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
            if self.p_bar_count_weighted:
                # Weight the update by how many contacts the rollout actually
                # emitted, i.e. run the EMA per CONTACT rather than per rollout.
                #
                # The unweighted form is what arm S ran, and it drifts UP: a
                # 3-contact rollout that got 2 right moves p_bar as hard as a
                # 200-contact rollout at 0.25, so p_bar estimates the mean of
                # per-rollout ratios instead of the count-weighted precision the
                # reward's zero point needs. Measured over 125 steps, p_bar ended
                # at 0.5501 against a true 0.4733, which makes every contact
                # net-negative and shrinks the policy -- and shorter rollouts have
                # noisier, higher ratios, so the drift feeds itself.
                w = 1.0 - self.precision_ema_decay ** scored
            else:
                w = 1.0 - self.precision_ema_decay
            self.p_bar = (1.0 - w) * self.p_bar + w * observed

        d = self._diag
        d["n_rollouts"] = d.get("n_rollouts", 0.0) + 1.0
        d["scored"] = d.get("scored", 0.0) + float(scored)
        d["correct"] = d.get("correct", 0.0) + float(correct)
        d["gt"] = d.get("gt", 0.0) + float(len(state["gt"]))
        d["resp_tokens"] = d.get("resp_tokens", 0.0) + float(len(response_ids))
        # Accumulated in BOTH modes: it is the reward in document_f1 and a free
        # diagnostic in dense, and the two arms have to be read on one axis.
        d["doc_f1"] = d.get("doc_f1", 0.0) + float(reward.episode_reward)

        pairs = {c.pair for c in cr.walk_contacts(response_ids, state["pos_to_seq"], state["gt"])
                 if c.pair is not None and c.reason == "ok"}
        self._group_pairs.setdefault(state["instance_id"], {})[state["repetition_id"]] = pairs

        if self.reward_mode == "document_f1":
            # ONE scalar for the whole rollout: the section's F1 against ground
            # truth. No per-token shaping, no p_bar, no err_decay -- those
            # constants do not enter the return value at all here, and the
            # baseline comes from the GROUP (GRPO/RLOO centre a scalar reward
            # against its siblings) rather than from an EMA that can drift away
            # from the quantity it is meant to track.
            #
            # F1 is also the thing arm S could not express: its reward is
            # `n_scored * (precision - p_bar)`, which pays for precision and is
            # indifferent to recall except through the count. F1 prices both.
            return float(reward.episode_reward)

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
            # `apply_consensus_term` is what normally applies `lam_step`; on this
            # path it never runs, so a non-unit lam_step would be silently dropped.
            if self.lam_step != 1.0 and out.get("rewards"):
                out["rewards"] = [[self.lam_step * v for v in row] for row in out["rewards"]]
            return self._emit_contact_metrics(out)
        try:
            out = self._fold_document_term(out, input_batch)
            self._doc_term_failures = 0
        except Exception:
            # Don't let one bad batch take down a run that has already paid for its
            # rollouts -- but a mapping bug fails EVERY batch, and silently falling
            # back to stepwise-only would turn arm B/F into arm S without saying so.
            self._doc_term_failures += 1
            logger.exception(
                "[exp208] consensus term failed (%d consecutive); stepwise reward stands",
                self._doc_term_failures,
            )
            if self._doc_term_failures >= _MAX_DOC_TERM_FAILURES:
                raise RuntimeError(
                    f"[exp208] the {self.doc_term!r} document term failed "
                    f"{self._doc_term_failures} batches running -- this arm is not testing what "
                    f"it claims to. Fix the mapping or run doc_term=none deliberately."
                )
        return self._emit_contact_metrics(out)

    def _emit_contact_metrics(self, out):
        """Attach per-contact tallies to `rollout_metrics`.

        `precision` and `pred_per_gt` are the pair that makes the run readable:
        rising reward with rising `pred_per_gt` and falling `precision` is the
        policy emitting more contacts, not better ones. `err_decay` < 1 makes that
        strategy profitable (the k-th error in a section costs only p_bar*decay^k,
        a convergent series, while every correct contact pays a full 1-p_bar), and
        `p_bar` is an EMA of observed precision, so the cheaper errors get, the
        cheaper they get. This is the number that catches it.
        """
        d = self._diag
        if not d.get("n_rollouts"):
            return out
        scored, gt = d.get("scored", 0.0), d.get("gt", 0.0)
        metrics = out.setdefault("rollout_metrics", {}) or {}
        metrics.update({
            "contacts/precision": (d.get("correct", 0.0) / scored) if scored else 0.0,
            "contacts/pred_per_gt": (scored / gt) if gt else 0.0,
            "contacts/scored_per_rollout": scored / d["n_rollouts"],
            "contacts/correct_per_rollout": d.get("correct", 0.0) / d["n_rollouts"],
            "contacts/p_bar": self.p_bar,
            # Logged in BOTH modes so the two arms are compared on one number.
            # In document_f1 this is the reward; in dense it is a free diagnostic,
            # and it is the axis arm S cannot optimise directly.
            "contacts/doc_f1": d.get("doc_f1", 0.0) / d["n_rollouts"],
            "contacts/recall": (d.get("correct", 0.0) / gt) if gt else 0.0,
        })
        out["rollout_metrics"] = metrics
        logger.info(
            "[exp208] contacts: precision=%.4f pred/gt=%.3f scored/rollout=%.1f p_bar=%.4f",
            metrics["contacts/precision"], metrics["contacts/pred_per_gt"],
            metrics["contacts/scored_per_rollout"], self.p_bar,
        )
        self._check_for_collapse(metrics["contacts/pred_per_gt"], gt)
        return out

    def _check_for_collapse(self, pred_per_gt: float, gt: float) -> None:
        """Abort if the policy has stopped emitting contacts.

        SkyRL's FSDP2 policy sharding diverges from the inference engines and the
        first weight sync pushes the divergent copy into them, after which the
        policy is destroyed: measured pred/gt 1.11 -> 0.006 at 8-way, and 1.10 ->
        0.000 at 2-way after two syncs. Nothing else in the stack objects. The run
        keeps training, keeps logging, and reports a falling reward, so the wasted
        compute is only discovered when someone reads the contact tallies.

        pred/gt is stable in a healthy run (0.99-1.31 observed across every
        non-collapsed configuration), so a fall to a fifth of the opening value is
        far outside normal variation and means the rollouts are no longer
        contacts-v1 documents. Fail loudly instead of training on them.
        """
        if not gt:      # no ground truth in this batch: pred/gt is meaningless
            return
        if self._pred_per_gt_baseline is None:
            self._pred_per_gt_baseline = pred_per_gt
            return
        floor = self.collapse_ratio * self._pred_per_gt_baseline
        if pred_per_gt < floor:
            raise RuntimeError(
                f"[exp208] POLICY COLLAPSE: contacts/pred_per_gt fell to {pred_per_gt:.4f}, "
                f"below {self.collapse_ratio:g}x the opening {self._pred_per_gt_baseline:.4f} "
                f"(floor {floor:.4f}). The rollouts are no longer contacts-v1 documents. "
                f"If the policy is sharded (trainer.placement.policy_num_gpus_per_node > 1) "
                f"this is the known SkyRL sharding divergence -- run unsharded. Check "
                f"rollout_train_logprobs_abs_diff_mean: healthy is ~0.017, collapse reads >1."
            )

    def _fold_document_term(self, out, input_batch=None):
        """Add the group-centred consensus marginal to each rollout's rewards.

        The row -> rollout mapping is read from `out["trajectory_ids"]`, which
        carries `(instance_id, repetition_id)` PER ROW. That is deliberate: SkyRL
        documents that `generate` returns rows in input order, but relying on that
        would make a misattributed marginal — one protein's consensus landing on
        another's tokens — a silent, plausible-looking number. Explicit per-row ids
        cannot drift. Index alignment against `input_batch` is only a fallback, and
        if neither source is available this raises rather than guesses.

        Any instance whose rollouts are not all present in `_group_pairs` is
        SKIPPED (stepwise reward stands). A missing sibling changes `C(all)`, so
        its marginals would be computed against a different group than the one
        that was actually sampled.
        """
        rewards = out.get("rewards")
        if not rewards:
            raise RuntimeError("[exp208] generator output has no rewards to fold into")

        traj = out.get("trajectory_ids") or (input_batch or {}).get("trajectory_ids")
        if traj is None:
            raise RuntimeError(
                "[exp208] no trajectory_ids on the generator output or input batch; refusing to "
                "map consensus marginals by position alone"
            )
        if len(traj) != len(rewards):
            raise RuntimeError(
                f"[exp208] trajectory_ids ({len(traj)}) and rewards ({len(rewards)}) disagree"
            )

        keys, rows_by_key = [], {}
        for i, tid in enumerate(traj):
            key = f"{getattr(tid, 'instance_id', '')}:{getattr(tid, 'repetition_id', '')}"
            keys.append(key)
            rows_by_key.setdefault(key, []).append(i)

        # Every rollout must be scored, and no key may appear twice: a duplicate
        # key means two rollouts would share one marginal.
        dupes = {k: v for k, v in rows_by_key.items() if len(v) > 1}
        if dupes:
            raise RuntimeError(f"[exp208] duplicate trajectory ids in one batch: {list(dupes)[:3]}")

        # Drop instances whose group is incomplete -- see the docstring.
        seen_by_instance: Dict[str, set] = {}
        for key in keys:
            inst, _, rep = key.rpartition(":")
            seen_by_instance.setdefault(inst, set()).add(rep)
        usable, skipped = {}, []
        for inst, reps in seen_by_instance.items():
            scored = set(self._group_pairs.get(inst, {}))
            if scored != reps:
                skipped.append(inst)
                continue
            usable[inst] = reps
        if skipped:
            logger.warning(
                "[exp208] consensus term skipped for %d/%d instances with incomplete groups "
                "(stepwise reward stands); first: %s",
                len(skipped), len(seen_by_instance), skipped[:3],
            )

        rewards_by_key = {}
        for key, idxs in rows_by_key.items():
            inst, _, _ = key.rpartition(":")
            if inst in usable:
                rewards_by_key[key] = list(rewards[idxs[0]])

        if not rewards_by_key:
            logger.warning("[exp208] no complete rollout groups this batch; document term is a no-op")
            return out

        gt_by_instance = {i: self._group_meta[i]["gt"] for i in usable if i in self._group_meta}
        lengths = {i: self._group_meta[i]["L"] for i in usable if i in self._group_meta}
        folded = self.apply_consensus_term(
            rewards_by_key, gt_by_instance, lengths,
            lam_step=self.lam_step, lam_doc=self.lam_doc,
        )

        n = 0
        for key, row in folded.items():
            i = rows_by_key[key][0]
            if len(row) != len(rewards[i]):
                raise RuntimeError(
                    f"[exp208] folded reward length {len(row)} != original {len(rewards[i])} for {key}"
                )
            rewards[i] = row
            n += 1
        logger.info("[exp208] consensus term folded into %d/%d rollouts (lam_doc=%.3g)",
                    n, len(rewards), self.lam_doc)
        out["rewards"] = rewards
        return out

    def group_pairs(self) -> Dict[str, Dict[str, set]]:
        """Per-protein rollout pair sets, for the consensus-marginal term."""
        return self._group_pairs

    def reset_groups(self) -> None:
        self._group_pairs.clear()
        self._group_meta.clear()
        self._diag.clear()


__all__ = ["DenseContactsGenerator"]
