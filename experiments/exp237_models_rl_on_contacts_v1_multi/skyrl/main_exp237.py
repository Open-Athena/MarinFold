# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL entrypoint for exp237 — issue #237.

Wires three exp237-specific pieces into SkyRL's stock training loop:

* ``ContactsV1Env`` registered with ``skyrl_gym`` under ``contacts_v1`` (vendored
  from #208 unchanged — it carries per-protein state and deliberately does no
  scoring, because the reward walks token ids and `BaseTextEnv.step` is
  text-in/float-out);
* ``MultiSectionGenerator`` substituted via ``BasePPOExp.get_generator``, the
  documented factory the base class itself uses to choose between its text and
  VLM generators;
* ``compute_section_advantage`` registered with ``AdvantageEstimatorRegistry``
  and selected by ``trainer.algorithm.advantage_estimator=contacts_section``.

Nothing monkey-patches SkyRL.

Run ON the GPU host (the launcher takes --host with no default)::

    python main_exp237.py \\
        trainer.policy.model.path=<local dir or hf repo> \\
        data.train_data=[<parquet>] \\
        trainer.algorithm.advantage_estimator=contacts_section \\
        reward_mode=section_consensus vocab_size=2845
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import ray
from skyrl.train.config import SkyRLTrainConfig

logger = logging.getLogger(__name__)

ENV_NAME = "contacts_v1"
ADV_ESTIMATOR = "contacts_section"
ADV_ROLLOUT = "contacts_rollout"
GROUP_ESTIMATORS = ("grpo", "rloo")


@dataclass
class Exp237Config(SkyRLTrainConfig):
    """SkyRL's config plus exp237's knobs."""

    # "section_consensus" (arm M-C)  -- per-section leave-one-out marginal on the
    #                                   rollout's OWN consensus, group-centred.
    #                                   Needs advantage_estimator=contacts_section.
    # "final_f1"          (arm M-F)  -- F1 of the LAST section, one scalar, GRPO
    #                                   group baseline. #230 measured last 0.4566
    #                                   against best 0.5342, so +0.078 of headroom
    #                                   exists in selection alone.
    # "best_f1"           (arm M-B)  -- F1 of the BEST section (ORACLE). Raises the
    #                                   ceiling rather than the selector; reported
    #                                   knowing it optimises a quantity that is not
    #                                   deployable on its own.
    # "best_plus_consensus" (arm M-BC) -- the two ROLLOUT-level scalars, each
    #                                   GRPO-standardised over the prompt group and
    #                                   then summed with `lam_consensus`. Neither
    #                                   term can be gamed by section count, which is
    #                                   why this blend and not M-B + M-C's marginal.
    #                                   Needs advantage_estimator=contacts_rollout.
    # "final_plus_consensus" (arm M-FC) -- SYNTHESIS rather than selection.
    #                                   GRPO(F1(last)) + lam * GRPO(C_i(all)).
    #                                   Measured on M-B's own generations: an
    #                                   ORACLE selector of one section reads
    #                                   0.5646, while the consensus of the
    #                                   rollout's OWN preceding drafts reads
    #                                   0.5750 -- so selecting is dominated, and
    #                                   the target for a final section is to
    #                                   synthesise the drafts it can already see.
    #                                   The consensus term is also the restoring
    #                                   force M-F lacked: C(all) collapses under a
    #                                   section-count runaway (0.33 at M-F's worst
    #                                   against ~0.50 healthy).
    #                                   Needs advantage_estimator=contacts_rollout.
    # "consensus_only"    (arm M-K)  -- the rollout's OWN consensus R-precision,
    #                                   one scalar, GRPO baseline. The deployed
    #                                   metric computed on the object the model
    #                                   emits, and the ONLY reward here that is
    #                                   scale-correct in the section count by
    #                                   construction: measured advantage -1.37 at
    #                                   one section against +0.79 at 22, exactly
    #                                   inverting M-C's +4.79 / -0.22. Needs a
    #                                   group estimator (grpo).
    reward_mode: str = "section_consensus"
    # Constrains sampling to real token ids. vLLM pads the vocabulary
    # (2845 -> 2848) with zero rows that emit a logit of exactly 0.0, and
    # contacts-v1 logits sit low enough (top-logit median 1.16) that #208 measured
    # those rows taking 12.4% of sampled tokens in 256 of 256 rollouts, NaN-ing
    # the trainer on step 1. The trap belongs to the engine, not the framework.
    vocab_size: Optional[int] = None
    # Overall scale on the section advantage. 1.0 by default and it should stay
    # there: the advantage is already normalised to unit spread per prompt group,
    # so this is the same units GRPO's own advantage is in, and the learning rate
    # is the knob to turn instead.
    lam: float = 1.0
    # Arm M-BC's mixing weight. Because both terms are standardised SEPARATELY,
    # this is a ratio of standardised quantities: 1.0 means the best-section and
    # consensus objectives get equal weight in units of within-group spread. That
    # interpretability is the point -- #208's `lam_doc` had to be calibrated
    # against raw spreads and was wrong twice, in both directions.
    lam_consensus: float = 1.0
    # `beta * min(0, K - floor)` on the candidate count, added to the RAW scalar
    # of the final_f1 / best_f1 arms. OFF by default. Arm M-B's `max_k F1` is
    # count-NEUTRAL -- it does not depend on K -- so unlike arm M-C there is no
    # divergent term for this to fight, and unlike a two-sided bonus it cannot
    # pay for padding. beta 0.03 puts a rollout at K = 11 against a floor of 18
    # roughly two within-group standard deviations down; at K = 1 it is -0.51
    # against a reward in [0, 1], i.e. decisive and bounded.
    count_penalty_beta: float = 0.0
    # Has to sit ABOVE the count at which the run dies, not at it. M-B's decline
    # is 20.2 -> 19.3 -> 15.9 -> 11.0 sections/rollout and the gate fires at 11,
    # so a floor of 10 is identically zero for the whole decline and engages only
    # after the run has already been stopped.
    count_penalty_floor: float = 18.0

    # ---- #237's preregistered diversity kill criteria, checked every batch ----
    # #230's checkpoint reads 22.0 sections, Jaccard 0.304 and 658 union pairs
    # per rollout, so it starts AT exp200's 0.30 diversity-collapse criterion
    # before any RL. These are the numbers that decide whether an arm's result is
    # a result or a collapse.
    min_sections: float = 12.0
    max_jaccard: float = 0.45
    # Added after arm M-F ran to 146-259 sections carrying 1.4 contacts each
    # without tripping ANY gate: every criterion above is one-sided toward a
    # failure already seen, and M-F's pushed all of them away from their
    # thresholds. These are the two instruments that did see it.
    max_sections: float = 60.0
    min_precision: float = 0.15
    # Union pairs per rollout against R = |gt|, which is where #208's coverage
    # mechanism actually lives: R-precision cuts a ranking at R, so zero-vote
    # pairs begin padding the top-R only once the union drops below R.
    min_union_over_r: float = 1.25
    # #237's preregistered coverage criterion -- union against the run's OWN
    # warmup. OFF by default, deliberately. It stopped all three arms, and the
    # evaluation then showed union/R never left 2.8-4.0 in any of them; arm M-B
    # was stopped at step 36 while improving every aggregation mode, consensus
    # included. Set it back to 0.80 to reproduce the preregistered behaviour.
    min_union_ratio: float = 0.0
    # Stop the run when a gate is violated on 3 consecutive batches. Tripping a
    # preregistered kill criterion IS the result, and continuing past it only
    # spends GPU hours confirming it.
    gates_fatal: bool = True


def register_everything(vocab_size: Optional[int] = None) -> None:
    """Register the env and the advantage estimator. Idempotent."""
    import skyrl_gym

    from advantage_section import register as register_advantage
    from advantage_section import register_rollout

    try:
        skyrl_gym.register(id=ENV_NAME, entry_point="contacts_env_skyrl:ContactsV1Env")
    except Exception as exc:      # already registered on a re-entry
        logger.info("skyrl_gym.register(%s): %s", ENV_NAME, exc)
    register_advantage(ADV_ESTIMATOR)
    register_rollout(ADV_ROLLOUT)
    logger.info("registered env=%s adv_estimators=%s,%s vocab_size=%s",
                ENV_NAME, ADV_ESTIMATOR, ADV_ROLLOUT, vocab_size)


def build_exp(cfg):
    """A ``BasePPOExp`` whose generator emits exp237's section reward."""
    from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
    from skyrl.train.entrypoints.main_base import BasePPOExp

    from multi_generator import MultiSectionGenerator

    class Exp237PPOExp(BasePPOExp):
        def get_generator(self, cfg, tokenizer, inference_engine_client):
            return MultiSectionGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=inference_engine_client,
                tokenizer=tokenizer,
                policy_model_name=resolve_policy_model_name(cfg),
                reward_mode=cfg.reward_mode,
                vocab_size=cfg.vocab_size,
                lam=cfg.lam,
                min_sections=cfg.min_sections,
                max_jaccard=cfg.max_jaccard,
                min_union_ratio=cfg.min_union_ratio,
                min_union_over_r=cfg.min_union_over_r,
                lam_consensus=cfg.lam_consensus,
                max_sections=cfg.max_sections,
                min_precision=cfg.min_precision,
                gates_fatal=cfg.gates_fatal,
                count_penalty_beta=cfg.count_penalty_beta,
                count_penalty_floor=cfg.count_penalty_floor,
            )

    return Exp237PPOExp(cfg)


def check_reward_mode(cfg) -> None:
    """The reward's shape and the advantage estimator have to agree.

    One direction fails loudly and the other fails silently, so both are checked
    here rather than left to the first optimiser step:

    * a scalar reward (``final_f1`` / ``best_f1``) under ``contacts_section``
      produces an advantage that is constant across the response by construction,
      which is precisely what the estimator's guard raises on — minutes into a
      run, with a message about a missing dense signal that describes a different
      problem;
    * the per-section reward under a group estimator is the dangerous direction:
      ``grpo`` sums the token rewards to one number per rollout and broadcasts it
      back, discarding the within-sequence credit assignment that is the whole of
      #237 — with **no error at all**.
    """
    estimator = cfg.trainer.algorithm.advantage_estimator
    mode = cfg.reward_mode

    if mode == "section_consensus" and estimator != ADV_ESTIMATOR:
        raise ValueError(
            f"reward_mode='section_consensus' emits a per-section reward, but "
            f"advantage_estimator='{estimator}' would sum it to one number per rollout and "
            f"discard the within-rollout credit assignment WITHOUT error. Use "
            f"advantage_estimator={ADV_ESTIMATOR}.")
    if mode in ("best_plus_consensus", "final_plus_consensus") and estimator != ADV_ROLLOUT:
        raise ValueError(
            f"reward_mode='{mode}' writes a finished per-rollout advantage, so "
            f"it needs advantage_estimator={ADV_ROLLOUT} (a pass-through). '{estimator}' would "
            f"either re-standardise an already-standardised blend (grpo) or refuse it for "
            f"being constant within the rollout ({ADV_ESTIMATOR}).")
    if mode in ("final_f1", "best_f1", "consensus_only"):
        if estimator == ADV_ESTIMATOR:
            raise ValueError(
                f"reward_mode='{mode}' emits one scalar per rollout, but advantage_estimator="
                f"'{ADV_ESTIMATOR}' requires a per-token signal and will refuse it at the first "
                f"step. Use advantage_estimator=grpo.")
        if estimator not in GROUP_ESTIMATORS:
            raise ValueError(
                f"reward_mode='{mode}' has no baseline of its own -- it needs a group estimator "
                f"{GROUP_ESTIMATORS} to centre it, got '{estimator}'.")
    logger.info("[exp237] reward_mode=%s advantage_estimator=%s", mode, estimator)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: Exp237Config):
    """Registration happens HERE, inside the ray task.

    Generators and environments are constructed in ray actors, so registering
    only in the launching process would leave them invisible to the workers — the
    env would fail to resolve by name, or the stock estimator would be selected
    and the section reward silently ignored.
    """
    register_everything(vocab_size=cfg.vocab_size)
    build_exp(cfg).run()


def main() -> int:
    from skyrl.train.utils import initialize_ray, validate_cfg

    cfg = Exp237Config.from_cli_overrides(sys.argv[1:])
    # Register HERE TOO, before validate_cfg: the validator checks
    # `advantage_estimator` against the live registry, so registering only inside
    # the ray task (which runs later) fails validation. Both registrations are
    # needed and neither is redundant.
    register_everything(vocab_size=cfg.vocab_size)
    check_reward_mode(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
