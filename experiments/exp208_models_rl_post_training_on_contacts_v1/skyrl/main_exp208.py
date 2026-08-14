# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL entrypoint for exp208 — issue #208, SkyRL port.

Wires the three exp208-specific pieces into SkyRL's stock training loop:

* ``ContactsV1Env`` registered with ``skyrl_gym`` under ``contacts_v1``;
* ``DenseContactsGenerator`` substituted via ``BasePPOExp.get_generator`` — the
  documented factory the base class itself uses to choose between its text and
  VLM generators, so this adds a third case rather than patching either;
* ``compute_contacts_dense_advantage`` registered with
  ``AdvantageEstimatorRegistry`` and selected by
  ``trainer.algorithm.advantage_estimator=contacts_dense``.
  (The field is ``advantage_estimator``; ``adv_estimator`` is the name used
  inside ``compute_advantages_and_returns`` and is NOT the config key. SkyRL's
  config validation rejects the wrong one loudly, which is how this was found.)

Nothing here monkey-patches SkyRL. That is the point of the port: on marin.rl the
equivalent capability rode on ``np.full`` broadcasting an array ``fill_value``,
which was undocumented, unpromised, and needed a dedicated test to stop it
silently regressing to constant advantages.

SHAPE FOLLOWS SKYRL'S OWN EXAMPLES, not a hydra decorator. There is no
``ppo_base_config.yaml`` — the config *is* a dataclass, read with
``from_cli_overrides``, exactly as ``examples/train/algorithms/dapo/main_dapo.py``
does. An earlier draft of this file used ``@hydra.main(config_name=...)`` and
would have failed at startup.

Run ON the GPU host (the launcher takes --host with no default):

    uv run python main_exp208.py \\
        trainer.policy.model.path=<hf repo or path> \\
        data.train_data=[<parquet>] \\
        trainer.algorithm.advantage_estimator=contacts_dense \\
        vocab_size=2845
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import ray
from skyrl.train.config import SkyRLTrainConfig

logger = logging.getLogger(__name__)

ENV_NAME = "contacts_v1"
ADV_ESTIMATOR = "contacts_dense"


@dataclass
class Exp208Config(SkyRLTrainConfig):
    """SkyRL's config plus exp208's four knobs."""

    # p_bar's starting value. Centring the stepwise reward on the policy's own
    # precision is what stops "emit nothing" being optimal: precision is ~0.20 on
    # the AFDB training pool, so a FIXED penalty makes silence the best policy.
    # Phase 0 measured 0.482 on the eval set, but the marin.rl nano's EMA settled
    # near 0.20 on the training distribution, which is the one that matters.
    p_bar: float = 0.26
    # 1.0, i.e. NO decay. Measured on 9,900 Phase 0 rollouts
    # ([analyze_err_decay.py](../analyze_err_decay.py)): at 0.5 the penalty is
    # 2.3% of the positive term and the next wrong contact costs a median of
    # 0.000000 against +0.745 for a correct one, so errors are free and the
    # p_bar baseline -- the whole point of the reward -- does not exist. There is
    # no intermediate setting: ranking quality is flat from 0.0 to 0.9
    # (rho with f1 0.861-0.878) and only recovers at 1.0 (0.9215), which also
    # tracks precision better (0.906 vs 0.827) and volume less (0.471 vs 0.677).
    err_decay: float = 1.0
    lam_step: float = 1.0
    # RE-CALIBRATED 2026-08-13 against measured spreads, after lam_doc=4.5 made
    # arm B numerically identical to arm S (precision and pred/gt agreeing to 3
    # decimals across all 125 steps, same KL). At 4.5 the document term carries
    # **0.42%** of the stepwise term's spread -- it was not a weak signal, it was
    # no signal.
    #
    # Measured on the Phase 0 rollouts: per-rollout stepwise total
    # `n_pred*(precision - p_bar)` has sd 18.6; the group-of-16 centred consensus
    # marginal has sd 0.0174. The document term's TOTAL contribution to a
    # rollout's summed reward is exactly `lam_doc * marginal` (it is spread over
    # the response, so the per-token share times the length is the whole of it).
    # Equal spread therefore needs lam_doc = 18.6/0.0174 ~= 1067.
    #
    # Prior calibrations of this constant have been wrong twice, in both
    # directions, so the arithmetic is spelled out rather than asserted.
    lam_doc: float = 1067.0
    # "none" (arm S, dense stepwise only), "consensus" (arm B), "own_f1" (arm F).
    # B and F additionally need the GeneratorOutput mapping in
    # dense_generator._fold_document_term, which currently raises.
    doc_term: str = "none"
    # Constrains sampling to real token ids. vLLM pads the vocabulary (2845 ->
    # 2848) with zero rows that emit a logit of exactly 0.0, and exp199's logits
    # sit low enough (top-logit median 1.16) that those rows were sampled in
    # 12.4% of tokens, in 256 of 256 rollouts, NaN-ing the marin.rl trainer on
    # step 1. The trap belongs to the engine, not the framework, so it travels.
    vocab_size: Optional[int] = None
    # Abort when contacts/pred_per_gt falls below this fraction of its opening
    # value. SkyRL's sharded policy diverges from the engines and the weight sync
    # destroys the policy (pred/gt 1.11 -> 0.006), and nothing else in the stack
    # notices -- it trains and logs a plausible falling reward on rollouts that
    # are no longer contacts-v1 documents. Healthy pred/gt is 0.99-1.31.
    collapse_ratio: float = 0.2
    # "dense"        -- per-token per-contact reward (arm S), needs advantage_estimator=contacts_dense
    # "document_f1"  -- ONE scalar per rollout, the section F1, with the baseline
    #                   coming from the group. Needs a group estimator (grpo/rloo);
    #                   contacts_dense refuses it by design, since a constant
    #                   per-token advantage is exactly the failure it guards.
    # "novelty" -- dense per-contact, but a CORRECT contact is weighted by how few
    # siblings also found it: r = (1-p_bar)*(floor + (1-floor)*novelty), while a
    # wrong one still costs a full -p_bar. Synthesis of the two failure modes:
    # the stepwise term alone sharpens (coverage -65%), the consensus marginal
    # alone over-emits (KL diverged to 3.96, precision below baseline). Keeping the
    # full wrong-contact penalty makes volume never free; the novelty weight makes
    # duplicating the consensus pay less than finding what the group missed.
    reward_mode: str = "dense"
    novelty_floor: float = 0.25
    # Weight p_bar's EMA by each rollout's contact count. False reproduces arm S,
    # whose p_bar drifted to 0.5501 against a true precision of 0.4733 and shrank
    # the policy; see ARM_S_RESULTS.md. Irrelevant to reward_mode=document_f1,
    # which has no p_bar at all.
    p_bar_count_weighted: bool = True


def register_everything(vocab_size: Optional[int] = None) -> None:
    """Register the env and the advantage estimator. Idempotent."""
    import skyrl_gym

    from advantage import register as register_advantage

    try:
        skyrl_gym.register(id=ENV_NAME, entry_point="contacts_env_skyrl:ContactsV1Env")
    except Exception as exc:      # already registered on a re-entry
        logger.info("skyrl_gym.register(%s): %s", ENV_NAME, exc)
    register_advantage(ADV_ESTIMATOR)
    logger.info("registered env=%s adv_estimator=%s vocab_size=%s",
                ENV_NAME, ADV_ESTIMATOR, vocab_size)


def build_exp(cfg, *, p_bar: float, err_decay: float, vocab_size: Optional[int],
              doc_term: str = "none", lam_step: float = 1.0, lam_doc: float = 0.0,
              collapse_ratio: float = 0.2, reward_mode: str = "dense",
              p_bar_count_weighted: bool = True, novelty_floor: float = 0.25):
    """A ``BasePPOExp`` whose generator emits exp208's dense per-contact reward."""
    from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
    from skyrl.train.entrypoints.main_base import BasePPOExp

    from dense_generator import DenseContactsGenerator

    class Exp208PPOExp(BasePPOExp):
        def get_generator(self, cfg, tokenizer, inference_engine_client):
            return DenseContactsGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=inference_engine_client,
                tokenizer=tokenizer,
                policy_model_name=resolve_policy_model_name(cfg),
                p_bar=p_bar,
                err_decay=err_decay,
                vocab_size=vocab_size,
                doc_term=doc_term,
                lam_step=lam_step,
                lam_doc=lam_doc,
                collapse_ratio=collapse_ratio,
                reward_mode=reward_mode,
                p_bar_count_weighted=p_bar_count_weighted,
                novelty_floor=novelty_floor,
            )

    return Exp208PPOExp(cfg)


def check_reward_mode(cfg) -> None:
    """The reward shape and the advantage estimator have to agree.

    ``document_f1`` returns ONE scalar per rollout, so its advantage is constant
    across the response by construction — which is precisely what
    ``contacts_dense`` raises on. Pairing them fails at the first optimiser step,
    minutes into a run, with an error about a missing dense signal that describes
    a different problem entirely. Conversely a group estimator (grpo/rloo) given
    the dense per-token reward collapses it to a single number per rollout,
    discarding the per-contact signal #208 exists to test — silently, with no
    error at all. That direction is the dangerous one.
    """
    estimator = cfg.trainer.algorithm.advantage_estimator
    mode = cfg.reward_mode

    # lam_step=0 makes the reward SEQUENCE-level even in "dense" mode: every
    # response token carries the same document share, so there is no per-token
    # signal to preserve and `contacts_dense` (a pass-through with no baseline)
    # would feed the raw, unnormalised marginal straight to the optimiser. That
    # is what arm C did: terminal KL 0.00036, the least movement of any arm, and
    # a checkpoint indistinguishable from its warm start. A group estimator both
    # fits the reward's shape and normalises it.
    if mode == "dense" and cfg.lam_step == 0.0:
        if estimator == ADV_ESTIMATOR:
            raise ValueError(
                "lam_step=0 makes the per-token reward constant within each rollout, so "
                f"advantage_estimator='{ADV_ESTIMATOR}' has nothing per-token to propagate and "
                "will hand the optimiser an unnormalised sequence-level advantage (arm C: "
                "terminal KL 0.00036, no measurable training). Use advantage_estimator=grpo."
            )
        logger.info("[exp208] lam_step=0: sequence-level reward from the document term, "
                    "estimator=%s", estimator)
        return

    if mode == "document_f1" and estimator == ADV_ESTIMATOR:
        raise ValueError(
            f"reward_mode='document_f1' emits one scalar per rollout, but "
            f"advantage_estimator='{ADV_ESTIMATOR}' requires a per-token signal and will "
            f"refuse it at the first step. Use advantage_estimator=grpo (or rloo)."
        )
    if mode == "novelty" and estimator != ADV_ESTIMATOR:
        raise ValueError(
            f"reward_mode='novelty' emits a per-token reward and needs "
            f"advantage_estimator={ADV_ESTIMATOR}, got '{estimator}'.")
    if mode == "dense" and estimator != ADV_ESTIMATOR:
        raise ValueError(
            f"reward_mode='dense' emits a per-token reward, but advantage_estimator="
            f"'{estimator}' would reduce it to one number per rollout and discard the "
            f"per-contact signal WITHOUT error. Use advantage_estimator={ADV_ESTIMATOR}."
        )
    logger.info("[exp208] reward_mode=%s advantage_estimator=%s", mode, estimator)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: Exp208Config):
    """Registration happens HERE, inside the ray task.

    Generators and environments are constructed in ray actors, so registering
    only in the launching process would leave them invisible to the workers —
    the env would fail to resolve by name, or the stock estimator would be
    selected and the dense reward silently ignored.
    """
    register_everything(vocab_size=cfg.vocab_size)
    build_exp(cfg, p_bar=cfg.p_bar, err_decay=cfg.err_decay, vocab_size=cfg.vocab_size,
              doc_term=cfg.doc_term, lam_step=cfg.lam_step, lam_doc=cfg.lam_doc,
              collapse_ratio=cfg.collapse_ratio, reward_mode=cfg.reward_mode,
              p_bar_count_weighted=cfg.p_bar_count_weighted,
              novelty_floor=cfg.novelty_floor).run()


def main() -> int:
    from skyrl.train.utils import initialize_ray, validate_cfg

    cfg = Exp208Config.from_cli_overrides(sys.argv[1:])
    # Register HERE TOO, before validate_cfg. The validator checks
    # `advantage_estimator` against the live registry, so registering only inside
    # the ray task (which runs later) fails with "invalid advantage_estimator:
    # contacts_dense. Must be one of [...]". Both registrations are needed and
    # neither is redundant: this one satisfies validation in the launching
    # process, the one in `skyrl_entrypoint` makes the estimator and env visible
    # to the ray actors that actually construct them.
    register_everything(vocab_size=cfg.vocab_size)
    check_reward_mode(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
