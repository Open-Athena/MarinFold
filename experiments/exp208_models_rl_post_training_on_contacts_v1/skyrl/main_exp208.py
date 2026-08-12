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
    p_bar: float = 0.45
    err_decay: float = 0.5
    lam_step: float = 1.0
    # CALIBRATED, not chosen. The two terms differ by ~an order of magnitude in
    # natural scale AND the document scalar is broadcast over every response
    # token, so comparing them per rollout understates the document term by the
    # response length; a plausible guess was off by ~65x on the marin.rl path.
    lam_doc: float = 4.5
    # Constrains sampling to real token ids. vLLM pads the vocabulary (2845 ->
    # 2848) with zero rows that emit a logit of exactly 0.0, and exp199's logits
    # sit low enough (top-logit median 1.16) that those rows were sampled in
    # 12.4% of tokens, in 256 of 256 rollouts, NaN-ing the marin.rl trainer on
    # step 1. The trap belongs to the engine, not the framework, so it travels.
    vocab_size: Optional[int] = None


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


def build_exp(cfg, *, p_bar: float, err_decay: float, vocab_size: Optional[int]):
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
            )

    return Exp208PPOExp(cfg)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: Exp208Config):
    """Registration happens HERE, inside the ray task.

    Generators and environments are constructed in ray actors, so registering
    only in the launching process would leave them invisible to the workers —
    the env would fail to resolve by name, or the stock estimator would be
    selected and the dense reward silently ignored.
    """
    register_everything(vocab_size=cfg.vocab_size)
    build_exp(cfg, p_bar=cfg.p_bar, err_decay=cfg.err_decay,
              vocab_size=cfg.vocab_size).run()


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
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
