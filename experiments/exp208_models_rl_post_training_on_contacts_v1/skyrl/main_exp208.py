# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL entrypoint for exp208 — issue #208, SkyRL port.

Wires the three exp208-specific pieces into SkyRL's stock training loop:

* ``ContactsV1Env`` registered with ``skyrl_gym`` under ``contacts_v1``;
* ``DenseContactsGenerator`` substituted via ``BasePPOExp.get_generator``, which
  is the documented hook — the base implementation picks between the stock text
  and VLM generators, and this adds a third case rather than patching either;
* ``compute_contacts_dense_advantage`` registered with
  ``AdvantageEstimatorRegistry`` and selected by ``adv_estimator=contacts_dense``.

Nothing here monkey-patches SkyRL. Every extension point used is one SkyRL
documents: a gym registration, an overridable factory method, and a function
registry. That is the whole reason for the port — on marin.rl the equivalent
capability rode on `np.full` broadcasting an array `fill_value`, which was
undocumented, unpromised, and needed a test to stop it silently regressing to
constant advantages.

Run ON the GPU host (see the launcher, which takes --host with no default):

    uv run --active python main_exp208.py \\
        trainer.policy.model.path=<hf repo or path> \\
        data.train_data=[<parquet>] \\
        trainer.algorithm.adv_estimator=contacts_dense
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ENV_NAME = "contacts_v1"
ADV_ESTIMATOR = "contacts_dense"


def register_everything(vocab_size: Optional[int] = None) -> None:
    """Register the env and the advantage estimator. Idempotent."""
    import skyrl_gym

    from advantage import register as register_advantage
    from contacts_env_skyrl import ContactsV1Env

    try:
        skyrl_gym.register(id=ENV_NAME, entry_point="contacts_env_skyrl:ContactsV1Env")
    except Exception as exc:      # already registered on a re-entry
        logger.info("skyrl_gym.register(%s): %s", ENV_NAME, exc)
    register_advantage(ADV_ESTIMATOR)
    logger.info("registered env=%s adv_estimator=%s (vocab_size=%s)",
                ENV_NAME, ADV_ESTIMATOR, vocab_size)
    _ = ContactsV1Env      # imported for the side effect of failing loudly here


def build_exp(cfg, *, p_bar: float = 0.45, err_decay: float = 0.5,
              vocab_size: Optional[int] = None, lam_step: float = 1.0,
              lam_doc: float = 4.5):
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


def main() -> int:
    """Hydra entry. Kept thin so the wiring above stays unit-testable."""
    import hydra
    from omegaconf import DictConfig

    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import config_dir

    @hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
    def _run(raw: DictConfig) -> None:
        cfg = SkyRLTrainConfig(**raw) if not isinstance(raw, SkyRLTrainConfig) else raw
        vocab = getattr(getattr(cfg, "exp208", None), "vocab_size", None)
        register_everything(vocab_size=vocab)
        build_exp(cfg, vocab_size=vocab).run()

    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
