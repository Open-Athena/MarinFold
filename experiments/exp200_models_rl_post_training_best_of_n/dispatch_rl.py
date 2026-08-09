# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch exp200 RL jobs, one coordinator per learning rate — issue #200.

Two modes.

``--submit`` (from the workstation) submits THIS script as a small CPU driver
job. That indirection is required: ``submit_rl_job`` resolves its client through
``current_client()``, which silently falls back to a ``LocalClient`` when no
cluster context exists, so an off-cluster invocation would quietly run the
coordinator on the workstation instead of the pool.

Without ``--submit`` (on the pod) it builds one ``RLJobConfig`` per learning rate,
submits a coordinator for each, and waits — iris finalizes a job's children when
the parent exits, so a driver that returned early would kill the run.

This directory is the iris workspace (see :mod:`_submit`), so the pod resolves
exp200's own pinned manifest: marin 0.2.76, the last release that still ships
``marin.rl``. TPU work goes in the INTERACTIVE band on the marin v5p pool — the
opposite of the CoreWeave rule. ``submit_rl_job`` never sets a priority, so the
default band is already correct; do not "fix" it to batch.

    uv run python dispatch_rl.py --submit \
        -e EXP200_LRS=1e-6,3e-6,1e-5 -e EXP200_TARGETS=gs://... -e EXP200_PROMPTS=gs://...

    EXP200_DRY_RUN=1 EXP200_TARGETS=... uv run python dispatch_rl.py   # build only
"""

import logging
import os
import sys

from marin.rl.orchestration import submit_rl_job

from rl_config import build_rl_job_config

logger = logging.getLogger(__name__)

# An HF repo id, not a gs:// path: the rollout worker resolves its tokenizer from
# this string via levanter's load_tokenizer, which cannot read object-store URLs.
# Published by dispatch_publish.py from the open-athena bucket copy (the one with
# the renamed <contacts-v1.multi> tokenizer).
DEFAULT_CHECKPOINT = "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404"
DEFAULT_TARGETS = "gs://marin-us-east5/protein-structure/MarinFold/exp200/train/targets.parquet"
DEFAULT_PROMPTS = "gs://marin-us-east5/protein-structure/MarinFold/exp200/train/prompts"
DEFAULT_OUTPUT_PREFIX = "gs://marin-us-east5/protein-structure/MarinFold/exp200"


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"{name} is required (see the module docstring for the launch command)")
    return value


def _float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def build_configs():
    """Build one RLJobConfig per learning rate from the environment knobs."""
    checkpoint = _env("EXP200_CHECKPOINT", DEFAULT_CHECKPOINT)
    suffix = os.environ.get("EXP200_RUN_SUFFIX", "")
    max_sections = int(os.environ.get("EXP200_MAX_SECTIONS", "8"))
    limit = os.environ.get("EXP200_LIMIT")

    configs = []
    for lr in _float_list(os.environ.get("EXP200_LRS", "3e-6")):
        # W&B-safe: alphanumerics and hyphens only.
        lr_tag = f"{lr:.0e}".replace("-", "m").replace("+", "")
        run_name = f"plm-exp200-rl-cv1-1_5b-lr{lr_tag}-s{max_sections}"
        if suffix:
            run_name = f"{run_name}-{suffix}"
        configs.append(
            build_rl_job_config(
                run_name=run_name,
                checkpoint=checkpoint,
                tokenizer=_env("EXP200_TOKENIZER", checkpoint),
                targets_path=_env("EXP200_TARGETS", DEFAULT_TARGETS),
                prompts_path=_env("EXP200_PROMPTS", DEFAULT_PROMPTS),
                output_prefix=_env("EXP200_OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX),
                learning_rate=lr,
                num_train_steps=int(os.environ.get("EXP200_STEPS", "150")),
                train_batch_size=int(os.environ.get("EXP200_TRAIN_BATCH", "32")),
                n_prompts=int(os.environ.get("EXP200_N_PROMPTS", "16")),
                n_generations=int(os.environ.get("EXP200_N_GENERATIONS", "8")),
                max_sections=max_sections,
                lam_step=float(os.environ.get("EXP200_LAM_STEP", "1.0")),
                lam_doc=float(os.environ.get("EXP200_LAM_DOC", "1.0")),
                err_decay=float(os.environ.get("EXP200_ERR_DECAY", "0.5")),
                kl_beta=float(os.environ.get("EXP200_KL_BETA", "0.01")),
                train_tpu_type=os.environ.get("EXP200_TRAIN_TPU", "v5p-8"),
                inference_tpu_type=os.environ.get("EXP200_INFERENCE_TPU", "v5p-8"),
                num_rollout_workers=int(os.environ.get("EXP200_ROLLOUT_WORKERS", "2")),
                regions=tuple(os.environ.get("EXP200_REGIONS", "us-central1").split(",")),
                steps_per_eval=int(os.environ.get("EXP200_STEPS_PER_EVAL", "50")),
                limit=int(limit) if limit else None,
            )
        )
    return configs


def describe(config) -> str:
    """One-screen summary — what a reviewer needs to sanity-check before burning v5p."""
    sampling = next(iter(config.curriculum.lessons.values())).sampling_params
    loss = config.train_params.rl_loss
    return "\n".join(
        [
            f"run_name           {config.run_id}",
            f"  checkpoint       {config.initial_checkpoint}",
            f"  vocab_size       {config.vocab_size}",
            f"  lessons          {sorted(config.curriculum.lessons)} "
            f"(min_sample_prob={config.curriculum.minimum_sample_probability} -> pinned 50:50)",
            f"  lr               {config.train_params.optimizer.learning_rate}",
            f"  steps            {config.trainer.num_train_steps} x batch {config.trainer.train_batch_size}",
            f"  rollouts/step    {sampling.n_prompts} prompts x {sampling.n_generations_per_prompt} gens",
            f"  max_output_tok   {{{', '.join(f'{k}: {v.sampling_params.max_output_tokens}' for k, v in sorted(config.curriculum.lessons.items()))}}}",
            f"  reward           lam_step={loss.lam_step} lam_doc={loss.lam_doc} kl={loss.kl.mode}/{loss.kl.beta}",
            f"  resources        train {config.run_config.train_tpu_type} + "
            f"{config.run_config.num_rollout_workers} x {config.run_config.inference_tpu_type} "
            f"in {config.run_config.regions}",
            f"  rollout spill    {config.rollout_storage.path}",
            f"  checkpoints      {config.trainer.checkpointer.base_path}",
        ]
    )


def wandb_api_key() -> str:
    """W&B key from the environment, else ~/.netrc.

    `fray.create_environment` forwards WANDB_API_KEY from `os.getenv` of whatever
    process calls it — the driver when it submits the coordinator, the coordinator
    when it submits the workers. Nothing else in marin's RL path propagates it:
    the coordinator builds worker env from `{"EQX_ON_ERROR": ...}` plus
    `add_run_env_variables`, which adds only GIT/HF vars, and RLJobConfig has no
    worker-env hook. So if the driver is launched without the key, both workers
    die on `wandb.errors.UsageError: No API key configured` after the gang has
    scheduled and the model has loaded. Reading netrc here makes that
    unforgettable rather than a documented step.
    """
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    import netrc

    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except FileNotFoundError:
        auth = None
    if not auth or not auth[2]:
        raise SystemExit(
            "no W&B API key: set WANDB_API_KEY or log in so ~/.netrc has an "
            "api.wandb.ai entry. Both RL workers call wandb.init() and will fail without it."
        )
    return auth[2]


def submit_driver(argv: list[str]) -> int:
    """Submit this script as a CPU driver job carrying the EXP200_* environment."""
    from _submit import check_clean, submit

    check_clean()
    env = {k: v for k, v in os.environ.items() if k.startswith("EXP200_")}
    env["WANDB_API_KEY"] = wandb_api_key()
    for item in argv:
        key, _, value = item.partition("=")
        env[key] = value
    submit(
        job_name=os.environ.get("EXP200_JOB_NAME", "exp200-rl-driver"),
        extras=("cpu",),
        cpu=2, memory="6GB", disk="16GB",
        region=os.environ.get("EXP200_REGION", "us-east5"),
        command=["python", "-m", "dispatch_rl"],
        env=env,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--submit" in sys.argv:
        rest = [a for a in sys.argv[1:] if a not in ("--submit", "-e")]
        return submit_driver(rest)
    configs = build_configs()

    for config in configs:
        print(describe(config))
        print()

    if os.environ.get("EXP200_DRY_RUN"):
        print(f"[exp200] dry run: {len(configs)} config(s) built, nothing submitted")
        return 0

    handles = []
    for config in configs:
        handle = submit_rl_job(config)
        logger.info("[exp200] submitted %s -> %s", config.run_id, handle)
        handles.append((config.run_id, handle))

    # Iris finalizes children when the parent exits, so the driver has to outlive
    # every coordinator it submitted.
    failures = []
    for run_id, handle in handles:
        try:
            handle.wait(raise_on_failure=True)
            logger.info("[exp200] %s finished", run_id)
        except Exception as exc:  # one arm failing must not abandon the others
            logger.error("[exp200] %s FAILED: %s", run_id, exc)
            failures.append(run_id)
    if failures:
        logger.error("[exp200] %d/%d arm(s) failed: %s", len(failures), len(handles), failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
