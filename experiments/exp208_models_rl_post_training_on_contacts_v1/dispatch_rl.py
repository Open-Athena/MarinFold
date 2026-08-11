# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch exp208 RL arms — issue #208.

ONE DRIVER PER ARM. exp200 ran three learning rates under a single driver and
lost two of them: the driver was preempted, the two trailing arms' trainers
stopped advancing while their rollout workers kept generating, and after ~25
minutes with no step progress they had to be stopped to return the slices. Iris
finalizes a job's children when the parent exits, so a shared driver makes every
arm a single point of failure for its siblings. Here each arm is its own driver
job and one preemption costs exactly one arm.

``--submit`` (from the workstation) submits each arm as a small CPU driver job.
That indirection is required: ``submit_rl_job`` resolves its client through
``current_client()``, which silently falls back to a ``LocalClient`` when no
cluster context exists — so an off-cluster invocation would quietly run the
coordinator on the workstation instead of the pool.

Without ``--submit`` (on the pod) it builds one ``RLJobConfig`` from ``EXP208_ARM``
and waits, because a driver that returned early would kill its own run.

TPU work goes in the INTERACTIVE band on the marin v5p pool — the opposite of the
CoreWeave rule. ``submit_rl_job`` never sets a priority, so the default is
already correct; do not "fix" it to batch, which on this pool never schedules.

    uv run python dispatch_rl.py --submit --arms S,B,D,F
    uv run python dispatch_rl.py --submit --arms probe --steps 30
    EXP208_ARM=B EXP208_DRY_RUN=1 uv run python dispatch_rl.py
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# An HF repo id, not a gs:// path: the rollout worker resolves its tokenizer from
# this string via levanter's load_tokenizer, which cannot read object-store URLs.
DEFAULT_CHECKPOINT = "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199"
# us-central1, where the v5p capacity is AND where the pool already lives.
# check_region_locality enforces the match; exp200 lost a sweep an hour in to
# TransferBudgetExceeded when rollout spill started crossing regions.
DEFAULT_TARGETS = "gs://marin-us-central1/protein-structure/MarinFold/exp200/train/targets.parquet"
DEFAULT_PROMPTS = "gs://marin-us-central1/protein-structure/MarinFold/exp200/train/prompts"
DEFAULT_OUTPUT_PREFIX = "gs://marin-us-central1/protein-structure/MarinFold/exp208"

# The #208 arms. `rho_target` is the MEASURED ratio of the document term's
# magnitude to the stepwise term's, not a raw lambda: the two differ by roughly an
# order of magnitude in natural scale (a per-contact reward of ~p_bar/3 per token
# over ~300 contact tokens, against a consensus marginal of order 0.01 broadcast
# over the whole response), so raw lambdas do not express the axis the issue calls
# primary. The environment logs `rho_doc_over_step` every step so the realised
# ratio is observable rather than assumed.
ARMS = {
    # name: (doc_term, lam_doc, what it tests)
    "S": ("none", 0.0,
          "step-only — the issue's literal question with the minimum machinery, "
          "and the vote-collapse prediction"),
    "B": ("consensus", 30.0, "the main arm: dense step + consensus marginal at rho ~ 1"),
    "D": ("consensus", 90.0, "more document weight — is the axis monotone here"),
    "F": ("own_f1", 30.0,
          "exp200's document term in plain mode — the ablation that decides whether "
          "the consensus FORM is load-bearing or only its weight"),
}
# Same config as B, used for the learning-rate probe.
PROBE_ARM = "B"


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"{name} is required (see the module docstring)")
    return value


def build_config(arm: str):
    """Build the RLJobConfig for one arm from the EXP208_* environment."""
    from rl_config import build_rl_job_config

    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    doc_term, lam_doc, _ = ARMS[arm]
    checkpoint = _env("EXP208_CHECKPOINT", DEFAULT_CHECKPOINT)
    lr = float(os.environ.get("EXP208_LR", "1e-5"))
    steps = int(os.environ.get("EXP208_STEPS", "400"))
    suffix = os.environ.get("EXP208_RUN_SUFFIX", "")

    lr_tag = f"{lr:.0e}".replace("-", "m").replace("+", "")
    run_name = f"plm-exp208-rl-cv1-1_5b-arm{arm}-lr{lr_tag}-s{steps}"
    if suffix:
        run_name = f"{run_name}-{suffix}"

    return build_rl_job_config(
        run_name=run_name,
        checkpoint=checkpoint,
        tokenizer=_env("EXP208_TOKENIZER", checkpoint),
        targets_path=_env("EXP208_TARGETS", DEFAULT_TARGETS),
        prompts_path=_env("EXP208_PROMPTS", DEFAULT_PROMPTS),
        output_prefix=_env("EXP208_OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX),
        learning_rate=lr,
        num_train_steps=steps,
        doc_term=doc_term,
        lam_step=float(os.environ.get("EXP208_LAM_STEP", "1.0")),
        lam_doc=float(os.environ.get("EXP208_LAM_DOC", str(lam_doc))),
        train_batch_size=int(os.environ.get("EXP208_TRAIN_BATCH", "64")),
        n_prompts=int(os.environ.get("EXP208_N_PROMPTS", "8")),
        n_generations=int(os.environ.get("EXP208_N_GENERATIONS", "16")),
        err_decay=float(os.environ.get("EXP208_ERR_DECAY", "0.5")),
        kl_beta=float(os.environ.get("EXP208_KL_BETA", "0.01")),
        train_tpu_type=os.environ.get("EXP208_TRAIN_TPU", "v5p-8"),
        inference_tpu_type=os.environ.get("EXP208_INFERENCE_TPU", "v5p-8"),
        num_rollout_workers=int(os.environ.get("EXP208_ROLLOUT_WORKERS", "4")),
        regions=tuple(os.environ.get("EXP208_REGIONS", "us-central1").split(",")),
        steps_per_eval=int(os.environ.get("EXP208_STEPS_PER_EVAL", "50")),
        checkpoint_every_steps=int(os.environ.get("EXP208_CKPT_EVERY", "25")),
        sync_interval_steps=int(os.environ.get("EXP208_SYNC_INTERVAL", "8")),
        limit=int(os.environ["EXP208_LIMIT"]) if os.environ.get("EXP208_LIMIT") else None,
    )


def describe(config, arm: str) -> str:
    """One screen — what a reviewer needs before burning v5p."""
    sampling = next(iter(config.curriculum.lessons.values())).sampling_params
    loss = config.train_params.rl_loss
    env_args = next(iter(config.curriculum.lessons.values())).env_config.env_args
    return "\n".join([
        f"arm {arm}: {ARMS[arm][2]}",
        f"  run_name         {config.run_id}",
        f"  checkpoint       {config.initial_checkpoint}  (vocab {config.vocab_size})",
        f"  doc_term         {env_args['doc_term']}",
        f"  lr               {config.train_params.optimizer.learning_rate}",
        f"  steps            {config.trainer.num_train_steps} x batch {config.trainer.train_batch_size}"
        f"  ({config.trainer.train_batch_size // sampling.n_generations_per_prompt} groups/batch)",
        f"  rollouts/step    {sampling.n_prompts} prompts x {sampling.n_generations_per_prompt} gens",
        f"  max_output_tok   {sampling.train_decoding.max_output_tokens}",
        f"  reward           lam_step={loss.lam_step} lam_doc={loss.lam_doc} "
        f"kl={loss.kl.mode}/{loss.kl.beta}",
        f"  resources        train {config.run_config.train_tpu_type} + "
        f"{config.run_config.num_rollout_workers} x {config.run_config.inference_tpu_type} "
        f"in {config.run_config.regions}",
        f"  checkpoints      {config.trainer.checkpointer.base_path} "
        f"(keep every {config.trainer.checkpointer.keep[0]['every']} steps)",
    ])


def wandb_api_key() -> str:
    """W&B key from the environment, else ~/.netrc.

    ``fray.create_environment`` forwards WANDB_API_KEY from ``os.getenv`` of
    whatever process calls it — the driver when it submits the coordinator, the
    coordinator when it submits the workers. Nothing else in marin's RL path
    propagates it: the coordinator builds worker env from ``{"EQX_ON_ERROR": ...}``
    plus ``add_run_env_variables``, which adds only GIT/HF vars, and RLJobConfig
    has no worker-env hook. So a driver launched without the key means both
    workers die on ``wandb.errors.UsageError`` after the gang has scheduled and
    the model has loaded.
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


def submit_drivers(arms: list[str], extra_env: dict[str, str]) -> int:
    """One CPU driver job per arm, each carrying its own EXP208_* environment."""
    from _submit import check_clean, submit

    check_clean()
    base = {k: v for k, v in os.environ.items() if k.startswith("EXP208_")}
    base["WANDB_API_KEY"] = wandb_api_key()
    base.update(extra_env)

    names = []
    for arm in arms:
        env = dict(base, EXP208_ARM=arm)
        names.append(submit(
            job_name=f"exp208-rl-arm{arm.lower()}",
            command=["python", "-m", "dispatch_rl"],
            extras=("cpu",), cpu=2, memory="6GB", disk="16GB",
            region=os.environ.get("EXP208_REGION", "us-central1"),
            priority="batch",     # the DRIVER is a cheap CPU babysitter; the TPU
                                  # children are submitted at the pool default.
            env=env,
        ))
    print(f"[exp208] submitted {len(names)} driver(s)")
    for name in names:
        print(f"    /bizon/{name}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--arms", default="S,B,D,F",
                    help="comma-separated arm names, or 'probe' for the LR probe")
    ap.add_argument("--lrs", default=None, help="probe only: comma-separated learning rates")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()

    if a.submit:
        extra = {}
        if a.steps:
            extra["EXP208_STEPS"] = str(a.steps)
        if a.arms == "probe":
            # The LR probe runs ONE arm's config at several learning rates. exp200
            # ended 150 steps at KL 0.00051 — a policy that did not move, which is
            # why its flat result cannot separate "the reward is wrong" from
            # "nothing happened". Pick the LR from the KL trajectory instead.
            lrs = [x for x in (a.lrs or "3e-6,1e-5,3e-5").split(",") if x]
            names = []
            for lr in lrs:
                os.environ["EXP208_LR"] = lr
                os.environ["EXP208_RUN_SUFFIX"] = "probe"
                names.append(lr)
                submit_drivers([PROBE_ARM], {"EXP208_LR": lr, "EXP208_RUN_SUFFIX": "probe",
                                             "EXP208_STEPS": str(a.steps or 30)})
            return 0
        return submit_drivers([x for x in a.arms.split(",") if x], extra)

    arm = _env("EXP208_ARM")
    config = build_config(arm)
    print(describe(config, arm))
    print()
    if os.environ.get("EXP208_DRY_RUN"):
        print("[exp208] dry run: config built, nothing submitted")
        return 0

    from marin.rl.orchestration import submit_rl_job

    handle = submit_rl_job(config)
    logger.info("[exp208] submitted %s -> %s", config.run_id, handle)
    # Iris finalizes children when the parent exits, so the driver must outlive
    # the coordinator it submitted.
    handle.wait(raise_on_failure=True)
    logger.info("[exp208] %s finished", config.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
