# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct batch-priority Fray dispatch of the exp163 refiner fine-tune (issue #163).

Warm-start Eric's tuned contacts-v1 1.5B (E8, ``prot-exp75-...-bc3084`` step-35679,
eval loss 2.7566) and continue-train it on the rollout-**refinement** corpus with
the answer-span loss mask, on the CoreWeave **rno-2a** H100 cluster (``cw-rno2a``)
at iris **batch** priority: given a protein's sequence + K noisy candidate
rollouts, emit the TRUE contact set.

Modelled on exp108's ``dispatch_train.py`` (+ its
``train_qwen_3b_contacts_v1_sweep.py`` entry, merged here so this file is the
``python -m`` entry point) and on the sibling ``dispatch_rollouts.py``. Like
exp108 we submit each LR as its OWN ``fray.types.JobRequest(priority=3)`` gang
that WE control — the marin executor submits child jobs with NO priority band
(-> interactive), and #163/#108 require the batch band for all CoreWeave work.

The training config itself lives in ``refine_ft_common.build_on_pod_config``;
this module is the launcher (env knobs -> JobRequests -> submit -> wait).

Recipe: a dedicated **1-epoch cosine** continue-train (peak LR decays to the
``min_lr_ratio=0.1`` floor over exactly one epoch) at the low LRs {1e-4, 3e-4},
``weight_decay=0.2``, ``warmup=0.1``, global batch **128** (== Eric's #75 training
batch — no batch-scaling confound), seq **8192**. The train corpus is answer-span
**masked**; the val corpus (original contacts-v1 val) is monitored **unmasked**
every eval, as a base-task-retention monitor.

WARM-START VERIFICATION — use **bpb, not loss**. #163 originally called for "step-0
val loss ~ 2.7566" (Eric's logged E8 value). That check does not work as written,
for two independent reasons found on the first smoke run:

* levanter has **no step-0 eval** on this path — hooks fire at multiples of
  ``steps_per_eval``, so the first recorded value already includes LR-warmup
  damage. Use the ``EXP163_STEPS_PER_EVAL=1 EXP163_MAX_STEPS=1`` probe for an
  anchor.
* per-token ``loss`` is **not comparable across harnesses**. levanter's ``bpb``
  divides by summed per-token-type byte lengths weighted by the loss mask, so a
  different packing/eval config changes the effective bytes-per-token and hence
  the loss scale, at identical model quality. Measured: Eric logged
  loss 2.75660 / bpb 0.39151 (10.16 bytes/token); this harness gives
  loss 3.16941 / bpb 0.39489 (11.58 bytes/token) — the loss ratio 1.1497 is
  exactly (bpb ratio 1.0086) x (bytes/token ratio 1.1399). **bpb agrees to 0.9%**,
  which is the real signal that E8 loaded. Random init would be ~ln(2845) = 7.95.

--------------------------------------------------------------------------------
PREREQUISITES on S3 (must exist BEFORE this can run):
  1. E8 **HF export** (config.json + sharded safetensors) — already staged:
       s3://marin-us-east-02a/MarinFold/exp163/model/step-35679
     (the same artifact ``dispatch_rollouts.py`` feeds to vLLM; levanter's HF
      converter reads fsspec URLs, so it doubles as the warm-start source.)
  2. The **tokenized** refinement corpus (`input_ids` + `loss_weights`), built by
     ``tokenize_refinement_corpus.py`` from the raw corpus:
       s3://marin-us-east-02a/MarinFold/exp163/val10k/refinement_tokenized/*.parquet
  3. The unmasked val split (original contacts-v1 val; exp108 staged this tree):
       s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/val/*.parquet
--------------------------------------------------------------------------------

Env knobs (so a smoke run needs no code edit):
  EXP163_LRS              comma list of peak LRs (default "1e-4,3e-4"); set ONE for a smoke
  EXP163_EPOCHS           default 1
  EXP163_STEPS_PER_EPOCH  REQUIRED unless EXP163_TRAIN_SEQUENCES is set.
                          ``tokenize_refinement_corpus.py`` PRINTS this number.
  EXP163_TRAIN_SEQUENCES  alternative: packed sequences in the corpus; steps/epoch
                          = ceil(sequences / batch)
  EXP163_MAX_STEPS        cap steps for a smoke run (default: full 1-epoch count)
  EXP163_ARMS             comma list of multi-draft loss-weight arms to run, e.g.
                          "A,B,C,D". Each arm N reads the corpus tokenized with its
                          own weight profile from .../scale50k/tok_<N> and gets its
                          own run name. Unset = the v1 single-corpus behaviour.
  EXP163_STEPS_PER_EVAL   override the eval cadence (default: a quarter epoch).
                          Set to 1 with EXP163_MAX_STEPS=1 for a warm-start anchor probe.
  EXP163_REPLICAS         number of 8xH100 nodes per run (default 1 = 8 GPUs; <=4)
  EXP163_CORPUS           tokenized train corpus glob (S3 default above)
  EXP163_VAL              unmasked val glob            (S3 default above)
  EXP163_INIT_HF          E8 HF export dir             (S3 default above)
  EXP163_INIT_CKPT        Levanter-native ckpt dir; wins over EXP163_INIT_HF if set
  EXP163_S3_PREFIX        output/cache prefix (default s3://marin-us-east-02a/MarinFold/exp163)
  WANDB_API_KEY           forwarded into the pod (does NOT inherit the launch shell)

Launch (batch priority — required by #163):

    cd experiments/exp163_models_teach_contacts_v1_to_refine_a
    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \\
        --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \\
        -e WANDB_API_KEY "$WK" -e EXP163_STEPS_PER_EPOCH <N> \\
        -- python -m dispatch_refine_train

Smoke first (ONE LR, ~50 steps, single 8xH100 node) — confirm the batch fits, the
job reports the batch band, the E8 warm-start loads (step-0 val ~ 2.7566), and the
S3 caches build:

    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \\
        --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB --extra gpu \\
        -e WANDB_API_KEY "$WK" -e EXP163_LRS 1e-4 -e EXP163_STEPS_PER_EPOCH 50 \\
        -e EXP163_MAX_STEPS 50 -- python -m dispatch_refine_train

Dry-run locally (build + print the JobRequests, no submit):

    EXP163_DRY_RUN=1 EXP163_STEPS_PER_EPOCH=100 python -m dispatch_refine_train
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os

# fray symbols live in ``fray.types`` in this marin build — a plain
# ``from fray import ResourceConfig`` FAILS here. Mirrors dispatch_rollouts.py.
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, JobStatus, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env, run_levanter_train_lm

from refine_ft_common import (
    EXP163_S3_PREFIX,
    INIT_FROM_HF,
    INIT_FROM_LEVANTER,
    LR_SCHEDULE,
    PROTEIN_RESOURCES_H100,
    PROTEIN_RESOURCES_TPU,
    REFINEMENT_TOKENIZED_GLOB,
    SEQ_LEN,
    TRAIN_BATCH,
    VAL_GLOB,
    WARMUP,
    WEIGHT_DECAY,
    build_on_pod_config,
)

logger = logging.getLogger(__name__)

# iris PriorityBand enum value (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray
# maps JobRequest.priority (int) straight to the iris band (iris_backend.submit():
# priority_band=request.priority).
IRIS_PRIORITY_BAND_BATCH = 3
IRIS_PRIORITY_BAND_INTERACTIVE = 0


def _device_label(resources, replicas: int) -> str:
    """Human-readable device string for the launch banner (GPU count or TPU type)."""
    dev = getattr(resources, "device", None)
    variant = getattr(dev, "variant", None) or getattr(dev, "tpu_type", None)
    count = getattr(dev, "count", None)
    if count:                                  # GpuConfig
        return f"{replicas * count} {variant}/run"
    return f"{variant} x{replicas}/run"        # TpuConfig


def priority_band() -> int:
    """Batch on CoreWeave GPUs (#108's hard requirement); interactive on marin TPU.

    Not an inconsistency: the bands mean different things in the two pools. On
    CoreWeave, batch keeps long training runs from displacing interactive work. On the
    marin v5p pool, nearly everything IS interactive, so a batch job simply never gets
    scheduled and never signals demand to the autoscaler.
    """
    if os.environ.get("EXP163_DEVICE", "gpu").lower() == "tpu":
        return IRIS_PRIORITY_BAND_INTERACTIVE
    return IRIS_PRIORITY_BAND_BATCH

# Fail loudly on the frozen 0.99.dev fray, whose JobRequest has no `priority` field,
# so `priority=3` would be silently dropped -> interactive band (which would disrupt
# the very interactive users batch priority protects). Same guard as exp108/exp112/
# dispatch_rollouts.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line (exp108/exp112 pins), not the frozen 0.99.dev build."
)

# Runtime-tuning env vars forwarded from the driver to the training gang. Iris tasks
# don't inherit the submitter's shell, and the gang runs in a SEPARATE pod from this
# driver, so anything passed to the driver via `iris job run -e XLA_FLAGS ...` (or
# NCCL_*/JAX_*) must be re-exported explicitly onto the gang (verbatim from exp108).
# JAX_PLATFORMS is excluded so the CPU driver's value can't leak onto the GPU gang.
# The prefix is ``XLA_`` (not just ``XLA_FLAGS``) so ``XLA_PYTHON_CLIENT_MEM_FRACTION``
# gets through. That one matters on H100s: JAX preallocates only **75%** of each device
# by default, and this recipe (batch 128 x seq 8192 with ``per_device_parallelism=-1``
# => 16 seqs x 8192 tokens per GPU, no microbatching) peaks close enough to that ceiling
# that ``jit__train_step`` OOM'd on a 29.91GiB allocation while ~20GiB per H100 sat
# unusable outside the arena. Raising the fraction is allocator-only -- it does not touch
# the gradient, so runs stay numerically comparable to ones launched without it (unlike
# microbatching, which would re-normalize the per-token loss weights per microbatch and
# so change the effective objective).
_FORWARD_ENV_PREFIXES = ("XLA_", "NCCL_", "JAX_", "LIBTPU_INIT_ARGS")
_FORWARD_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_perf_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_FORWARD_ENV_PREFIXES) and k not in _FORWARD_ENV_EXCLUDE
    }


def _steps_per_epoch() -> int:
    """steps/epoch from EXP163_STEPS_PER_EPOCH, else derived from EXP163_TRAIN_SEQUENCES.

    The refinement corpus is pre-packed into fixed ``SEQ_LEN`` sequences, so one
    full pass is simply ``ceil(sequences / batch)``. ``tokenize_refinement_corpus.py``
    prints both numbers when it writes the corpus.
    """
    steps = os.environ.get("EXP163_STEPS_PER_EPOCH")
    if steps:
        return int(steps)
    sequences = os.environ.get("EXP163_TRAIN_SEQUENCES")
    if sequences:
        return math.ceil(int(sequences) / TRAIN_BATCH)
    raise SystemExit(
        "EXP163_STEPS_PER_EPOCH is required (or set EXP163_TRAIN_SEQUENCES and it is "
        "derived as ceil(sequences / batch)). tokenize_refinement_corpus.py prints "
        "'EXP163_STEPS_PER_EPOCH = <N>' when it writes the corpus."
    )


def _lr_tag(lr: float) -> str:
    """`1e-4`-style tag for run names (mirrors the #75 naming)."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def _lrs() -> list[float]:
    return [float(x) for x in os.environ.get("EXP163_LRS", "1e-4,3e-4").split(",") if x.strip()]


def _epochs() -> int:
    return int(os.environ.get("EXP163_EPOCHS", "1"))


def _arms() -> list[str | None]:
    """Multi-draft loss-weight arms to run, or ``[None]`` for the v1 corpus."""
    spec = os.environ.get("EXP163_ARMS", "").strip()
    return [x.strip() for x in spec.split(",") if x.strip()] if spec else [None]


def corpus_for(arm: str | None) -> str:
    """Corpus glob for an arm.

    Within a sweep, arms train on the SAME documents tokenized with different
    weight profiles — the corpora differ only in ``loss_weights``, never in
    ``input_ids``. An arm name containing ``/`` is taken as a literal path under
    the exp163 prefix (e.g. ``v3/tok_mix50``), which is how a differently-BUILT
    corpus is selected rather than a differently-weighted one.
    """
    if arm is None:
        return REFINEMENT_TOKENIZED_GLOB
    if "/" in arm:
        return f"{EXP163_S3_PREFIX}/{arm}/*.parquet"
    return f"{EXP163_S3_PREFIX}/scale50k/tok_{arm}/*.parquet"


def _run_suffix() -> str:
    """EXP163_RUN_SUFFIX distinguishes a run that shares an arm's CORPUS but must not
    share its output path — otherwise levanter would resume from the finished run's
    checkpoint instead of warm-starting from E8 again."""
    v = os.environ.get("EXP163_RUN_SUFFIX", "").strip().strip("-")
    return f"-{v}" if v else ""


def run_name_for(lr: float, epochs: int, arm: str | None = None) -> str:
    # W&B-safe (alnum + hyphens, < 64 chars). A path-style arm is flattened so the
    # run name (and therefore its checkpoint prefix) stays distinct from the sweeps.
    suffix = f"-{arm.replace('/', '-').replace('_', '-')}" if arm and "/" in arm else (
        f"-md{arm}" if arm else "")
    return (f"plm-exp163-refine-cv1-1_5b-lr{_lr_tag(lr)}-e{epochs}-cos"
            f"{suffix}{_run_suffix()}")


def build_request(
    *,
    run_name: str,
    learning_rate: float,
    num_train_steps: int,
    output_path: str,
    corpus_glob: str,
    resources,
    replicas: int,
    env_vars: dict[str, str],
    steps_per_eval: int,
    steps_per_checkpoint: int,
    hf_save_steps: int | None,
    tags: tuple[str, ...],
    max_retries_failure: int = 3,
) -> JobRequest:
    """Assemble one batch-priority Fray ``JobRequest`` for a single LR gang."""
    # Merge driver-forwarded perf env (XLA_FLAGS/NCCL_/JAX_) under the explicit
    # env_vars (WANDB_*), which win on conflict (exp108).
    env_vars = {**_forwarded_perf_env(), **env_vars}

    on_pod_config = build_on_pod_config(
        run_name=run_name,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        output_path=output_path,
        corpus_glob=corpus_glob,
        resources=resources,
        env_vars=env_vars,
        steps_per_eval=steps_per_eval,
        steps_per_checkpoint=steps_per_checkpoint,
        hf_save_steps=hf_save_steps,
        tags=tags,
    )

    environment = create_environment(
        # resolve_training_env: hardware defaults + GIT_COMMIT + JAX compile cache.
        env_vars=resolve_training_env(base_env=dict(env_vars), resources=resources),
        extras=extras_for_resources(resources),  # GpuConfig -> ["gpu"] (marin-core[gpu])
    )

    return JobRequest(
        name=run_name,  # fray/iris-safe (alnum + hyphens)
        entrypoint=Entrypoint.from_callable(run_levanter_train_lm, args=[on_pod_config]),
        resources=resources,                 # with_gpu("H100", count=8)
        environment=environment,
        replicas=replicas,                   # nodes in the gang
        priority=priority_band(),            # CoreWeave: BATCH (#108). TPU: interactive.
        processes_per_task=1,                # one JAX process driving all 8 local GPUs
        max_retries_failure=max_retries_failure,
    )


def main() -> None:
    lrs = _lrs()
    epochs = _epochs()
    spe = _steps_per_epoch()
    num_train_steps = spe * epochs
    _max_steps = os.environ.get("EXP163_MAX_STEPS")
    if _max_steps:
        num_train_steps = min(num_train_steps, int(_max_steps))

    replicas = int(os.environ.get("EXP163_REPLICAS", "1"))
    assert 1 <= replicas <= 4, f"EXP163_REPLICAS must be in [1, 4], got {replicas}"
    # EXP163_DEVICE=tpu switches the gang to the marin v5p pool. The priority band
    # flips WITH it, deliberately and in the opposite direction to the CoreWeave rule:
    # the v5p pool is dominated by other people's *interactive* jobs, so a batch-band
    # TPU job yields indefinitely on "Insufficient TPUs (need N, available 0)" and
    # registers no autoscaler demand. Interactive is correct for a bounded run there.
    # On CoreWeave the batch band remains mandatory (#108).
    device = os.environ.get("EXP163_DEVICE", "gpu").lower()
    resources = PROTEIN_RESOURCES_TPU if device == "tpu" else PROTEIN_RESOURCES_H100
    if replicas != 1:
        resources = dataclasses.replace(resources, replicas=replicas)

    # W&B routing — the pod does NOT inherit the launcher's shell, so forward the key
    # from the driver env (set at launch with `-e WANDB_API_KEY <key>`). Never hard-coded.
    env_vars: dict[str, str] = {"WANDB_ENTITY": "open-athena"}
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # Eval every quarter epoch; keep a permanent checkpoint every half epoch (the
    # final one is always written) so the HF export / eval has a step-N to read.
    #
    # NOTE: levanter fires eval hooks at multiples of steps_per_eval and there is
    # NO step-0 eval, so a normal run never records the pristine warm-start value
    # (its first eval already includes LR-warmup damage). To capture a near-step-0
    # anchor, run a 1-step probe: EXP163_MAX_STEPS=1 EXP163_STEPS_PER_EVAL=1.
    steps_per_eval = int(os.environ.get("EXP163_STEPS_PER_EVAL") or max(1, spe // 4))
    steps_per_checkpoint = max(1, spe // 2)
    # EXP163_HF_SAVE_STEPS writes an HF export every N steps. Off by default: each
    # export is a full weight copy (~5.9GB for the 1.5B), so this is for a
    # deliberately short probe, not a production run.
    _hf_every = os.environ.get("EXP163_HF_SAVE_STEPS")
    hf_save_steps = int(_hf_every) if _hf_every else None

    warm_start = INIT_FROM_LEVANTER or INIT_FROM_HF
    warm_kind = "Levanter ckpt" if INIT_FROM_LEVANTER else "HF export"
    print(
        f"[exp163] refiner fine-tune (direct batch dispatch): "
        f"LRs={[_lr_tag(lr) for lr in lrs]} wd={WEIGHT_DECAY} warmup={WARMUP} "
        f"sched={LR_SCHEDULE} batch={TRAIN_BATCH} seq={SEQ_LEN} "
        f"replicas={replicas} ({_device_label(resources, replicas)}) | "
        f"{spe} steps/epoch x {epochs} epoch(s) = {spe * epochs} steps"
        + (f" (capped to {num_train_steps} for smoke)" if _max_steps else "")
        + f" | {len(lrs) * len(_arms())} job(s)\n"
        f"         warm-start ({warm_kind}): {warm_start}\n"
        f"         arms:                     {[a or 'v1' for a in _arms()]}\n"
        f"         corpus (masked train):    {corpus_for(_arms()[0])}\n"
        f"         val (unmasked):           {VAL_GLOB}"
    )

    requests: list[JobRequest] = []
    for arm in _arms():
        for lr in lrs:
            name = run_name_for(lr, epochs, arm)
            output_path = f"{EXP163_S3_PREFIX}/checkpoints/{name}"
            tags = (
                "protein", "contacts-v1", "qwen3", "1_5b", "answer-masked",
                "exp163", "refinement", "coreweave", f"e{epochs}", f"lr{_lr_tag(lr)}",
            ) + ((f"multi-draft", f"arm{arm}") if arm else ())
            requests.append(
                build_request(
                    run_name=name,
                    learning_rate=lr,
                    num_train_steps=num_train_steps,
                    output_path=output_path,
                    corpus_glob=corpus_for(arm),
                    resources=resources,
                    replicas=replicas,
                    env_vars=env_vars,
                    steps_per_eval=steps_per_eval,
                    steps_per_checkpoint=steps_per_checkpoint,
                    hf_save_steps=hf_save_steps,
                    tags=tags,
                )
            )

    if os.environ.get("EXP163_DRY_RUN"):
        print("[exp163] DRY RUN -- JobRequests built, not submitting.")
        for req in requests:
            dev = req.resources.device
            inner = req.entrypoint  # keep the built config reachable for eyeballing
            print(
                f"  {req.name}: priority={req.priority} replicas={req.replicas} "
                f"gpu={getattr(dev, 'variant', '?')}x{getattr(dev, 'count', '?')} "
                f"cpu={req.resources.cpu} ram={req.resources.ram} disk={req.resources.disk} "
                f"steps={num_train_steps} extras={extras_for_resources(req.resources)} "
                f"entrypoint={type(inner).__name__}"
            )
        return

    # Runs as an in-cluster CPU driver job -> current_client() is the controller.
    # CRITICAL: the training gangs are CHILDREN of this driver job. If the driver
    # exits first, iris finalizes (kills) them. So submit ALL first (they run
    # concurrently), then block on every one -- the driver must outlive the gangs.
    client = current_client()
    handles = []
    for req in requests:
        job = client.submit(req)
        handles.append((req.name, job))
        print(f"[exp163] submitted {req.name} (job_id={job.job_id})", flush=True)
    band = "batch" if priority_band() == IRIS_PRIORITY_BAND_BATCH else "interactive"
    print(f"[exp163] submitted {len(handles)} gang(s) at iris {band} priority; awaiting completion.")

    # Wait for every gang, reporting per-run status WITHOUT aborting siblings on one
    # failure. Fail the driver at the end if any run did not succeed (visible in iris).
    results: dict[str, JobStatus] = {}
    for name, job in handles:
        status = job.wait(raise_on_failure=False)
        results[name] = status
        print(f"[exp163] {name}: {status}", flush=True)

    failed = [name for name, st in results.items() if st != JobStatus.SUCCEEDED]
    if failed:
        raise SystemExit(f"[exp163] {len(failed)}/{len(handles)} run(s) did not succeed: {failed}")
    print(f"[exp163] all {len(handles)} run(s) SUCCEEDED -> checkpoints under {EXP163_S3_PREFIX}/checkpoints/")


if __name__ == "__main__":
    main()
