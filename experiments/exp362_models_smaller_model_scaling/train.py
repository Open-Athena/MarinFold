"""Depth-only scaling of the exact exp277 full-corpus training recipe."""

import os
from dataclasses import replace

import click
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.exp232_sweep_cv1_decontam.training_contract import MODEL_CONFIG, VOCAB_SIZE
from experiments.exp277_models_single_mpnn_pilot.config import CORPORA
from experiments.exp277_models_single_mpnn_pilot.prepare import verify_cache
from experiments.exp277_models_single_mpnn_pilot.train import build_run as baseline_run
from experiments.exp362_models_smaller_model_scaling.runtime import run_train_job

PREFIX = "s3://marin-us-east-02a/MarinFold/exp362_models_smaller_model_scaling"
VERSION = "2026.10.13.1"
TRIALS = {"depth4": 4, "depth8": 8, "depth16": 16}
PROFILES = {
    "cw-us-east-02a": ("H100", 8),
    "cw-rno2a": ("H100", 8),
    "cw-us-east-08a": ("GB200", 4),
}


def run_identity(trial: str, *, smoke: bool, cluster: str, nodes: int, attempt: int) -> str:
    """Keep production identity independent of placement and restart attempt."""
    if trial not in TRIALS:
        raise ValueError(f"Unknown trial: {trial}")
    identity = f"contacts-v1-exp362-{trial}-full-epoch"
    if smoke:
        identity += f"-smoke-{cluster}-n{nodes}-a{attempt:02d}"
    return identity


def parameter_count(layers: int) -> int:
    """Include QK-normalization weights omitted by Qwen's analytic counter."""
    model = replace(MODEL_CONFIG, num_layers=layers)
    return int(model.total_trainable_params(VOCAB_SIZE)) + 2 * layers * model.actual_head_size


def build_run(
    *, trial: str, smoke: bool, cluster: str, nodes: int, attempt: int = 1
) -> ArtifactStep[LevanterCheckpoint]:
    """Reuse the baseline builder, replacing depth and operational identities."""
    if cluster not in PROFILES or nodes not in (1, 2, 4, 8, 16):
        raise ValueError(f"Unsupported placement: {cluster}, {nodes}")
    identity = run_identity(trial, smoke=smoke, cluster=cluster, nodes=nodes, attempt=attempt)
    gpu, per_node = PROFILES[cluster]
    step = baseline_run(smoke=smoke, nodes=nodes)
    base_build = step.build_config
    resources = ResourceConfig.with_gpu(
        gpu, count=per_node, replicas=nodes, cpu=32, ram="256g", disk="256g"
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod = base_build(ctx)
        config = pod.train_config
        trainer = replace(
            config.trainer,
            id=identity,
            tracker=replace(
                config.trainer.tracker,
                name=identity,
                group="exp362-smaller-model-scaling",
                tags=[
                    "exp362", trial, "full-corpus-one-epoch", "depth-only",
                    "smoke" if smoke else "production",
                    f"cluster={cluster}", f"gpu={gpu}", f"nodes={nodes}",
                ],
            ),
        )
        if not ctx.is_fingerprint:
            per_device = min(8, 128 // (per_node * nodes))
            trainer = replace(
                trainer,
                per_device_parallelism=per_device,
                per_device_eval_parallelism=per_device,
            )
        return replace(
            pod,
            train_config=replace(
                config,
                trainer=trainer,
                model=replace(config.model, num_layers=TRIALS[trial]),
            ),
            resources=resources,
            env_vars={**pod.env_vars, "MARIN_PREFIX": PREFIX},
        )

    return replace(
        step,
        name=f"runs/{identity}",
        build_config=build_config,
        run=run_train_job,
        runtime_args={**step.runtime_args, "train_resources": resources},
        override_path=f"{PREFIX}/runs/{identity}",
    )


@click.command()
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    """Verify complete inputs and train one explicitly selected trial."""
    for corpus in CORPORA:
        if not verify_cache(corpus):
            raise ValueError(f"Incomplete cache: {corpus.cache}")
    return build_run(
        trial=os.environ["TRIAL"],
        smoke=os.environ.get("SMOKE") == "1",
        cluster=os.environ["TARGET_CLUSTER"],
        nodes=int(os.environ["NODES"]),
        attempt=int(os.environ["ATTEMPT"]),
    )


if __name__ == "__main__":
    main()
