# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview or run one matched recipe phase using Levanter's stock entry point."""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import draccus
import jax
import jmp
import numpy as np
import wandb
from jax.experimental import multihost_utils
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig, discover_latest_checkpoint
from levanter.data.text.datasets import DatasetComponent, DatasetComponentBase
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.main.train_lm import main as train_main
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync
from rigging.filesystem.storage_path import StoragePath

from experiments.exp232_sweep_cv1_decontam.training_contract import SHUFFLE
from scripts.history import _existing_run_files

from .checkpoints import validate_training_restore
from .data import ContactDataConfig
from .inputs import ROOT, verify_manifest
from .model import reference_model_config
from .recipe import AFDB_TOKENS, ESM_TOKENS, PHASES, TOKENIZER, optimizer_for_phase


def phase_for_update(update: int) -> str | None:
    """Resolve the prescribed phase from the full-state next update number."""
    for name, phase in PHASES.items():
        if phase.start <= update < phase.stop:
            return name
    if update == PHASES["final"].stop:
        return None
    raise ValueError(f"Checkpoint update {update} is outside the experiment")


@TrackerConfig.register_subclass("exp279_wandb")
@dataclass
class HistoryWandbConfig(WandbConfig):
    """Use the normal tracker and immediately write the required run history."""

    def init(self, run_id):
        tracker = super().init(run_id)
        if jax.process_index() == 0 and self.mode != "disabled":
            run = wandb.run
            if run is None or run.url is None or run.name is None:
                raise RuntimeError("W&B did not expose the initialized run")
            command = [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                "scripts/history.py",
            ]
            if not any(existing.run_id == run.id for existing in _existing_run_files()):
                subprocess.run(
                    command
                    + [
                        "new",
                        "--wandb-url",
                        run.url,
                        "--wandb-name",
                        run.name,
                        "--experiment",
                        "exp279_models_exact_soft_contact_targets",
                        "--kind",
                        "models",
                        "--short",
                        "Exact soft contact targets / matched CE control",
                    ],
                    cwd=ROOT,
                    check=True,
                )
            subprocess.run(command + ["update-index"], cwd=ROOT, check=True)
        return tracker


def build_config(
    manifest: dict,
    *,
    arm: str,
    phase_name: str,
    run_name: str,
    output: str,
    resume: str | None,
    per_device_batch: int = 1,
    model_seed: int = 0,
    stop_after: int | None = None,
) -> TrainLmConfig:
    """Resolve every arm through the same model/data/trainer configuration."""
    phase = PHASES[phase_name]
    if arm not in ("ce", "soft"):
        raise ValueError("arm must be ce or soft")
    if phase.start and resume is None:
        raise ValueError(
            "Continuation phases require an explicit full-state checkpoint"
        )
    if not run_name.startswith(f"exp279-{arm}-"):
        raise ValueError(f"Run name must begin exp279-{arm}-")
    start = 0
    if resume:
        start = (
            json.loads(StoragePath(resume.rstrip("/") + "/metadata.json").read_text())[
                "step"
            ]
            + 1
        )
        if not phase.start <= start < phase.stop:
            raise ValueError("Resume checkpoint is outside the selected recipe phase")
    model = reference_model_config(soft_targets=arm == "soft")
    stop = phase.stop if stop_after is None else stop_after
    if not start < stop <= phase.stop:
        raise ValueError(
            "Stop must follow the restored update and remain in this phase"
        )
    components: dict[str, DatasetComponentBase] = {
        name: DatasetComponent(
            cache_dir=entry["cache_dir"],
            flat_cache=True,
            split="validation" if name == "val" else "train",
            pack=True,
            tags=[name],
            format=TextLmDatasetFormat(text_key="document"),
        )
        for name, entry in manifest["inputs"].items()
    }
    data = ContactDataConfig(
        tokenizer=TOKENIZER,
        auto_build_caches=False,
        components=components,
        train_weights={
            "afdb": AFDB_TOKENS / (AFDB_TOKENS + ESM_TOKENS),
            "esm": ESM_TOKENS / (AFDB_TOKENS + ESM_TOKENS),
        },
        shuffle=SHUFFLE,
        block_cross_document_attention=True,
    )
    trainer = TrainerConfig(
        seed=model_seed,
        id=run_name,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=128,
        per_device_parallelism=per_device_batch,
        per_device_eval_parallelism=per_device_batch,
        num_train_steps=stop,
        steps_per_eval=2114,
        watch=WatchConfig(watch_targets=[], interval=0),
        tracker=HistoryWandbConfig(
            entity="open-athena",
            project="MarinFold",
            name=run_name,
            group="exp279",
            tags=["exp279", arm],
        ),
        checkpointer=CheckpointerConfig(
            base_path=output.rstrip("/") + f"/checkpoints/{run_name}",
            append_run_id_to_base_path=False,
            keep=[{"every": 14520}],
        ),
        load_checkpoint=resume is not None,
        load_checkpoint_path=resume,
        # Reference transition: initialize only the five new
        # SkipStep buffers, restore every existing state leaf.
        allow_partial_checkpoint=phase_name == "recovery" and start == phase.start,
    )
    return TrainLmConfig(
        data=data,
        model=model,
        trainer=trainer,
        optimizer=optimizer_for_phase(phase),
        train_seq_len=8192,
        data_seed=phase.data_seed,
        z_loss_weight=0.0,
        hf_save_path=output.rstrip("/") + f"/checkpoints/{run_name}/hf",
        hf_save_steps=14520,
        hf_save_dtype="bfloat16",
        hf_generation_eos_token_ids=[1, 10],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("ce", "soft"), required=True)
    parser.add_argument("--phase", choices=PHASES, default="base")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--output", required=True, help="Storage prefix in the compute region"
    )
    parser.add_argument(
        "--resume",
        help="Exact native step-N checkpoint, including optimizer/RNG/data position",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume this run's latest committed checkpoint and select its recipe phase",
    )
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Absolute update count for a resumable production pilot; preserves the LR schedule",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Paired initialization seed; data seeds remain the reference's",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run training; default prints the resolved configuration",
    )
    args = parser.parse_args()
    if args.resume_latest and (args.resume is not None or args.phase != "base"):
        parser.error("--resume-latest selects both checkpoint and phase")
    manifest = json.loads(args.manifest.read_text())
    config = build_config(
        manifest,
        arm=args.arm,
        phase_name=args.phase,
        run_name=args.run_name,
        output=args.output,
        resume=args.resume,
        per_device_batch=args.per_device_batch,
        model_seed=args.seed,
        stop_after=args.stop_after,
    )
    if not args.run:
        print(draccus.dump(config))
        return
    verify_manifest(manifest)
    # Initialize distributed JAX before any backend-dependent preflight (even
    # process_index() initializes a backend). The stock entry point still owns
    # tracker/trainer setup, and skips only the already-completed distributed
    # initialization. This matters on real multi-host jobs, not local smokes.
    for name, value in config.trainer.jax_config.items():
        jax.config.update(name, value)
    if config.trainer.jax_compilation_cache_dir is not None:
        jax.config.update(
            "jax_compilation_cache_dir", config.trainer.jax_compilation_cache_dir
        )
    config.trainer.distributed.initialize()
    if args.resume_latest:
        # A single leader chooses the committed checkpoint so ranks cannot race
        # checkpoint publication/listing and restore different training states.
        checkpoint = None
        if jax.process_index() == 0:
            checkpoint = discover_latest_checkpoint(
                args.output.rstrip("/") + f"/checkpoints/{args.run_name}"
            )
        args.resume = multihost_broadcast_sync(checkpoint)
        update = 0
        if args.resume is not None:
            update = (
                json.loads(StoragePath(args.resume + "/metadata.json").read_text())[
                    "step"
                ]
                + 1
            )
        selected_phase = phase_for_update(update)
        if selected_phase is None or (
            args.stop_after is not None and update >= args.stop_after
        ):
            print(
                f"Requested training complete at update {update}; checkpoint={args.resume}",
                flush=True,
            )
            return
        args.phase = selected_phase
        config = build_config(
            manifest,
            arm=args.arm,
            phase_name=args.phase,
            run_name=args.run_name,
            output=args.output,
            resume=args.resume,
            per_device_batch=args.per_device_batch,
            model_seed=args.seed,
            stop_after=args.stop_after,
        )
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            distributed=replace(
                config.trainer.distributed, initialize_jax_distributed=False
            ),
        ),
    )
    # Persist enough to distinguish either arm and prevent an accidental
    # cross-arm/full-state resume. This file lives alongside the checkpoints.
    identity = {
        "manifest": manifest,
        "arm": args.arm,
        "run_name": args.run_name,
        "model_seed": args.seed,
        "per_device_batch": args.per_device_batch,
        "device_count": jax.device_count(),
        "microbatch_size": config.trainer.microbatch_size,
    }
    record = StoragePath(
        args.output.rstrip("/") + f"/checkpoints/{args.run_name}/experiment.json"
    )
    if args.resume is None:
        # Only the leader checks; broadcast before it writes. Otherwise a slow
        # rank could see the leader's newly created record as a pre-existing run.
        exists = record.exists() if jax.process_index() == 0 else False
        exists = bool(multihost_utils.broadcast_one_to_all(np.asarray(exists)))
        if exists:
            if not args.resume_latest or json.loads(record.read_text()) != identity:
                raise ValueError(
                    "Run identity already exists; use a matching explicit resume or a new run name"
                )
            print(
                "Restarting the same pinned run before its first completed checkpoint",
                flush=True,
            )
    if args.resume:
        source_record = StoragePath(
            args.resume.rstrip("/").rsplit("/", 1)[0] + "/experiment.json"
        )
        if json.loads(source_record.read_text()) != identity:
            raise ValueError("Resume checkpoint has a different experiment identity")
        validate_training_restore(config, args.resume)
    if jax.process_index() == 0:
        record.parent.mkdirs(exist_ok=True)
        record.write_text(json.dumps(identity, indent=2) + "\n")
        print(
            json.dumps(
                {"experiment": identity, "phase": args.phase, "resume": args.resume},
                indent=2,
            ),
            flush=True,
        )
    train_main(config)


if __name__ == "__main__":
    main()
