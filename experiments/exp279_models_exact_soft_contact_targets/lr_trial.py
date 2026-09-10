"""Fork one exact full-state checkpoint into a bounded learning-rate trial."""

import argparse
import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
from levanter.checkpoint import discover_latest_checkpoint
from levanter.main.train_lm import main as train_main
from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync
from rigging.filesystem.storage_path import StoragePath

from .checkpoints import validate_training_restore
from .inputs import ROOT, resolve_tokenizer, verify_manifest
from .lr_metrics import LrWatchConfig
from .model import ContactQwen3Config
from .recipe import RecipeAdamConfig
from .train import HistoryWandbConfig, build_config

PARENT_RUN = "exp279-soft-s0-cw-h100x32-b02"
PARENT_REVISION = "b0dd33eca8836c1dddbe6588e4befc3b2dd5c67d"
START = 14521
UPDATES = 5000
STOP = START + UPDATES
RATES = {"lr100": 0.001, "lr150": 0.0015, "lr200": 0.002}


@OptimizerConfig.register_subclass("exp279_continuation_adam")
@dataclass(frozen=True)
class ContinuationAdamConfig(RecipeAdamConfig):
    """Keep the exact Adam state structure while replacing its LR schedule."""

    def lr_scheduler(self, num_train_steps, override_lr=None):
        if override_lr is not None:
            raise ValueError("The trial learning rate is explicit")

        def constant(count):
            return jnp.asarray(self.learning_rate, dtype=jnp.float32)

        return constant


def check_parent_code(parent_source: dict, paths: list[str]) -> None:
    """Require every original runtime file to match the parent's frozen hash."""
    code = b""
    for relative in paths:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".py":
            raise ValueError("Invalid parent source path")
        code += relative.encode() + b"\0" + path.read_bytes() + b"\0"
    if hashlib.sha256(code).hexdigest() != parent_source["code_sha256"]:
        raise ValueError("An original model/data/training runtime file changed")


def validate_parent(contract: dict, manifest: dict) -> None:
    """Check lineage, input identity and the immutable full-state starting point."""
    parent = contract["parent_identity"]
    expected = {
        "run_name": PARENT_RUN,
        "arm": "soft",
        "model_seed": 0,
        "per_device_batch": 1,
        "device_count": 32,
        "microbatch_size": 32,
    }
    if any(parent.get(key) != value for key, value in expected.items()):
        raise ValueError("Parent run does not match the authorized continuation")
    source = parent["manifest"]["source"]
    if source["git_sha"] != PARENT_REVISION:
        raise ValueError("Unexpected parent runtime revision")
    for key in ("runtime_packages", "uv_lock_sha256", "tokenizer"):
        if source[key] != manifest["source"][key]:
            raise ValueError(f"Parent dependency identity changed: {key}")
    if manifest["inputs"] != parent["manifest"]["inputs"]:
        raise ValueError("Fork inputs differ from the parent")
    check_parent_code(source, contract["parent_source_paths"])
    checkpoint = contract["parent_checkpoint"]
    if checkpoint.rstrip("/").rsplit("/", 2)[-2:] != [PARENT_RUN, f"step-{START - 1}"]:
        raise ValueError("Unexpected parent checkpoint")
    metadata = json.loads(StoragePath(checkpoint + "/metadata.json").read_text())
    if metadata["step"] != START - 1 or metadata["is_temporary"]:
        raise ValueError("Parent checkpoint must be permanent at the selected step")
    for name in ("metadata.json", "manifest.json"):
        digest = hashlib.sha256(
            StoragePath(checkpoint + "/" + name).read_bytes()
        ).hexdigest()
        if digest != contract["checkpoint_hashes"][name]:
            raise ValueError(f"Parent checkpoint metadata changed: {name}")


def trial_identity(manifest: dict, contract: dict, trial: str, run_name: str) -> dict:
    """Persist all scientifically relevant fork choices, including its lineage."""
    return {
        "manifest": manifest,
        "parent": contract,
        "trial": trial,
        "learning_rate": RATES[trial],
        "run_name": run_name,
        "start_update": START,
        "stop_update": STOP,
        "device_count": 32,
        "per_device_batch": 1,
        "microbatch_size": 32,
    }


def select_resume(root: str, identity: dict, parent_checkpoint: str) -> tuple[str, int]:
    """Resume only this exact fork; use the parent only before any fork save."""
    record = StoragePath(root + "/experiment.json")
    if record.exists() and json.loads(record.read_text()) != identity:
        raise ValueError("Existing trial identity differs")
    checkpoint = discover_latest_checkpoint(root)
    if checkpoint is None:
        return parent_checkpoint, START
    if not record.exists():
        raise ValueError("Trial checkpoint has no lineage record")
    update = (
        json.loads(StoragePath(checkpoint + "/metadata.json").read_text())["step"] + 1
    )
    if not START <= update <= STOP:
        raise ValueError("Trial checkpoint is outside the continuation window")
    return checkpoint, update


def build_trial_config(
    manifest: dict,
    *,
    trial: str,
    run_name: str,
    output: str,
    resume: str,
    stop_after: int = STOP,
):
    """Change only LR, identity, bounded duration and diagnostic callbacks."""
    if trial not in RATES or not run_name.startswith(f"exp279-soft-{trial}-"):
        raise ValueError("Run identity must match the declared LR trial")
    if not START < stop_after <= STOP:
        raise ValueError("Trial stop is outside the fixed continuation window")
    config = build_config(
        manifest,
        arm="soft",
        phase_name="base",
        run_name=run_name,
        output=output,
        resume=resume,
        stop_after=stop_after,
    )
    parameters: dict[str, Any] = {
        field.name: getattr(config.optimizer, field.name)
        for field in fields(config.optimizer)
    }
    parameters["learning_rate"] = RATES[trial]
    optimizer = ContinuationAdamConfig(**parameters)
    root = output.rstrip("/") + f"/checkpoints/{run_name}"
    trainer = replace(
        config.trainer,
        steps_per_eval=1000,
        watch=LrWatchConfig(
            start=START,
            stop=STOP,
            final_update=stop_after,
            validation_cache=manifest["inputs"]["val"]["cache_dir"],
            tokenizer=config.data.tokenizer,
            include_per_parameter_norms=False,
            split_scan_layers=False,
            checkpoint_root=root + "/contact_diagnostics",
        ),
        tracker=replace(
            cast(HistoryWandbConfig, config.trainer.tracker),
            group="exp279-lr-continuations",
            tags=["exp279", "soft", trial, "H100", "nodes=4"],
        ),
        checkpointer=replace(config.trainer.checkpointer, keep=[{"every": 1000}]),
    )
    return replace(
        config, trainer=trainer, optimizer=optimizer, hf_save_steps=stop_after + 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trial", choices=RATES, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-after", type=int, default=STOP)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    verify_manifest(manifest)
    validate_parent(contract, manifest)
    identity = trial_identity(manifest, contract, args.trial, args.run_name)
    root = args.output.rstrip("/") + f"/checkpoints/{args.run_name}"
    config = build_trial_config(
        manifest,
        trial=args.trial,
        run_name=args.run_name,
        output=args.output,
        resume=contract["parent_checkpoint"],
        stop_after=args.stop_after,
    )
    for name, value in config.trainer.jax_config.items():
        jax.config.update(name, value)
    config.trainer.distributed.initialize()
    if jax.device_count() != 32 or config.trainer.microbatch_size != 32:
        raise ValueError("LR trials require the same 32-GPU / microbatch-32 placement")
    selected = (
        select_resume(root, identity, contract["parent_checkpoint"])
        if jax.process_index() == 0
        else None
    )
    resume, update = cast(tuple[str, int], multihost_broadcast_sync(selected))
    if update >= args.stop_after:
        print(
            f"Requested trial window complete at update {update}: {resume}", flush=True
        )
        return
    config = build_trial_config(
        manifest,
        trial=args.trial,
        run_name=args.run_name,
        output=args.output,
        resume=resume,
        stop_after=args.stop_after,
    )
    tokenizer = resolve_tokenizer()
    config = replace(
        config,
        model=replace(cast(ContactQwen3Config, config.model), tokenizer=tokenizer),
        data=replace(config.data, tokenizer=tokenizer),
        trainer=replace(
            config.trainer,
            watch=replace(
                cast(LrWatchConfig, config.trainer.watch), tokenizer=tokenizer
            ),
            distributed=replace(
                config.trainer.distributed, initialize_jax_distributed=False
            ),
        ),
    )
    validate_training_restore(config, resume)
    if jax.process_index() == 0:
        StoragePath(root).mkdirs(exist_ok=True)
        StoragePath(root + "/experiment.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "trial": args.trial,
                    "lr": RATES[args.trial],
                    "resume": resume,
                    "update": update,
                    "stop": args.stop_after,
                }
            ),
            flush=True,
        )
    train_main(config)


if __name__ == "__main__":
    main()
