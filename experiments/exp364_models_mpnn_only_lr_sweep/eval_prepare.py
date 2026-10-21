"""Prepare a read-only snapshot and the exact requested contact-evaluation sets."""

import configparser
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import pandas as pd
import wandb

from experiments.exp364_models_mpnn_only_lr_sweep.config import PREFIX, run_id_for
from experiments.exp364_models_mpnn_only_lr_sweep.cooldown_train import RUN_ID

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "contact_eval"
EVAL_ROOT = f"{PREFIX}/evals/rollout-v2/2026-10-21/latest-01"
SOURCE_HASHES = {
    "eval_sets.csv": "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861",
    "eval_targets_foldbench_monomers.parquet": "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8",
    "gt_universe_scored.jsonl": "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5",
}


def source_filesystem() -> fsspec.AbstractFileSystem:
    """Use the workstation's CoreWeave credentials without printing them."""
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    return fsspec.filesystem(
        "s3",
        key=cw["aws_access_key_id"],
        secret=cw["aws_secret_access_key"],
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
        use_listings_cache=False,
    )


def select_targets(
    targets: pd.DataFrame, sets: pd.DataFrame, truth: list[dict]
) -> tuple[pd.DataFrame, list[dict]]:
    """Require exactly the requested 97 natural and 19 designed monomers."""
    selected = sets[
        (sets.scorable == 1) & sets.eval_set.isin(["eval-val", "eval-denovo"])
    ]
    if selected.eval_set.value_counts().to_dict() != {
        "eval-val": 97,
        "eval-denovo": 19,
    }:
        raise ValueError("The requested evaluation-set membership changed")
    targets = targets.merge(
        selected[["stem", "eval_set", "is_viral"]], on="stem", validate="one_to_one"
    )
    if len(targets) != 116 or (targets.input_seq.str.len() != targets.L).any():
        raise ValueError("Targets do not cover the 116 expected sequences")
    units = set(zip(targets.dataset, targets.stem, strict=True))
    truth = [record for record in truth if (record["dataset"], record["stem"]) in units]
    if len(truth) != 116 or {(r["dataset"], r["stem"]) for r in truth} != units:
        raise ValueError("Ground truth and targets differ")
    if any(r["L"] != int(targets.set_index("stem").loc[r["stem"], "L"]) for r in truth):
        raise ValueError("Ground truth and target lengths differ")
    return targets, truth


def main() -> None:
    """Pin source inventories without copying or submitting anything."""
    DATA.mkdir(parents=True, exist_ok=True)
    source = HERE.parent / "exp245_evals_foldbench_held_out_monomers" / "data"
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published input digest mismatch: {name}")
    targets, truth = select_targets(
        pd.read_parquet(source / "eval_targets_foldbench_monomers.parquet"),
        pd.read_csv(source / "eval_sets.csv"),
        [
            json.loads(line)
            for line in (source / "gt_universe_scored.jsonl").read_text().splitlines()
        ],
    )
    targets.to_parquet(DATA / "targets.parquet", index=False)
    (DATA / "ground_truth.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in truth)
    )
    targets[["dataset", "stem", "L", "eval_set", "is_viral"]].to_csv(
        DATA / "targets.csv", index=False
    )

    fs = source_filesystem()
    api = wandb.Api()
    checkpoints = []
    for trial, run_id in [
        ("lr5e-4", run_id_for("lr5e-4")),
        ("lr1e-3", run_id_for("lr1e-3")),
        ("lr2e-3", run_id_for("lr2e-3")),
        ("lr5e-4-cooldown", RUN_ID),
    ]:
        run = api.run(f"open-athena/MarinFold/{run_id}")
        config = run.config["trainer"]["checkpointer"]
        candidates = []
        for kind in ("base_path", "temporary_base_path"):
            root = config[kind]
            if not fs.exists(root):
                continue
            for path in fs.ls(root, detail=False):
                if not path.rsplit("/", 1)[-1].startswith("step-") or not fs.exists(
                    path + "/metadata.json"
                ):
                    continue
                with fs.open(path + "/metadata.json") as handle:
                    metadata = json.load(handle)
                candidates.append((int(metadata["step"]), "s3://" + path))
        step, source_uri = max(candidates)
        root = fs._strip_protocol(source_uri).rstrip("/") + "/"
        objects = [
            {
                "path": path.removeprefix(root),
                "size": info["size"],
                "etag": info["ETag"],
            }
            for path, info in sorted(fs.find(source_uri, detail=True).items())
            if info["type"] == "file"
        ]
        checkpoints.append(
            {
                "trial": trial,
                "run_id": run_id,
                "step": step,
                "label": f"{run_id}-step-{step}",
                "source": source_uri,
                "frozen": f"{EVAL_ROOT}/sources/{run_id}/step-{step}",
                "hf_export": f"{EVAL_ROOT}/models/{run_id}/hf/step-{step}",
                "objects": objects,
                "source_bytes": sum(obj["size"] for obj in objects),
                "wandb_url": run.url,
                "model_config": run.config["model"],
                "latest_logged_step": run.summary.get("global_step"),
                "latest_eval_loss": run.summary.get("eval/loss"),
            }
        )
        print(trial, step, len(objects), checkpoints[-1]["source_bytes"])
    plan = {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "root": EVAL_ROOT,
        "cluster": "cw-us-east-02a",
        "priority": "batch",
        "gpu_limit": 12,
        "checkpoints": checkpoints,
        "source_hashes": SOURCE_HASHES,
        "inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in DATA.iterdir()
            if p.name in {"targets.parquet", "targets.csv", "ground_truth.jsonl"}
        },
        "sets": {"eval-val": 97, "eval-denovo": 19},
        "sampling": {
            "rollouts": 100,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "token_budget": "6L+128",
            "seed": 0,
        },
        "local_option": "Freeze and export each checkpoint in the same CoreWeave bucket; score there on at most 12 H100s. No model egress.",
        "mirror_option": "Export four float32 HF models and mirror about 23.54 GB to the public HF bucket for reuse elsewhere; additional egress and >10 GB approval required.",
    }
    (DATA / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
