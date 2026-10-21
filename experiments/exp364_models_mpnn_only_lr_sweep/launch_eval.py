"""Bundle the reviewed evaluation and submit a checkpoint-local CPU driver."""

import argparse
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def build_bundle() -> Path:
    """Package only the evaluation sources, locked runtime, and selected inputs."""
    bundle = Path(tempfile.mkdtemp(prefix="exp364-contact-eval-"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(HERE / name, bundle / name)
    sources = [HERE / name for name in ("eval_run.py", "eval_export.py")]
    sources += [
        HERE / "data/contact_eval" / name
        for name in (
            "plan.json",
            "targets.csv",
            "targets.parquet",
            "ground_truth.jsonl",
        )
    ]
    sources += [
        REPO / "experiments" / name
        for name in (
            "exp232_sweep_cv1_decontam/training_contract.py",
            "exp277_models_single_mpnn_pilot/config.py",
            "exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2/score_rollout_worker.py",
            "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py",
        )
    ]
    for source in sources:
        destination = bundle / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = {
        str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    }
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return bundle


def main() -> None:
    """Dry-run by default; --submit is reserved for the chosen execution plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    args = parser.parse_args()
    plan = json.loads((HERE / "data/contact_eval/plan.json").read_text())
    if plan["cluster"] != "cw-us-east-02a" or plan["gpu_limit"] != 12:
        raise ValueError("Execution placement or GPU limit changed")
    bundle = build_bundle()
    entry = [
        "uv",
        "run",
        "--frozen",
        "--with",
        "scikit-learn==1.7.2",
        "python",
        "-m",
        "experiments.exp364_models_mpnn_only_lr_sweep.eval_run",
    ]
    command = [
        args.iris_bin,
        "--cluster",
        "marin",
        "job",
        "run",
        "--user",
        "bizon",
        "--target-cluster",
        plan["cluster"],
        "--priority",
        "batch",
        "--job-name",
        f"exp364-contact-eval-a{args.attempt:02d}",
        "--no-wait",
        "--max-retries",
        "0",
        "--enable-extra-resources",
        "--cpu",
        "4",
        "--memory",
        "64GB",
        "--disk",
        "64GB",
    ]
    print(
        f"Bundle: {bundle}; CPU export driver; up to 12 batch H100s in {plan['cluster']}"
    )
    if not args.submit:
        print(f"Dry run only. Entrypoint: {entry}")
        return
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    env = {
        "FSSPEC_S3": json.dumps(
            {
                "key": cw["aws_access_key_id"],
                "secret": cw["aws_secret_access_key"],
                "endpoint_url": "http://cwlota.com",
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            }
        ),
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    token = Path.home() / ".cache/huggingface/token"
    if token.exists():
        env["HF_TOKEN"] = token.read_text().strip()
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend(["--", *entry])
    subprocess.run(command, cwd=bundle, check=True)


if __name__ == "__main__":
    main()
