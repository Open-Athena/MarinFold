"""Bundle the reviewed evaluation and submit a checkpoint-local CPU driver."""

import argparse
import configparser
import hashlib
import json
import os
import shlex
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
    sources = [
        HERE / name
        for name in (
            "eval_run.py",
            "eval_export.py",
            "eval_local.py",
            "eval_worker.py",
            "eval_generation.py",
        )
    ]
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


def local_entrypoint(
    trial: str, eval_set: str = "all", *, export_only: bool = False
) -> list[str]:
    """Bootstrap the validated vLLM image and isolated CPU export environment."""
    if export_only:
        return [
            "bash",
            "-lc",
            "\n".join(
                [
                    "set -euo pipefail",
                    "python3 -m pip install --quiet uv==0.9.18",
                    "exec uv run --frozen python -m experiments.exp364_models_mpnn_only_lr_sweep.eval_export --plan experiments/exp364_models_mpnn_only_lr_sweep/data/contact_eval/plan.json --trial "
                    + shlex.quote(trial),
                ]
            ),
        ]
    package = "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@d1bea417a64cc042ad931422200c3edeb873f2e0#subdirectory=marinfold"
    shell = "\n".join(
        [
            "set -euo pipefail",
            "VLLM_PY=''",
            'for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do if "$candidate" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$candidate"; break; fi; done',
            'test -n "$VLLM_PY"',
            'export VLLM_PORT=$("$VLLM_PY" -c \'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()\')',
            '"$VLLM_PY" -m pip install --quiet "uv==0.9.18" "fsspec==2026.1.0" "s3fs==2026.1.0" "aiobotocore==2.26.0" "pyarrow>=23,<24"',
            '"$VLLM_PY" -m pip install --quiet --no-deps ' + shlex.quote(package),
            "exec uv run --frozen --with scikit-learn==1.7.2 python -m experiments.exp364_models_mpnn_only_lr_sweep.eval_local --trial "
            + shlex.quote(trial)
            + " --eval-set "
            + shlex.quote(eval_set)
            + ' --vllm-python "$VLLM_PY"',
        ]
    )
    return ["bash", "-lc", shell]


def main() -> None:
    """Dry-run by default; --submit is reserved for the chosen execution plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--local-scratch", action="store_true")
    parser.add_argument(
        "--trial", choices=("lr5e-4", "lr1e-3", "lr2e-3", "lr5e-4-cooldown")
    )
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
    if args.local_scratch:
        if not args.trial:
            parser.error("--local-scratch requires --trial")
        entry = local_entrypoint(args.trial)
    command = [
        args.iris_bin,
        "--cluster",
        "cw-us-east-02a" if args.local_scratch else "marin",
        "job",
        "run",
        "--user",
        "bizon",
        "--priority",
        "batch",
        "--job-name",
        f"exp364-contact-eval-{args.trial}-a{args.attempt:02d}"
        if args.local_scratch
        else f"exp364-contact-eval-a{args.attempt:02d}",
        "--no-wait",
        "--max-retries",
        "0",
        "--enable-extra-resources",
        "--cpu",
        "8" if args.local_scratch else "4",
        "--memory",
        "64GB",
        "--disk",
        "128GB" if args.local_scratch else "64GB",
    ]
    if args.local_scratch:
        command.extend(
            ["--gpu", "H100x1", "--no-sync", "--task-image", "vllm/vllm-openai:v0.9.2"]
        )
    else:
        command.extend(["--target-cluster", plan["cluster"]])
    print(
        f"Bundle: {bundle}; {'one H100 with local models' if args.local_scratch else 'CPU export driver, up to 12 H100s'}; batch in {plan['cluster']}"
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
