"""Submit without an Iris workspace write while CoreWeave storage is suspended."""

import argparse
import configparser
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

from experiments.exp364_models_mpnn_only_lr_sweep.launch_eval import (
    HERE,
    REPO,
    build_bundle,
    local_entrypoint,
)


def main() -> None:
    """Publish only source code and small inputs; keep all model bytes regional."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--attempt", type=int, default=3)
    parser.add_argument(
        "--eval-set", choices=("all", "eval-val", "eval-denovo"), default="all"
    )
    parser.add_argument(
        "--trials", nargs="+", choices=("lr5e-4", "lr1e-3", "lr2e-3", "lr5e-4-cooldown")
    )
    args = parser.parse_args()
    plan = json.loads((HERE / "data/contact_eval/plan.json").read_text())
    bundle = build_bundle()
    archive = Path(shutil.make_archive(str(bundle), "zip", bundle))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = plan["artifact_root"] + "/source/" + digest + ".zip"
    prefix = destination.removeprefix("hf://buckets/open-athena/MarinFold/")
    url = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/" + prefix
    print(
        f"Source bundle: {archive.stat().st_size} bytes; SHA-256 {digest}", flush=True
    )
    if not args.submit:
        print(
            f"Dry run: {args.trials or 'all trials'}, set={args.eval_set}, batch CPU export per trial{'' if args.export_only else '; evaluation uses one H100'}, no Iris workspace upload"
        )
        return
    subprocess.run(["hf", "buckets", "cp", str(archive), destination], check=True)
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
        "HF_TOKEN": os.environ.get("HF_TOKEN")
        or (Path.home() / ".cache/huggingface/token").read_text().strip(),
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    workdir = "/tmp/exp364-eval-source-" + digest[:16]
    download = (
        "import hashlib,io,urllib.request,zipfile; "
        f"payload=urllib.request.urlopen({url!r},timeout=120).read(); "
        f"assert hashlib.sha256(payload).hexdigest()=={digest!r}; "
        f"zipfile.ZipFile(io.BytesIO(payload)).extractall({workdir!r})"
    )
    record = HERE / "data/contact_eval/launches.json"
    launches = json.loads(record.read_text()) if record.exists() else []
    with open_iris_client(cluster_name="cw-us-east-02a", workspace=None) as iris:
        client = FrayIrisClient.from_iris_client(iris)
        for checkpoint in plan["checkpoints"]:
            if args.trials and checkpoint["trial"] not in args.trials:
                continue
            shell = "\n".join(
                [
                    "set -euo pipefail",
                    "python3 -c " + shlex.quote(download),
                    "cd " + shlex.quote(workdir),
                    local_entrypoint(
                        checkpoint["trial"], args.eval_set, export_only=args.export_only
                    )[-1],
                ]
            )
            image = (
                "python:3.12-bookworm"
                if args.export_only
                else "vllm/vllm-openai:v0.9.2"
            )
            resources = (
                ResourceConfig(cpu=8, ram="64Gi", disk="128Gi", image=image)
                if args.export_only
                else ResourceConfig.with_gpu(
                    "H100", count=1, cpu=8, ram="64Gi", disk="128Gi", image=image
                )
            )
            job = client.submit(
                JobRequest(
                    name=f"exp364-contact-{'export' if args.export_only else 'eval'}-{checkpoint['trial']}{'-' + args.eval_set if args.eval_set != 'all' else ''}-a{args.attempt:02d}",
                    entrypoint=Entrypoint.from_binary("bash", ["-lc", shell]),
                    resources=resources,
                    environment=create_environment(
                        docker_image=image,
                        env_vars=env,
                        setup_scripts=[],
                    ),
                    priority=3,
                    max_retries_failure=0,
                    max_retries_preemption=10,
                )
            )
            launches.append(
                {
                    "job_id": job.job_id,
                    "trial": checkpoint["trial"],
                    "step": checkpoint["step"],
                    "eval_set": args.eval_set,
                    "export_only": args.export_only,
                    "bundle_sha256": digest,
                    "bundle_uri": destination,
                    "git_commit": env["GIT_COMMIT"],
                }
            )
            record.write_text(json.dumps(launches, indent=2) + "\n")
            print("SUBMITTED", job.job_id, flush=True)


if __name__ == "__main__":
    main()
