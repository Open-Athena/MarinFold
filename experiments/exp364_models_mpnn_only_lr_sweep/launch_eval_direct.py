"""Submit independent regional evaluation jobs from a verified source bundle."""

import argparse
import configparser
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import iris
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
    parser.add_argument("--bf16-export", action="store_true")
    parser.add_argument(
        "--cluster", choices=("cw-us-east-02a", "cw-rno2a"), default="cw-us-east-02a"
    )
    parser.add_argument("--attempt", type=int, default=3)
    parser.add_argument(
        "--eval-set", choices=("all", "eval-val", "eval-denovo"), default="all"
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        choices=(
            "lr5e-4",
            "lr1e-3",
            "lr2e-3",
            "lr5e-4-cooldown",
            "lr5e-4-cooldown-tpu",
        ),
    )
    args = parser.parse_args()
    if args.bf16_export and args.trials != ["lr5e-4-cooldown-tpu"]:
        parser.error("--bf16-export requires only --trials lr5e-4-cooldown-tpu")
    if args.cluster == "cw-rno2a" and (
        args.trials != ["lr5e-4-cooldown-tpu"] or args.export_only or args.bf16_export
    ):
        parser.error("RNO placement is only for the bounded TPU evaluation mirror")
    plan = json.loads((HERE / "data/contact_eval/plan.json").read_text())
    if args.cluster == "cw-rno2a":
        checkpoint = next(
            c for c in plan["checkpoints"] if c["trial"] == args.trials[0]
        )
        if (
            checkpoint.get("evaluation_cluster") != args.cluster
            or checkpoint["model_egress_bytes"] >= 10_000_000_000
        ):
            raise ValueError(
                "The RNO mirror is not verified within its transfer budget"
            )
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
        "CW_KEY_ID": cw["aws_access_key_id"],
        "CW_KEY_SECRET": cw["aws_secret_access_key"],
        "AWS_ACCESS_KEY_ID": cw["aws_access_key_id"],
        "AWS_SECRET_ACCESS_KEY": cw["aws_secret_access_key"],
        "FSSPEC_S3": json.dumps(
            {
                "key": cw["aws_access_key_id"],
                "secret": cw["aws_secret_access_key"],
                "endpoint_url": "https://cwobject.com"
                if args.cluster == "cw-rno2a"
                else "http://cwlota.com",
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
    # Use the config shipped with this pinned client, avoiding incompatible
    # workstation overrides written by a newer Iris release.
    config_file = Path(iris.__file__).parent / "config" / f"{args.cluster}.yaml"
    with open_iris_client(config_file=config_file, workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for checkpoint in plan["checkpoints"]:
            if args.trials and checkpoint["trial"] not in args.trials:
                continue
            shell = "\n".join(
                [
                    "set -euo pipefail",
                    "python3 -c " + shlex.quote(download),
                    "cd " + shlex.quote(workdir),
                    (
                        "python3 -m pip install --quiet fsspec==2026.1.0 s3fs==2026.1.0 aiobotocore==2.26.0\nexec python3 -m experiments.exp364_models_mpnn_only_lr_sweep.export_eval_bf16"
                        if args.bf16_export
                        else local_entrypoint(
                            checkpoint["trial"],
                            args.eval_set,
                            export_only=args.export_only,
                        )[-1]
                    ),
                ]
            )
            image = (
                "python:3.12-bookworm"
                if args.export_only and not args.bf16_export
                else "vllm/vllm-openai:v0.9.2"
            )
            resources = (
                ResourceConfig(cpu=8, ram="64Gi", disk="128Gi", image=image)
                if args.export_only or args.bf16_export
                else ResourceConfig.with_gpu(
                    "H100", count=1, cpu=8, ram="64Gi", disk="128Gi", image=image
                )
            )
            job = client.submit(
                JobRequest(
                    name=f"exp364-contact-{'bf16-export' if args.bf16_export else 'export' if args.export_only else 'eval'}-{checkpoint['trial']}{'-' + args.eval_set if args.eval_set != 'all' else ''}-a{args.attempt:02d}",
                    entrypoint=Entrypoint.from_binary("bash", ["-lc", shell]),
                    resources=resources,
                    environment=create_environment(
                        docker_image=image,
                        env_vars=env,
                        setup_scripts=[],
                    ),
                    priority=3,
                    max_retries_failure=0,
                    max_retries_preemption=0 if args.cluster == "cw-rno2a" else 10,
                )
            )
            launches.append(
                {
                    "job_id": job.job_id,
                    "trial": checkpoint["trial"],
                    "step": checkpoint["step"],
                    "eval_set": args.eval_set,
                    "export_only": args.export_only,
                    "bf16_export": args.bf16_export,
                    "cluster": args.cluster,
                    "bundle_sha256": digest,
                    "bundle_uri": destination,
                    "git_commit": env["GIT_COMMIT"],
                }
            )
            record.write_text(json.dumps(launches, indent=2) + "\n")
            print("SUBMITTED", job.job_id, flush=True)


if __name__ == "__main__":
    main()
