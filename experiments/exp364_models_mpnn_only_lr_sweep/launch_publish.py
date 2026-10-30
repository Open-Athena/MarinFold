"""Submit the approved 5.89 GB checkpoint publication on a regional CPU worker."""

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

import iris
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    """Submit a single batch CPU job with no automatic transfer retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    job_name = f"exp364-publish-gpu-cooldown-step186884-a{args.attempt:02d}"
    with tempfile.TemporaryDirectory(prefix="exp364-publication-") as temporary:
        directory = Path(temporary)
        bundle = directory / "source"
        for name in (
            "publish_checkpoint.py",
            "publication_model_card.md",
            "data/publication_source.json",
        ):
            destination = bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, destination)
        archive = Path(shutil.make_archive(str(directory / "source"), "zip", bundle))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        remote = (
            "hf://buckets/open-athena/MarinFold/data/exp364-models-mpnn-only-lr-sweep/publication/source/"
            + digest
            + ".zip"
        )
        print(
            "Publication: one batch CPU job in cw-us-east-02a; six unchanged model/tokenizer files, 5.89 GB",
            flush=True,
        )
        if not args.submit:
            print("DRY_RUN", job_name, "source_sha256", digest)
            return
        subprocess.run(["hf", "buckets", "cp", str(archive), remote], check=True)
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    env = {
        "CW_KEY_ID": cw["aws_access_key_id"],
        "CW_KEY_SECRET": cw["aws_secret_access_key"],
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
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "PUBLICATION_JOB": "/bizon/" + job_name,
    }
    url = remote.replace(
        "hf://buckets/open-athena/MarinFold/",
        "https://huggingface.co/buckets/open-athena/MarinFold/resolve/",
    )
    workdir = "/tmp/exp364-publish-" + digest[:16]
    bootstrap = (
        "import hashlib,io,urllib.request,zipfile; "
        f"payload=urllib.request.urlopen({url!r},timeout=120).read(); "
        f"assert hashlib.sha256(payload).hexdigest()=={digest!r}; "
        f"zipfile.ZipFile(io.BytesIO(payload)).extractall({workdir!r})"
    )
    shell = "\n".join(
        [
            "set -euo pipefail",
            "python3 -c " + shlex.quote(bootstrap),
            "cd " + shlex.quote(workdir),
            "python3 -m pip install --quiet 'fsspec==2026.1.0' 's3fs==2026.1.0' 'aiobotocore==2.26.0' 'huggingface-hub==1.27.0' 'hf-xet==1.6.1a0'",
            "exec python3 publish_checkpoint.py",
        ]
    )
    image = "python:3.12-bookworm"
    config = Path(iris.__file__).parent / "config/cw-us-east-02a.yaml"
    with open_iris_client(config_file=config, workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        job = client.submit(
            JobRequest(
                name=job_name,
                entrypoint=Entrypoint.from_binary("bash", ["-lc", shell]),
                resources=ResourceConfig(cpu=4, ram="16Gi", disk="32Gi", image=image),
                environment=create_environment(
                    docker_image=image, env_vars=env, setup_scripts=[]
                ),
                priority=3,
                max_retries_failure=0,
                max_retries_preemption=0,
            )
        )
        record = {
            "job_id": job.job_id,
            "git_commit": env["GIT_COMMIT"],
            "cluster": "cw-us-east-02a",
            "bundle_uri": remote,
            "bundle_sha256": digest,
            "source_bytes": 5885614887,
        }
        (HERE / "data/publication_launch.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        (HERE / f"data/publication_launch_a{args.attempt:02d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print("SUBMITTED", job.job_id, flush=True)


if __name__ == "__main__":
    main()
