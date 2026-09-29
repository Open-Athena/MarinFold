"""Submit the CoreWeave co-located latest-corpus sampling job."""

import configparser
import json
import os
import subprocess
from pathlib import Path


IRIS = Path("/home/bizon/git/marin-freshiris/.venv/bin/iris")
HERE = Path(__file__).resolve().parent


def main() -> None:
    """Pass credentials to Iris without writing or printing them."""
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    env = os.environ.copy()
    env["FSSPEC_S3"] = json.dumps(
        {
            "key": cw["aws_access_key_id"],
            "secret": cw["aws_secret_access_key"],
            "endpoint_url": "http://cwlota.com",
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
        }
    )
    subprocess.run(
        [
            str(IRIS),
            "--cluster=marin",
            "job",
            "run",
            "--target-cluster",
            "cw-us-east-02a",
            "--priority",
            "batch",
            "--job-name",
            "training-explorer-sample-20260922",
            "--no-wait",
            "--enable-extra-resources",
            "--cpu",
            "8",
            "--memory",
            "32GB",
            "--disk",
            "20GB",
            "-e",
            "FSSPEC_S3",
            env["FSSPEC_S3"],
            "--",
            "uv",
            "run",
            "python",
            "sample_latest.py",
        ],
        cwd=HERE,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
