"""Launch 16 disjoint CoreWeave CPU search workers for the MPNN corpus."""

import configparser
import argparse
import json
import os
import subprocess
from pathlib import Path


IRIS = Path("/home/bizon/git/marin-freshiris/.venv/bin/iris")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BUCKET = "marin-us-east-02a"
QUERY_KEY = (
    f"s3://{BUCKET}/MarinFold/training_explorer/2026-09-22/latest_eval_queries.fasta"
)
SHARDS = 16


def main() -> None:
    """Upload small query FASTA and dispatch co-located independent searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unmasked-query-id",
        help="One low-complexity query needing an unmasked full-corpus pass",
    )
    args = parser.parse_args()
    latest = json.loads((DATA / "latest.json").read_text())
    evaluation = json.loads((DATA / "eval.json").read_text())
    if len(latest["proteins"]) != 100 or len(evaluation["proteins"]) != 116:
        raise ValueError("Query set must contain 100 training and 116 eval proteins")
    all_proteins = latest["proteins"] + evaluation["proteins"]
    if args.unmasked_query_id:
        selected = [p for p in all_proteins if p["id"] == args.unmasked_query_id]
        if len(selected) != 1:
            raise ValueError(f"Unknown query {args.unmasked_query_id}")
        query = DATA / "low_complexity_query.fasta"
        query_key = f"s3://{BUCKET}/MarinFold/training_explorer/2026-09-22/low_complexity_query.fasta"
    else:
        selected = all_proteins
        query = DATA / "latest_eval_queries.fasta"
        query_key = QUERY_KEY
    query.write_text("".join(f">{p['id']}\n{p['sequence']}\n" for p in selected))
    subprocess.run(
        [
            "aws",
            "--profile",
            "cw",
            "--endpoint-url",
            "https://cwobject.com",
            "s3",
            "cp",
            str(query),
            query_key,
        ],
        check=True,
    )
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    config = json.dumps(
        {
            "key": cw["aws_access_key_id"],
            "secret": cw["aws_secret_access_key"],
            "endpoint_url": "http://cwlota.com",
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
        }
    )
    env = os.environ.copy()
    env["FSSPEC_S3"] = config
    for index in range(SHARDS):
        command = [
            str(IRIS),
            "--cluster=marin",
            "job",
            "run",
            "--target-cluster",
            "cw-us-east-02a",
            "--priority",
            "batch",
            "--job-name",
            f"training-explorer-{'unmasked' if args.unmasked_query_id else 'search'}-{index:02d}",
            "--no-wait",
            "--enable-extra-resources",
            "--cpu",
            "8",
            "--memory",
            "64GB",
            "--disk",
            "64GB",
            "-e",
            "FSSPEC_S3",
            config,
            "--",
            "uv",
            "run",
            "python",
            "search_mpnn.py",
            "--shard",
            str(index),
            "--shards",
            str(SHARDS),
        ]
        if args.unmasked_query_id:
            command.append("--unmasked")
        result = subprocess.run(
            command,
            cwd=HERE,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Submission {index} failed: {result.stdout[-1000:]}")
        print(f"Submitted search shard {index + 1}/{SHARDS}", flush=True)


if __name__ == "__main__":
    main()
