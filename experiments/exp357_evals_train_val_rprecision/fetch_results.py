"""Fetch only compact completed result tables, using workstation S3 access."""

import argparse
import configparser
import json
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
ROOT = "marin-us-east-02a/MarinFold/exp357_evals_train_val_rprecision"


def workstation_filesystem():
    """Use the existing CoreWeave profile without exposing its credentials."""
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    profile = credentials["cw"]
    return fsspec.filesystem(
        "s3",
        key=profile["aws_access_key_id"],
        secret=profile["aws_secret_access_key"],
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    fs = workstation_filesystem()
    root = f"{ROOT}/{args.run_id}"
    if args.status:
        models = json.loads((HERE / "data/checkpoints.json").read_text())
        print(
            json.dumps(
                {
                    c["job_label"]: {
                        phase: len(
                            fs.glob(f"{root}/{phase}/{c['label']}/complete/*.json")
                        )
                        for phase in ("smoke", "rollout")
                    }
                    for c in models
                }
            )
        )
        return
    if not fs.exists(f"{root}/results/complete.json"):
        raise RuntimeError(
            "The evaluation has not completed; refusing a partial headline"
        )
    output = HERE / "data/results"
    output.mkdir(exist_ok=True)
    for name in (
        "complete.json",
        "per_protein.csv",
        "single_samples.csv",
        "timings.csv",
    ):
        destination = (
            HERE / "data/timings.csv" if name == "timings.csv" else output / name
        )
        fs.get_file(f"{root}/results/{name}", str(destination))
        print(name, destination.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
