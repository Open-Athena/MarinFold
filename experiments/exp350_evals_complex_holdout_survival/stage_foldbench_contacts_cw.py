"""Stage the small frozen target parquet beside the CoreWeave checkpoint."""

import argparse
import hashlib
import os
from pathlib import Path

import s3fs

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data/foldbench_complex_eval_targets.parquet"
DESTINATION = (
    "s3://marin-us-east-02a/MarinFold/"
    "exp350_evals_complex_holdout_survival/foldbench-pair-holdout-v1/"
    "inputs/eval_targets.parquet"
)


def digest(data: bytes) -> str:
    """Return a SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def coreweave_filesystem() -> s3fs.S3FileSystem:
    """Return an S3 filesystem configured for CoreWeave object storage."""
    return s3fs.S3FileSystem(
        key=os.environ["CW_KEY_ID"],
        secret=os.environ["CW_KEY_SECRET"],
        client_kwargs={"endpoint_url": "https://cwobject.com"},
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )


def main() -> None:
    """Upload and byte-verify the immutable evaluation target table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", default=DESTINATION)
    args = parser.parse_args()
    if not os.environ.get("CW_KEY_ID") or not os.environ.get("CW_KEY_SECRET"):
        raise ValueError("source the CoreWeave object-storage credentials first")
    if not args.destination.startswith("s3://"):
        raise ValueError("destination must be a CoreWeave s3:// URI")
    payload = args.source.read_bytes()
    filesystem = coreweave_filesystem()
    destination_path = args.destination.removeprefix("s3://")
    with filesystem.open(destination_path, "wb") as handle:
        handle.write(payload)
    with filesystem.open(destination_path, "rb") as handle:
        recovered = handle.read()
    if recovered != payload:
        raise ValueError(
            f"staged bytes differ: {digest(payload)} != {digest(recovered)}"
        )
    print(
        f"staged {len(payload)} bytes sha256={digest(payload)} -> "
        f"{args.destination}"
    )


if __name__ == "__main__":
    main()
