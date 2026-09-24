"""Mirror CoreWeave or public-HF rollout parquets for local analysis."""

import argparse
from pathlib import Path

import fsspec
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
S3_SOURCE = "s3://marin-us-east-02a/MarinFold/exp333/alanine-masking-v1/main"
HF_BASELINE = "hf://buckets/open-athena/MarinFold/data/exp321/null-sequence-guidance-v1"


def source_files(source: str, mode: str, cohort: str) -> tuple[object, list[str]]:
    """Resolve result files from either storage backend."""
    if source.startswith("hf://"):
        filesystem = HfFileSystem(token=False)
        root = source.removeprefix("hf://").rstrip("/")
        prefix = root if mode == "." else f"{root}/{mode}"
        return filesystem, sorted(filesystem.glob(f"{prefix}/{cohort}/*.parquet"))
    filesystem, root = fsspec.core.url_to_fs(
        source,
        endpoint_url="https://cwobject.com",
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    prefix = root if mode == "." else f"{root}/{mode}"
    return filesystem, sorted(filesystem.glob(f"{prefix}/{cohort}/*.parquet"))


def main() -> None:
    """Download one or more complete modes without altering source storage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=S3_SOURCE)
    parser.add_argument("--modes", required=True, help="comma-separated mode names")
    parser.add_argument("--cohort", default="eval-val")
    parser.add_argument("--destination", type=Path, default=HERE / "_cache")
    args = parser.parse_args()
    for mode in [value.strip() for value in args.modes.split(",") if value.strip()]:
        filesystem, paths = source_files(args.source, mode, args.cohort)
        if not paths:
            raise FileNotFoundError(f"{args.source}/{mode}/{args.cohort}")
        destination = args.destination / args.cohort if mode == "." else args.destination / mode / args.cohort
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            filesystem.get_file(path, str(destination / Path(path).name))
        print(f"[exp333] mirrored {len(paths)} {mode} files into {destination}")


if __name__ == "__main__":
    main()
