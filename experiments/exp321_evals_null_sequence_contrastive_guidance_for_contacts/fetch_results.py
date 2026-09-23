"""Mirror one small CoreWeave result prefix for local scoring."""

import argparse
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = "s3://marin-us-east-02a/MarinFold/exp321/null-sequence-guidance-v1"


def main() -> None:
    """Download raw and timing parquets through CoreWeave's public endpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--cohort", choices=["eval-val", "foldswitch"], required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--endpoint-url", default="https://cwobject.com")
    args = parser.parse_args()
    uri = f"{args.source.rstrip('/')}/{args.mode}/{args.cohort}"
    filesystem, root = fsspec.core.url_to_fs(
        uri,
        endpoint_url=args.endpoint_url,
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    paths = sorted(filesystem.glob(f"{root}/*.parquet"))
    if not paths:
        raise FileNotFoundError(uri)
    destination = args.destination or HERE / "_cache" / args.mode / args.cohort
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        filesystem.get_file(path, str(destination / Path(path).name))
    print(f"mirrored {len(paths)} parquet files into {destination}")


if __name__ == "__main__":
    main()
