# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""List candidate CoreWeave S3 prefixes for exp156 data discovery.

Run this inside a CoreWeave Iris task, where AWS_* / FSSPEC_S3 credentials are
injected by the cluster. It intentionally lists only shallow prefixes by default.
"""

import argparse
import os
from collections.abc import Iterable
from urllib.parse import urlparse


def _s3_options() -> dict[str, object]:
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL")
    options: dict[str, object] = {}
    if endpoint_url:
        options["client_kwargs"] = {"endpoint_url": endpoint_url}
    return options


def _list_with_s3fs(prefixes: Iterable[str]) -> None:
    import s3fs

    fs = s3fs.S3FileSystem(anon=False, **_s3_options())
    for prefix in prefixes:
        print(f"\n## {prefix}")
        try:
            entries = fs.ls(prefix, detail=True)
        except FileNotFoundError:
            print("missing")
            continue
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            continue
        for entry in entries:
            name = entry.get("name", str(entry)) if isinstance(entry, dict) else str(entry)
            size = entry.get("size") if isinstance(entry, dict) else None
            kind = entry.get("type") if isinstance(entry, dict) else None
            suffix = "/" if kind == "directory" and not name.endswith("/") else ""
            size_text = "" if size is None else f"\t{size}"
            print(f"{name}{suffix}{size_text}")


def _parents(path: str) -> list[str]:
    parsed = urlparse(path)
    parts = [part for part in parsed.path.split("/") if part]
    out = [f"s3://{parsed.netloc}"]
    for i in range(1, len(parts) + 1):
        out.append(f"s3://{parsed.netloc}/{'/'.join(parts[:i])}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prefix",
        nargs="*",
        default=[
            "s3://marin-us-east-02a",
            "s3://marin-us-east-02a/marin",
            "s3://marin-us-east-02a/MarinFold",
            "s3://marin-us-east-02a/MarinFold/data",
            "s3://marin-us-east-02a/MarinFold/data/document_structures",
            "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1",
            "s3://marin-us-east-02a/marin/data",
            "s3://marin-us-east-02a/marin/checkpoints",
        ],
    )
    parser.add_argument(
        "--parents-of",
        help="Also list every parent of this full s3:// path.",
    )
    args = parser.parse_args()
    prefixes = list(args.prefix)
    if args.parents_of:
        prefixes.extend(_parents(args.parents_of))
    print("AWS_ACCESS_KEY_ID", "set" if os.environ.get("AWS_ACCESS_KEY_ID") else "unset")
    print("AWS_ENDPOINT_URL", os.environ.get("AWS_ENDPOINT_URL", "unset"))
    print("FSSPEC_S3", os.environ.get("FSSPEC_S3", "unset"))
    _list_with_s3fs(prefixes)


if __name__ == "__main__":
    main()
