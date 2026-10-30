"""Use the direct bucket API with bounded retries in an isolated HF environment."""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

T = TypeVar("T")
BUCKET = "open-athena/MarinFold"
PREFIX = "hf://buckets/" + BUCKET + "/"


def retry(operation: Callable[[], T]) -> T:
    """Retry only explicitly transient HTTP responses, then propagate failures."""
    for attempt in range(8):
        try:
            return operation()
        except HfHubHTTPError as error:
            status = error.response.status_code
            if status not in [429, 500, 502, 503, 504] or attempt == 7:
                raise
            delay = min(
                60, max(5, int(error.response.headers.get("retry-after", "30")) + 1)
            )
            print(
                f"Bucket HTTP {status}; retry {attempt + 1}/7 in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    """Publish or test one exact object without repeated filesystem discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exists")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination")
    args = parser.parse_args()
    api = HfApi()
    uri = args.exists or args.destination
    if not uri.startswith(PREFIX):
        raise ValueError("Expected the configured public experiment bucket")
    path = uri.removeprefix(PREFIX)
    if args.exists:
        try:
            retry(lambda: api.get_bucket_file_metadata(BUCKET, path))
        except HfHubHTTPError as error:
            if error.response.status_code != 404:
                raise
            print("false")
            return
        print("true")
        return
    retry(lambda: api.batch_bucket_files(BUCKET, add=[(args.source, path)]))


if __name__ == "__main__":
    main()
