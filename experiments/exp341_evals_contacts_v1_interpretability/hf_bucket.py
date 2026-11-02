# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Column-projected reads of parquet files on the MarinFold HF bucket.

Both the B2 budget scan and the A3/A4 panel read a few columns from thousands
of corpus shards. Reading through HTTP range requests against the bucket's
resolve URLs fetches only the parquet footer and the requested column chunks.

Requests carry the caller's Hugging Face credentials through
``huggingface_hub.utils.build_hf_headers`` (the same headers
``marinfold.registry`` uses) when the caller is logged in, which raises the
per-account rate limit; EC2 workers have no token and read anonymously. A 429
is retried with backoff for up to ~1 h, long enough to outlast an anonymous
rate-limit window; any other error propagates.
"""

import time
from urllib.parse import quote

import aiohttp
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub.utils import build_hf_headers

BUCKET = "open-athena/MarinFold"
ENDPOINT = "https://huggingface.co"
RETRY_DELAYS_SECONDS = (15, 30, 60, 120, 240, 480, 900, 900, 900)


def list_parquet(prefix: str) -> list[str]:
    """Every parquet path under ``prefix``, following the tree API's cursor."""
    url: str | None = f"{ENDPOINT}/api/buckets/{BUCKET}/tree/{quote(prefix, safe='')}"
    params: dict | None = {"recursive": "true"}
    paths: list[str] = []
    while url:
        response = requests.get(url, params=params, headers=build_hf_headers(), timeout=60)
        response.raise_for_status()
        paths += [e["path"] for e in response.json() if e.get("type") == "file" and e["path"].endswith(".parquet")]
        url = response.links.get("next", {}).get("url")
        params = None  # the cursor URL carries its own query string
    return sorted(paths)


def read_columns(path: str, columns: list[str], *, block_size: int = 1 << 20) -> pa.Table:
    """Read ``columns`` of one bucket parquet file, backing off on HTTP 429."""
    url = f"{ENDPOINT}/buckets/{BUCKET}/resolve/{quote(path, safe='/')}"
    filesystem = fsspec.filesystem("https", block_size=block_size, client_kwargs={"headers": build_hf_headers()})
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            with filesystem.open(url, "rb") as handle:
                return pq.ParquetFile(handle).read(columns=columns)
        except aiohttp.ClientResponseError as error:
            if error.status != 429 or delay is None:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")
