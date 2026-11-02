# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Inline the dashboard data and its viewer into the template.

The page is one self-contained file on purpose. It has to work from a GitHub
blob preview, from a proxy like htmlpreview, from a file:// path on someone's
laptop, and from the bucket — none of which agree about relative fetches, CORS,
or third-party scripts. htmlpreview in particular renders the page but blocks
the CDN, which silently costs the 3D viewer. So both the data and 3Dmol.js go
into the document: ~1 MB, and it works anywhere.

3Dmol.js is BSD-3-Clause; the license text travels with it in
``dashboard/3Dmol-LICENSE.txt``.

    uv run python dashboard/build_page.py
"""

import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "__DASHBOARD_DATA__"

#: Pinned viewer build, cached beside the template so a rebuild is offline.
VIEWER_VERSION = "2.4.2"
VIEWER_URL = f"https://cdn.jsdelivr.net/npm/3dmol@{VIEWER_VERSION}/build/3Dmol-min.js"
VIEWER_TAG = f'<script src="{VIEWER_URL}"></script>'
VIEWER_CACHE = HERE / "scratch" / f"3Dmol-min-{VIEWER_VERSION}.js"


def viewer_source() -> str:
    """Return the pinned 3Dmol.js bundle, downloading it once."""

    if not VIEWER_CACHE.exists() or not VIEWER_CACHE.stat().st_size:
        VIEWER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VIEWER_URL) as response:
            VIEWER_CACHE.write_bytes(response.read())
    return VIEWER_CACHE.read_text()


def main() -> None:
    template = (HERE / "template.html").read_text()
    data = (HERE / "data.json").read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"{PLACEHOLDER} is not in the template")
    # The payload sits in a <script type="application/json"> block, so the only
    # sequence that can end it early is a literal closing script tag.
    payload = data.replace("</", "<\\/")
    page = template.replace(PLACEHOLDER, payload)
    if VIEWER_TAG not in page:
        raise ValueError("the template no longer loads the viewer from the CDN")
    page = page.replace(
        VIEWER_TAG,
        f"<!-- 3Dmol.js {VIEWER_VERSION}, BSD-3-Clause; see dashboard/3Dmol-LICENSE.txt -->\n"
        f"<script>{viewer_source()}</script>",
    )
    destination = HERE / "index.html"
    destination.write_text(page)
    proteins = json.loads(data)["proteins"]
    print(
        json.dumps(
            {
                "proteins": len(proteins),
                "bytes": destination.stat().st_size,
                "out": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
