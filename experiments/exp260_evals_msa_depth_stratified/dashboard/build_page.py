# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Inline the dashboard data into the template to produce ``index.html``.

The page is one self-contained file on purpose. It has to work from a GitHub
blob preview, from a file:// path on someone's laptop, and from the bucket,
none of which agree about relative fetches or CORS — so the data goes in the
document rather than beside it. The only thing loaded from the network is
3Dmol.js, and the page degrades to contact maps without it.

    uv run python dashboard/build_page.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "__DASHBOARD_DATA__"


def main() -> None:
    template = (HERE / "template.html").read_text()
    data = (HERE / "data.json").read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"{PLACEHOLDER} is not in the template")
    # The payload sits in a <script type="application/json"> block, so the only
    # sequence that can end it early is a literal closing script tag.
    payload = data.replace("</", "<\\/")
    page = template.replace(PLACEHOLDER, payload)
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
