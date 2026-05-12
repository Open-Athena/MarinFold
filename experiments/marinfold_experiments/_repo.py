# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the PM scripts: repo-root resolution, gh-slug detection,
frontmatter parsing."""

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from marinfold_experiments import KINDS


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Filesystem path to the MarinFold repo root."""

REPO_SLUG_DEFAULT = "Open-Athena/MarinFold"


def git_repo_slug() -> str:
    """Best-effort: derive owner/name from the `origin` remote.

    Falls back to ``REPO_SLUG_DEFAULT`` if origin isn't set or the URL
    isn't a GitHub URL.
    """
    try:
        url = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "config", "--get", "remote.origin.url"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return REPO_SLUG_DEFAULT
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    if not m:
        return REPO_SLUG_DEFAULT
    return f"{m.group(1)}/{m.group(2)}"


def parse_experiment_dir_name(dir_name: str) -> tuple[int, str, str] | None:
    """Parse ``exp<N>_<kind>_<slug>`` → ``(N, kind, slug)``.

    Returns None if the name doesn't match the convention. ``<kind>``
    must be one of the recognised kinds (see :data:`KINDS` in
    ``__init__.py``); ``<slug>`` is everything after the kind token.
    """
    if not dir_name.startswith("exp"):
        return None
    rest = dir_name[3:]
    head, _, after_num = rest.partition("_")
    if not head.isdigit():
        return None
    n = int(head)

    # Longest-match against KINDS — `document_structures` contains an
    # underscore, so a simple split won't disambiguate it from a
    # `models` experiment whose slug happens to start with `structures_`.
    for kind in sorted(KINDS, key=len, reverse=True):
        prefix = kind + "_"
        if after_num.startswith(prefix):
            return n, kind, after_num[len(prefix):]
    return None


def read_frontmatter(readme_md: Path) -> dict[str, Any] | None:
    """Parse the ``marinfold_experiment:`` sub-block of a README's YAML frontmatter.

    Returns the parsed mapping (or None if the file has no frontmatter
    or no ``marinfold_experiment:`` block). Values keep their YAML
    types — lists like ``baselines: [...]`` come through as lists.
    """
    if not readme_md.exists():
        return None
    content = readme_md.read_text()
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    parsed = yaml.safe_load(parts[1]) or {}
    block = parsed.get("marinfold_experiment")
    return block if isinstance(block, dict) else None
