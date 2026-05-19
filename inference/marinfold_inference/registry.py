# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Model resolution for MarinFold inference.

``--model`` (or the ``model=`` arg to :func:`load_backend`) accepts:

1. A local directory that exists on disk → used as-is.
2. A nickname listed in repo-root ``MODELS.yaml`` → the matching HF
   URL is parsed into ``(repo_id, revision, subfolder)`` and the
   subfolder is downloaded via ``huggingface_hub.snapshot_download``.
   The returned :class:`Path` points at the local cache location.

Bare HF repo ids are not accepted by design — keeping the set of
known models small and named makes it easy to track which checkpoint
produced which eval number.

``MODELS.yaml`` is located in this order:

1. The path named by ``MARINFOLD_MODELS_YAML``.
2. Walking up from ``os.getcwd()``.
3. Walking up from this package's location, which covers the normal
   editable-install / repo-checkout case even when the caller's cwd is
   elsewhere.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MODELS_YAML_FILENAME = "MODELS.yaml"
_HF_URL_PATTERN = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)"
    r"(?:/tree/(?P<rev>[^/]+)(?:/(?P<subfolder>.+))?)?/?$"
)


@dataclass(frozen=True)
class _HFLocation:
    """A parsed HF URL: repo + revision + optional subfolder."""

    repo_id: str
    revision: str
    subfolder: str | None


def resolve_model(spec: str) -> Path:
    """Resolve a model spec to a local directory on disk.

    Args:
        spec: Either a path to a local directory (must exist), or a
            nickname listed in repo-root ``MODELS.yaml``.

    Returns:
        Absolute :class:`Path` to a directory holding the model
        files (``config.json``, ``model.safetensors``,
        ``tokenizer.json``, ``tokenizer_config.json``).

    Raises:
        KeyError: ``spec`` is not a local directory and is not a
            known nickname.
        ValueError: ``spec`` matches a nickname whose ``url`` is not
            a parseable HuggingFace tree URL.
    """
    # 1. Local directory takes precedence over nickname collisions.
    #    To force the local interpretation when a name collides with
    #    a nickname, use './<name>'.
    p = Path(spec)
    if p.is_dir():
        return p.resolve()

    # 2. MODELS.yaml nickname.
    entry = _find_entry_by_nickname(spec)
    if entry is None:
        raise KeyError(
            f"Model {spec!r} is not a local directory and is not "
            f"listed in MODELS.yaml. Add it to MODELS.yaml or pass "
            f"a local path."
        )
    location = _parse_hf_url(entry["url"])
    return _download_subfolder(location)


def _find_entry_by_nickname(nickname: str) -> dict[str, Any] | None:
    """Look up an entry in MODELS.yaml by ``nickname``."""
    import yaml

    yaml_path = _locate_models_yaml()
    with yaml_path.open() as fh:
        entries = yaml.safe_load(fh) or []
    if not isinstance(entries, list):
        raise ValueError(
            f"{yaml_path} must contain a YAML list of model entries; "
            f"got {type(entries).__name__}."
        )
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("nickname") == nickname:
            if "url" not in entry:
                raise ValueError(
                    f"MODELS.yaml entry for nickname {nickname!r} is "
                    f"missing the required 'url' field."
                )
            return entry
    return None


def _locate_models_yaml() -> Path:
    """Find ``MODELS.yaml`` via env override, cwd, or package location."""
    override = os.environ.get("MARINFOLD_MODELS_YAML")
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(
                f"MARINFOLD_MODELS_YAML={override!r} does not point at "
                f"an existing file."
            )
        return p

    search_roots = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for root in search_roots:
        for parent in [root, *root.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            candidate = parent / _MODELS_YAML_FILENAME
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not find {_MODELS_YAML_FILENAME} from cwd {Path.cwd().resolve()} "
        f"or from the package location {Path(__file__).resolve().parent}. "
        f"Set MARINFOLD_MODELS_YAML to point at it explicitly."
    )


def _parse_hf_url(url: str) -> _HFLocation:
    """Parse a ``https://huggingface.co/<repo>/tree/<rev>/<sub>`` URL."""
    m = _HF_URL_PATTERN.match(url.strip())
    if m is None:
        raise ValueError(
            f"Could not parse HuggingFace URL {url!r}. Expected "
            f"'https://huggingface.co/<org>/<repo>' optionally "
            f"followed by '/tree/<revision>[/<subfolder>]'."
        )
    return _HFLocation(
        repo_id=m.group("repo"),
        revision=m.group("rev") or "main",
        subfolder=m.group("subfolder"),
    )


def _download_subfolder(location: _HFLocation) -> Path:
    """Download just the model's subfolder from HF Hub; return local path."""
    from huggingface_hub import snapshot_download

    allow_patterns: list[str] | None
    if location.subfolder is None:
        allow_patterns = None
    else:
        allow_patterns = [f"{location.subfolder}/*"]

    local_root = snapshot_download(
        repo_id=location.repo_id,
        revision=location.revision,
        allow_patterns=allow_patterns,
    )
    local_path = Path(local_root)
    if location.subfolder is not None:
        local_path = local_path / location.subfolder
    if not local_path.is_dir():
        raise FileNotFoundError(
            f"Downloaded snapshot does not contain expected subfolder "
            f"{location.subfolder!r} at {local_path}."
        )
    return local_path.resolve()
