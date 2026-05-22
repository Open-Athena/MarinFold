# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Model resolution for MarinFold.

``--model`` (or the ``model=`` arg to :func:`load_backend`) accepts:

1. A local directory that exists on disk → used as-is.
2. A nickname listed in repo-root ``MODELS.yaml`` → the matching HF
   URL is parsed and the relevant subfolder/prefix is downloaded.
   Two URL shapes are supported: regular model/dataset repos
   (``https://huggingface.co/<org>/<repo>/tree/<rev>/<subfolder>``)
   are fetched via ``huggingface_hub.snapshot_download``; storage
   buckets (``https://huggingface.co/buckets/<org>/<bucket>/tree/<prefix>``)
   are mirrored via the bucket HTTP API.
3. ``None`` → the entry marked ``default: true`` in ``MODELS.yaml``
   is used.

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
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote


_MODELS_YAML_FILENAME = "MODELS.yaml"
# Regular model/dataset repos: https://huggingface.co/<org>/<repo>[/tree/<rev>[/<subfolder>]]
_HF_URL_PATTERN = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)"
    r"(?:/tree/(?P<rev>[^/]+)(?:/(?P<subfolder>.+))?)?/?$"
)
# Storage buckets: https://huggingface.co/buckets/<org>/<bucket>[/tree/<prefix>]
# Buckets are flat (no branches), so the "/tree/<prefix>" segment is just a
# path prefix within the bucket, not a separate revision.
_HF_BUCKET_URL_PATTERN = re.compile(
    r"^https://huggingface\.co/buckets/(?P<repo>[^/]+/[^/]+)"
    r"(?:/tree/(?P<prefix>.+?))?/?$"
)


@dataclass(frozen=True)
class ModelEntry:
    """One row of ``MODELS.yaml``.

    Attributes:
        nickname: Short name used on the CLI (e.g. ``"1B"``).
        url: HuggingFace tree URL pointing at the model's directory.
            Either a regular repo URL
            (``https://huggingface.co/<org>/<repo>/tree/<rev>/<subfolder>``)
            or a storage-bucket URL
            (``https://huggingface.co/buckets/<org>/<bucket>/tree/<prefix>``).
        document_structures: Ordered tuple of supported document-
            structure names. The first entry is the implicit default
            when ``marinfold infer`` / ``marinfold evaluate`` is called
            without ``--document-structure``.
        default: True for the entry that should be picked when the
            user doesn't pass ``--model``. At most one entry in
            ``MODELS.yaml`` may have this set.
    """

    nickname: str
    url: str
    document_structures: tuple[str, ...] = field(default_factory=tuple)
    default: bool = False


@dataclass(frozen=True)
class _HFLocation:
    """A parsed HF URL: a regular repo or a storage bucket.

    For regular repos, ``revision`` is the git ref and ``subfolder`` is
    an optional directory within the snapshot. For buckets, ``revision``
    is unused (buckets are flat) and ``subfolder`` is the path prefix
    within the bucket.
    """

    repo_id: str
    revision: str
    subfolder: str | None
    is_bucket: bool = False


@dataclass(frozen=True)
class _BucketFileEntry:
    """One file listed under a storage-bucket prefix."""

    path: str
    size: int


def list_model_entries() -> list[ModelEntry]:
    """Parse ``MODELS.yaml`` into a validated list of :class:`ModelEntry`.

    Raises:
        ValueError: the file is malformed, an entry is missing a
            required field, or more than one entry has
            ``default: true``.
    """
    import yaml

    yaml_path = _locate_models_yaml()
    with yaml_path.open() as fh:
        raw = yaml.safe_load(fh) or []
    if not isinstance(raw, list):
        raise ValueError(
            f"{yaml_path} must contain a YAML list of model entries; "
            f"got {type(raw).__name__}."
        )
    entries: list[ModelEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"{yaml_path} entry #{i} is not a mapping: {item!r}."
            )
        if "nickname" not in item:
            raise ValueError(f"{yaml_path} entry #{i} is missing 'nickname'.")
        if "url" not in item:
            raise ValueError(
                f"{yaml_path} entry {item['nickname']!r} is missing 'url'."
            )
        ds = item.get("document_structures") or []
        if not isinstance(ds, list) or not all(isinstance(x, str) for x in ds):
            raise ValueError(
                f"{yaml_path} entry {item['nickname']!r}: "
                f"'document_structures' must be a list of strings."
            )
        entries.append(
            ModelEntry(
                nickname=str(item["nickname"]),
                url=str(item["url"]),
                document_structures=tuple(ds),
                default=bool(item.get("default", False)),
            )
        )
    defaults = [e for e in entries if e.default]
    if len(defaults) > 1:
        names = ", ".join(repr(e.nickname) for e in defaults)
        raise ValueError(
            f"{yaml_path}: multiple entries marked default: true ({names}). "
            f"Exactly one entry may be the default."
        )
    return entries


def default_model_nickname() -> str:
    """Return the nickname of the entry marked ``default: true``.

    Raises:
        LookupError: no entry has ``default: true``.
    """
    for entry in list_model_entries():
        if entry.default:
            return entry.nickname
    raise LookupError(
        "No model in MODELS.yaml is marked 'default: true'. "
        "Pass --model <nickname-or-path> or add 'default: true' to one entry."
    )


def resolve_model_entry(spec: str | None) -> ModelEntry:
    """Look up the :class:`ModelEntry` for a nickname (or the default).

    Args:
        spec: A nickname listed in ``MODELS.yaml``, or ``None`` to
            select the entry marked ``default: true``. Local paths
            are not accepted here — :func:`resolve_model` handles
            those separately.

    Raises:
        KeyError: ``spec`` is not a known nickname.
        LookupError: ``spec`` is ``None`` and no entry is marked
            default.
    """
    entries = list_model_entries()
    if spec is None:
        for entry in entries:
            if entry.default:
                return entry
        raise LookupError(
            "No model in MODELS.yaml is marked 'default: true'. "
            "Pass --model <nickname> or add 'default: true' to one entry."
        )
    for entry in entries:
        if entry.nickname == spec:
            return entry
    known = ", ".join(repr(e.nickname) for e in entries)
    raise KeyError(
        f"Nickname {spec!r} not found in MODELS.yaml. Known: {known}."
    )


def resolve_model(spec: str | None) -> Path:
    """Resolve a model spec to a local directory on disk.

    Args:
        spec: Either a path to a local directory (must exist), a
            nickname listed in repo-root ``MODELS.yaml``, or
            ``None`` to use the entry marked ``default: true``.

    Returns:
        Absolute :class:`Path` to a directory holding the model
        files (``config.json``, ``model.safetensors``,
        ``tokenizer.json``, ``tokenizer_config.json``).

    Raises:
        KeyError: ``spec`` is not a local directory and is not a
            known nickname.
        LookupError: ``spec`` is ``None`` and no entry is marked
            default.
        ValueError: ``spec`` matches a nickname whose ``url`` is not
            a parseable HuggingFace tree URL.
    """
    # Default-by-omission.
    if spec is None:
        entry = resolve_model_entry(None)
        return _download_subfolder(_parse_hf_url(entry.url))

    # Local directory takes precedence over nickname collisions.
    # To force the local interpretation when a name collides with a
    # nickname, use './<name>'.
    p = Path(spec)
    if p.is_dir():
        return p.resolve()

    # MODELS.yaml nickname.
    entry = _find_entry_by_nickname(spec)
    if entry is None:
        raise KeyError(
            f"Model {spec!r} is not a local directory and is not "
            f"listed in MODELS.yaml. Add it to MODELS.yaml or pass "
            f"a local path."
        )
    return _download_subfolder(_parse_hf_url(entry.url))


def _find_entry_by_nickname(nickname: str) -> ModelEntry | None:
    """Look up an entry in MODELS.yaml by ``nickname``; ``None`` if absent."""
    for entry in list_model_entries():
        if entry.nickname == nickname:
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
    """Parse a HuggingFace tree URL (regular repo or storage bucket).

    Accepts either:

    - ``https://huggingface.co/<org>/<repo>[/tree/<rev>[/<subfolder>]]``
    - ``https://huggingface.co/buckets/<org>/<bucket>[/tree/<prefix>]``
    """
    s = url.strip()
    # Try bucket pattern first — the regular pattern would otherwise greedily
    # match the "buckets/<org>" segment as a repo id.
    m = _HF_BUCKET_URL_PATTERN.match(s)
    if m is not None:
        prefix = m.group("prefix")
        if prefix is not None:
            prefix = prefix.rstrip("/") or None
        return _HFLocation(
            repo_id=m.group("repo"),
            revision="",
            subfolder=prefix,
            is_bucket=True,
        )
    m = _HF_URL_PATTERN.match(s)
    if m is None:
        raise ValueError(
            f"Could not parse HuggingFace URL {url!r}. Expected either "
            f"'https://huggingface.co/<org>/<repo>[/tree/<revision>[/<subfolder>]]' "
            f"or 'https://huggingface.co/buckets/<org>/<bucket>[/tree/<prefix>]'."
        )
    return _HFLocation(
        repo_id=m.group("repo"),
        revision=m.group("rev") or "main",
        subfolder=m.group("subfolder"),
    )


def _download_subfolder(location: _HFLocation) -> Path:
    """Download just the model's subfolder from HF Hub; return local path."""
    if location.is_bucket:
        return _download_bucket_prefix(location)

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


def _download_bucket_prefix(location: _HFLocation) -> Path:
    """Mirror a bucket prefix into the local HF cache and return its path.

    HF storage buckets are flat (no git revisions), so they go through a
    separate HTTP API instead of ``snapshot_download``. Files are mirrored to
    ``<HF_HUB_CACHE>/buckets/<repo_id>/<remote-path>``; an entry is
    re-downloaded only if it is missing locally or its size does not
    match the bucket's metadata.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    files = _list_bucket_files(location.repo_id, location.subfolder)
    if not files:
        raise FileNotFoundError(
            f"No files found in bucket {location.repo_id!r} under prefix "
            f"{location.subfolder!r}."
        )

    local_root = Path(HF_HUB_CACHE) / "buckets" / location.repo_id
    for file_entry in files:
        local_path = local_root / file_entry.path
        if local_path.is_file() and local_path.stat().st_size == file_entry.size:
            continue
        _download_bucket_file(location.repo_id, file_entry, local_path)

    result = (
        local_root
        if location.subfolder is None
        else local_root / location.subfolder
    )
    if not result.is_dir():
        raise FileNotFoundError(
            f"Expected bucket prefix at {result} after download, but it "
            f"does not exist."
        )
    return result.resolve()


def _list_bucket_files(
    repo_id: str, prefix: str | None
) -> list[_BucketFileEntry]:
    """List all files under a storage-bucket prefix."""
    from huggingface_hub import constants
    from huggingface_hub.utils import (
        build_hf_headers,
        get_session,
        hf_raise_for_status,
    )

    tree_url = _bucket_tree_url(constants.ENDPOINT, repo_id, prefix)
    response = get_session().get(
        tree_url,
        params={"recursive": "true"},
        headers=build_hf_headers(),
    )
    hf_raise_for_status(response)
    entries = response.json()
    if not isinstance(entries, list):
        raise ValueError(
            f"Bucket tree response for {repo_id!r} was not a list: "
            f"{type(entries).__name__}."
        )

    files: list[_BucketFileEntry] = []
    for item in entries:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = item.get("path")
        size = item.get("size")
        if not isinstance(path, str) or not isinstance(size, int):
            raise ValueError(
                f"Bucket tree response for {repo_id!r} contained an "
                f"invalid file entry: {item!r}."
            )
        files.append(_BucketFileEntry(path=path, size=size))
    return files


def _download_bucket_file(
    repo_id: str, file_entry: _BucketFileEntry, local_path: Path
) -> None:
    """Download one bucket file to its mirrored cache location."""
    from huggingface_hub import constants
    from huggingface_hub.file_download import http_get
    from huggingface_hub.utils import build_hf_headers

    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f"{local_path.name}.incomplete")
    try:
        with temp_path.open("wb") as fh:
            http_get(
                _bucket_resolve_url(constants.ENDPOINT, repo_id, file_entry.path),
                fh,
                headers=build_hf_headers(),
                expected_size=file_entry.size,
                displayed_filename=local_path.name,
            )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    temp_path.replace(local_path)


def _bucket_tree_url(endpoint: str, repo_id: str, prefix: str | None) -> str:
    """Build the bucket-tree API URL for a prefix."""
    encoded_prefix = f"/{quote(prefix, safe='')}" if prefix else ""
    return f"{endpoint}/api/buckets/{repo_id}/tree{encoded_prefix}"


def _bucket_resolve_url(endpoint: str, repo_id: str, path: str) -> str:
    """Build the public resolve URL for one file in a bucket."""
    return f"{endpoint}/buckets/{repo_id}/resolve/{quote(path, safe='/')}"
