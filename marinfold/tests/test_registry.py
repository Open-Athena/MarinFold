# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import marinfold.registry as registry
import textwrap
from pathlib import Path
from typing import BinaryIO

import pytest

from marinfold.registry import (
    _locate_models_yaml,
    _parse_hf_url,
    default_model_nickname,
    list_model_entries,
    resolve_model_entry,
)


def test_locate_models_yaml_works_from_arbitrary_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nickname resolution should not depend on the caller's cwd being in-repo.

    The locator walks up from the package install location as a fallback,
    so it still finds the packaged MODELS registry when the caller's cwd
    is unrelated.
    """
    monkeypatch.delenv("MARINFOLD_MODELS_YAML", raising=False)
    monkeypatch.chdir(Path("/tmp"))

    yaml_path = _locate_models_yaml()

    assert yaml_path.name == "MODELS.yaml"
    assert yaml_path.is_file()
    assert yaml_path == Path(registry.__file__).resolve().with_name("MODELS.yaml")


def test_repo_models_yaml_has_exactly_one_default() -> None:
    """Validates the checked-in MODELS.yaml; also exercises list_model_entries."""
    entries = list_model_entries()
    defaults = [e for e in entries if e.default]
    assert len(defaults) == 1, f"expected exactly one default; got {defaults}"
    assert default_model_nickname() == defaults[0].nickname


def test_resolve_model_entry_by_nickname_and_default() -> None:
    by_default = resolve_model_entry(None)
    by_name = resolve_model_entry(by_default.nickname)
    assert by_default == by_name


def test_resolve_model_entry_unknown_nickname_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="not found in MODELS.yaml"):
        resolve_model_entry("definitely-not-a-real-nickname-xyz")


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def test_multiple_defaults_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "MODELS.yaml"
    _write_yaml(yaml_path, """
        - nickname: A
          default: true
          url: https://huggingface.co/x/y/tree/main/sub
        - nickname: B
          default: true
          url: https://huggingface.co/x/z/tree/main/sub
    """)
    monkeypatch.setenv("MARINFOLD_MODELS_YAML", str(yaml_path))
    with pytest.raises(ValueError, match="multiple entries marked default"):
        list_model_entries()


def test_wandb_url_field_parsed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "MODELS.yaml"
    _write_yaml(yaml_path, """
        - nickname: A
          default: true
          url: https://huggingface.co/x/y/tree/main/sub
          wandb_url: https://wandb.ai/o/p/runs/abc
        - nickname: B
          url: https://huggingface.co/x/z/tree/main/sub
    """)
    monkeypatch.setenv("MARINFOLD_MODELS_YAML", str(yaml_path))
    entries = {e.nickname: e for e in list_model_entries()}
    assert entries["A"].wandb_url == "https://wandb.ai/o/p/runs/abc"
    assert entries["B"].wandb_url is None


def test_wandb_url_non_string_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "MODELS.yaml"
    _write_yaml(yaml_path, """
        - nickname: A
          url: https://huggingface.co/x/y/tree/main/sub
          wandb_url: 42
    """)
    monkeypatch.setenv("MARINFOLD_MODELS_YAML", str(yaml_path))
    with pytest.raises(ValueError, match="'wandb_url' must be a string"):
        list_model_entries()


def test_parse_hf_url_regular_repo() -> None:
    loc = _parse_hf_url(
        "https://huggingface.co/timodonnell/LlamaFold-experiments/tree/main/marin-experiments.protein-contacts-1b"
    )
    assert loc.repo_id == "timodonnell/LlamaFold-experiments"
    assert loc.revision == "main"
    assert loc.subfolder == "marin-experiments.protein-contacts-1b"
    assert loc.is_bucket is False


def test_parse_hf_url_regular_repo_no_tree() -> None:
    loc = _parse_hf_url("https://huggingface.co/org/repo")
    assert loc.repo_id == "org/repo"
    assert loc.revision == "main"
    assert loc.subfolder is None
    assert loc.is_bucket is False


def test_parse_hf_url_bucket() -> None:
    loc = _parse_hf_url(
        "https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999"
    )
    assert loc.repo_id == "open-athena/MarinFold"
    assert loc.subfolder == "checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999"
    assert loc.is_bucket is True


def test_parse_hf_url_bucket_no_tree() -> None:
    loc = _parse_hf_url("https://huggingface.co/buckets/open-athena/MarinFold")
    assert loc.repo_id == "open-athena/MarinFold"
    assert loc.subfolder is None
    assert loc.is_bucket is True


def test_parse_hf_url_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Could not parse HuggingFace URL"):
        _parse_hf_url("not a url")


def test_download_bucket_prefix_uses_http_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub.constants as hf_constants
    import huggingface_hub.file_download as hf_file_download
    import huggingface_hub.utils as hf_utils

    location = _parse_hf_url(
        "https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999"
    )
    assert location.is_bucket is True

    expected_dir = (
        tmp_path
        / "hf-cache"
        / "buckets"
        / "open-athena/MarinFold"
        / "checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999"
    )
    download_url = (
        "https://example.test/buckets/open-athena/MarinFold/resolve/"
        "checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999/config.json"
    )
    tree_url = (
        "https://example.test/api/buckets/open-athena/MarinFold/tree/"
        "checkpoints%2Fprotein-contacts-1_5b-distance-masked-70f8f5%2Fstep-49999"
    )
    seen: dict[str, object] = {}

    class _FakeResponse:
        def json(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "file",
                    "path": (
                        "checkpoints/protein-contacts-1_5b-distance-masked-"
                        "70f8f5/step-49999/config.json"
                    ),
                    "size": 5,
                }
            ]

    class _FakeSession:
        def get(
            self,
            url: str,
            *,
            params: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
        ) -> _FakeResponse:
            seen["tree_url"] = url
            seen["tree_params"] = params
            seen["tree_headers"] = headers
            return _FakeResponse()

    def _fake_http_get(
        url: str,
        temp_file: BinaryIO,
        *,
        headers: dict[str, str] | None = None,
        expected_size: int | None = None,
        displayed_filename: str | None = None,
        **_: object,
    ) -> None:
        seen["download_url"] = url
        seen["download_headers"] = headers
        seen["expected_size"] = expected_size
        seen["displayed_filename"] = displayed_filename
        temp_file.write(b"hello")

    monkeypatch.setattr(hf_constants, "ENDPOINT", "https://example.test")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setattr(
        hf_utils, "build_hf_headers", lambda **_: {"authorization": "Bearer test"}
    )
    monkeypatch.setattr(hf_utils, "get_session", lambda: _FakeSession())
    monkeypatch.setattr(hf_utils, "hf_raise_for_status", lambda response: None)
    monkeypatch.setattr(hf_file_download, "http_get", _fake_http_get)

    resolved = registry._download_bucket_prefix(location)

    assert resolved == expected_dir.resolve()
    assert seen["tree_url"] == tree_url
    assert seen["tree_params"] == {"recursive": "true"}
    assert seen["download_url"] == download_url
    assert seen["expected_size"] == 5
    assert seen["displayed_filename"] == "config.json"
    assert (expected_dir / "config.json").read_bytes() == b"hello"


def test_no_default_means_default_lookup_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "MODELS.yaml"
    _write_yaml(yaml_path, """
        - nickname: A
          url: https://huggingface.co/x/y/tree/main/sub
    """)
    monkeypatch.setenv("MARINFOLD_MODELS_YAML", str(yaml_path))
    with pytest.raises(LookupError, match="default: true"):
        default_model_nickname()
    with pytest.raises(LookupError, match="default: true"):
        resolve_model_entry(None)
