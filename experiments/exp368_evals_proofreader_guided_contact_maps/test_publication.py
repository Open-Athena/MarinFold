"""Test recovery of immutable native source bundles without a workstation cache."""

import hashlib
from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from publish_to_hf import DESTINATION, restore_native_source


def source_record(payload: bytes) -> tuple[dict, str]:
    """Describe a small frozen archive and its published object key."""
    sha = hashlib.sha256(payload).hexdigest()
    record = dict(config_map='exp368-native-source-' + sha[:12],
                  bundle_sha256=sha, bundle_bytes=len(payload))
    key = DESTINATION.removeprefix('hf://') + '/source_bundles/' + sha + '.tar.gz'
    return record, key


def test_restore_missing_native_bundle_and_reuse_verified_cache(tmp_path: Path) -> None:
    payload = b'frozen executed source'
    record, key = source_record(payload)
    fs = MemoryFileSystem()
    fs.pipe_file(key, payload)
    restored = restore_native_source(record, tmp_path / 'cache', fs)
    assert restored.read_bytes() == payload
    fs.rm(key)
    assert restore_native_source(record, tmp_path / 'cache', fs) == restored
    restored.write_bytes(b'x' * len(payload))
    with pytest.raises(ValueError, match='hash mismatch'):
        restore_native_source(record, tmp_path / 'cache', fs)


@pytest.mark.parametrize('bad_payload', [b'wrong-size', b'x' * len(b'expected source bytes')])
def test_invalid_download_is_not_committed(tmp_path: Path, bad_payload: bytes) -> None:
    record, key = source_record(b'expected source bytes')
    fs = MemoryFileSystem()
    fs.pipe_file(key, bad_payload)
    with pytest.raises(ValueError, match='wrong size|hash mismatch'):
        restore_native_source(record, tmp_path / 'cache', fs)
    assert not list(tmp_path.rglob('*.tar.gz'))
    assert not list(tmp_path.rglob('*.partial'))


class InterruptedDownloadFileSystem(MemoryFileSystem):
    """Model a transport failure after writing only part of the source archive."""

    def get_file(self, rpath: str, lpath: str, **kwargs: object) -> None:
        Path(lpath).write_bytes(b'partial')
        raise OSError('interrupted download')


def test_interrupted_download_preserves_failure_and_removes_partial(tmp_path: Path) -> None:
    record, key = source_record(b'interrupted source bytes')
    fs = InterruptedDownloadFileSystem()
    fs.pipe_file(key, b'interrupted source bytes')
    with pytest.raises(OSError, match='interrupted download'):
        restore_native_source(record, tmp_path / 'cache', fs)
    assert not list(tmp_path.rglob('*.tar.gz'))
    assert not list(tmp_path.rglob('*.partial'))


def test_missing_source_is_not_silently_skipped(tmp_path: Path) -> None:
    record, _ = source_record(b'missing immutable archive')
    with pytest.raises(FileNotFoundError):
        restore_native_source(record, tmp_path / 'cache', MemoryFileSystem())


def test_oversized_native_bundle_is_rejected_before_download(tmp_path: Path) -> None:
    record, _ = source_record(b'oversized')
    record = {**record, 'bundle_bytes': 700_001}
    with pytest.raises(ValueError, match='small-artifact ceiling'):
        restore_native_source(record, tmp_path / 'cache', MemoryFileSystem())
