"""A regional copy must preserve replay even after original files disappear."""

import copy
import hashlib
import json
from pathlib import Path

import fsspec
import pytest

from corpus_stream import CorpusStream
from mirrored_corpus import MirroredCorpusStream
from test_corpus_stream import make_catalog, tiny_tokenizer


def make_mirror(tmp_path: Path) -> tuple[str, dict, dict]:
    prefix, manifest = make_catalog(tmp_path)
    root = tmp_path / "regional"
    root.mkdir()
    objects = {}
    for source in json.loads((tmp_path / "catalog.json").read_text()).values():
        for shard in source["shards"]:
            original = Path(shard["uri"])
            destination = root / original.name
            content = original.read_bytes()
            destination.write_bytes(content)
            fs, path = fsspec.core.url_to_fs(str(destination))
            info = fs.info(path)
            objects[str(original)] = {
                "uri": str(destination),
                "source_etag": shard["etag"],
                "metadata": {"size": info["size"], "mtime": info["mtime"]},
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    return prefix, manifest, {
        "root": str(root),
        "catalog_sha256": manifest["catalog_sha256"],
        "objects": objects,
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_mirror_replays_original_cursor_without_original_files(tmp_path, rank) -> None:
    prefix, manifest, mirror = make_mirror(tmp_path)
    original = CorpusStream(prefix, manifest, tiny_tokenizer(), rank, 2)
    for _ in range(834):
        original.next_raw()
    state = copy.deepcopy(original.state)
    # Include an epoch boundary, so coverage includes reshuffling and all sources.
    expected = [original.next_raw() for _ in range(900)]
    final_state = copy.deepcopy(original.state)
    original.close()
    for uri in mirror["objects"]:
        Path(uri).unlink()
    resumed = MirroredCorpusStream(prefix, manifest, tiny_tokenizer(), rank, 2, mirror)
    resumed.state = state
    assert [resumed.next_raw() for _ in range(900)] == expected
    assert resumed.state == final_state
    resumed.close()


@pytest.mark.parametrize("fault", ["missing", "remote", "identity", "version", "collision"])
def test_reject_incomplete_or_unverified_mirror(tmp_path, fault) -> None:
    prefix, manifest, mirror = make_mirror(tmp_path)
    uris = list(mirror["objects"])
    item = mirror["objects"][uris[0]]
    if fault == "missing":
        del mirror["objects"][uris[0]]
    elif fault == "remote":
        item["uri"] = uris[0]
    elif fault == "identity":
        item["source_etag"] = "different-source"
    elif fault == "version":
        item["metadata"] = {"size": item["metadata"]["size"]}
    else:
        mirror["objects"][uris[1]]["uri"] = item["uri"]
    with pytest.raises(ValueError):
        MirroredCorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2, mirror)


def test_mirror_mutation_after_verification_is_rejected(tmp_path) -> None:
    prefix, manifest, mirror = make_mirror(tmp_path)
    stream = MirroredCorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2, mirror)
    for item in mirror["objects"].values():
        path = Path(item["uri"])
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        stream.next_raw()


def test_mirror_cannot_change_catalog_identity(tmp_path) -> None:
    prefix, manifest, mirror = make_mirror(tmp_path)
    mirror = copy.deepcopy(mirror)
    mirror["catalog_sha256"] = "different-catalog"
    with pytest.raises(ValueError, match="different source catalog"):
        MirroredCorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2, mirror)
