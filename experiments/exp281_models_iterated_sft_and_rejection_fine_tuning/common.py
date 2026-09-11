"""Streaming storage and deterministic identities for exp281 artifacts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

EXPERIMENT = "exp281_models_iterated_sft_and_rejection_fine_tuning"
ROOT = "s3://marin-us-east-02a/protein-structure/MarinFold/exp281"
BASE_MODEL = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/models/"
    "exp232-decontam-train-m2-p06-step363000/hf/step-363000"
)


def identity(value: Any) -> str:
    """Hash JSON configuration independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_identity() -> str:
    """Pin experiment and format source bytes even in an Iris bundle without git."""
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((root / "marinfold/marinfold/document_structures/contacts_v1_multi").glob("*.py"))
    return identity({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def seed_for(*parts: Any) -> int:
    """Return a stable per-input seed, independent of shard assignment."""
    return int(identity(parts)[:8], 16)


def read_json(uri: str) -> Any:
    """Read a JSON artifact via the injected storage filesystem."""
    with fsspec.open(uri, "r") as handle:
        return json.load(handle)


def write_json(uri: str, value: Any) -> None:
    """Write a JSON artifact; callers write completion manifests last."""
    with fsspec.open(uri, "w", auto_mkdir=True) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def files(pattern: str) -> list[str]:
    """Expand an input glob deterministically and fail if it is empty."""
    fs, path = fsspec.core.url_to_fs(pattern)
    result = [fs.unstrip_protocol(p) for p in sorted(fs.glob(path))]
    if not result:
        raise ValueError(f"no files match {pattern}")
    return result


def rows(uri: str) -> Iterator[dict[str, Any]]:
    """Stream parquet record batches without constructing a whole-corpus table."""
    with fsspec.open(uri, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=128):
            yield from batch.to_pylist()


def write_rows(uri: str, records: list[dict[str, Any]]) -> None:
    """Write one bounded shard through fsspec, never pyarrow's native S3."""
    if not records:
        raise ValueError("cannot publish an empty data shard")
    with fsspec.open(uri, "wb", auto_mkdir=True) as handle:
        pq.write_table(pa.Table.from_pylist(records), handle, row_group_size=32)


def stage_model(uri: str, cache: Path, *, training_state: bool = False) -> Path:
    """Stage one immutable checkpoint locally for HF/vLLM weight loading.

    Model staging is confined to co-located worker storage. The source URI is
    part of the cache key; a completion marker is written only after all files.
    Training and generation parquet data remain streamed.
    """
    if "://" not in uri:
        path = Path(uri)
        if not path.is_dir():
            raise ValueError(f"model directory does not exist: {uri}")
        return path
    destination = cache / identity({"uri": uri, "training_state": training_state})[:20]
    if (destination / "_STAGED.json").exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    fs, source = fsspec.core.url_to_fs(uri)
    success = f"{source.rstrip('/')}/_SUCCESS.json"
    manifest = read_json(fs.unstrip_protocol(success)) if fs.exists(success) else None
    names = ([f"{source.rstrip('/')}/{name}" for name in manifest] + [success]
             if manifest else fs.find(source))
    if not names:
        raise ValueError(f"empty model prefix: {uri}")
    for name in names:
        relative = name.removeprefix(source.rstrip("/") + "/")
        if relative == "trainer.pt" and not training_state:
            continue
        local = destination / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with fs.open(name, "rb") as src, local.open("wb") as dst:
            while chunk := src.read(8 * 1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
        if manifest and relative in manifest and digest.hexdigest() != manifest[relative]["sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {relative}")
    if not (destination / "config.json").exists() or not (destination / "tokenizer.json").exists():
        raise ValueError(f"checkpoint is missing config/tokenizer: {uri}")
    write_json(str(destination / "_STAGED.json"), {"source": uri})
    return destination


def stage_tokenizer(uri: str, cache: Path) -> Path:
    """Fetch only tokenizer/config metadata, never weights, for CPU corpus jobs."""
    if "://" not in uri:
        return Path(uri)
    destination = cache / identity(uri)[:20]
    destination.mkdir(parents=True, exist_ok=True)
    fs, source = fsspec.core.url_to_fs(uri)
    for name in fs.ls(source, detail=False):
        leaf = name.rsplit("/", 1)[-1]
        if leaf.startswith(("tokenizer", "special_tokens", "added_tokens")) or leaf == "config.json":
            with fs.open(name, "rb") as src, (destination / leaf).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    if not (destination / "tokenizer.json").exists():
        raise ValueError("missing tokenizer.json")
    return destination


def publish_directory(local: Path, uri: str) -> None:
    """Publish files then a checksum manifest marking a complete checkpoint."""
    fs, success = fsspec.core.url_to_fs(f"{uri}/_SUCCESS.json")
    if fs.exists(success):
        raise FileExistsError(f"refusing to overwrite a complete checkpoint: {uri}")
    manifest = {}
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(local))
        digest = hashlib.sha256()
        with path.open("rb") as src, fsspec.open(f"{uri}/{relative}", "wb", auto_mkdir=True) as dst:
            while chunk := src.read(8 * 1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
        manifest[relative] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    write_json(f"{uri}/_SUCCESS.json", manifest)


def publish_with_deadline(local: Path, uri: str, timeout: float) -> None:
    """Bound all upload retries in a killable process; propagate failed publication.

    A socket timeout alone does not bound nested fsspec/botocore retries. Run the
    publisher outside the CUDA process so a deadline can actually terminate it.
    Failed uploads never earn a completion manifest. An interrupted multipart
    upload may need cleanup before retrying from the last complete checkpoint.
    """
    if timeout <= 0:
        raise ValueError("publication deadline must be positive")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(local), uri],
                   check=True, timeout=timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a complete checkpoint directory")
    parser.add_argument("local", type=Path)
    parser.add_argument("uri")
    arguments = parser.parse_args()
    publish_directory(arguments.local, arguments.uri)
