"""Fetch pinned upstream source and inference assets once per worker."""

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import fsspec

SOURCE_SHA = "a44b407daf6a5358e43cd68907f3e3f1cbc65fdc"
NGC_ROOT = "https://api.ngc.nvidia.com/v2/resources/nvidia/clara"
CHECKPOINTS = {
    "short": ("proteina_v1.2_dfs_200m_notri", "proteina_v1.2_DFS_200M_notri.ckpt"),
    "long": (
        "proteina_v1.6_dfs_200m_notri_long_chain_generation",
        "proteina_v1.6_DFS_200M_notri_long_chain_generation.ckpt",
    ),
}


def download(url: str, destination: Path) -> None:
    """Download an asset atomically, leaving incomplete downloads distinguishable."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    cache = (
        "s3://marin-us-east-02a/MarinFold/exp278-proteina/assets/"
        + hashlib.sha256(url.encode()).hexdigest()
        + "/"
        + destination.name
    )
    fs, path = fsspec.core.url_to_fs(cache)
    if fs.exists(path):
        print(f"Reading cached asset {destination.name}", flush=True)
        with fs.open(path, "rb") as source, temporary.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    else:
        print(f"Downloading {destination.name}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        with temporary.open("rb") as source, fs.open(path, "wb") as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    temporary.replace(destination)


def main() -> None:
    root = Path(os.environ["DATA_PATH"])
    archive = root / "proteina-source.tar.gz"
    download(
        f"https://github.com/NVIDIA-BioNeMo/proteina/archive/{SOURCE_SHA}.tar.gz",
        archive,
    )
    source = Path("/tmp/proteina")
    if not source.exists():
        with tarfile.open(archive) as handle:
            handle.extractall("/tmp", filter="data")
        Path(f"/tmp/proteina-{SOURCE_SHA}").rename(source)
    extras = root / "proteina_additional_files.zip"
    download(
        f"{NGC_ROOT}/proteina_additional_files/versions/1.0/files/{extras.name}", extras
    )
    with zipfile.ZipFile(extras) as handle:
        for member in handle.infolist():
            if member.filename.endswith("cath_label_mapping.pt"):
                target = root / "pdb_raw" / "cath_label_mapping.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(handle.read(member))
    manifests = []
    for key in filter(None, os.environ.get("PROTEINA_CHECKPOINTS", "short").split(",")):
        resource, filename = CHECKPOINTS[key]
        destination = root / filename
        download(f"{NGC_ROOT}/{resource}/versions/1.0/files/{filename}", destination)
        with destination.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        manifests.append(
            {
                "model": key,
                "filename": filename,
                "bytes": destination.stat().st_size,
                "sha256": checksum,
            }
        )
    print(json.dumps({"source_sha": SOURCE_SHA, "assets": manifests}), flush=True)
    (root / "assets.json").write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
