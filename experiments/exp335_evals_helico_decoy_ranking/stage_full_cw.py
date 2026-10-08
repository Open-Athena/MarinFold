"""Stage full-run inputs and pinned Helico assets to CoreWeave object storage."""

import argparse
import base64
import concurrent.futures
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import s3fs

S3_ROOT = "marin-us-east-02a/MarinFold/exp335"
ASSET_PREFIX = f"{S3_ROOT}/assets"
INPUT_PREFIX = f"{S3_ROOT}/full-v1/inputs"
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
CHECKPOINT_SHA256 = "779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644"
SOURCE_ARCHIVE_SHA256 = (
    "28db5ab75a21424c3ea977eb61152e60d33758d04cf798f49c1bce3ba1a276f2"
)
CCD_SHA256 = "8531c4b72693d3afddfe4242c56a257be578445a32b25d260d2deb676715da04"
DEFAULT_KUBECONFIG = Path.home() / ".kube" / "coreweave-iris-rno2a"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coreweave_s3(kubeconfig: Path) -> s3fs.S3FileSystem:
    """Use the current cluster task credentials with the external S3 endpoint."""
    raw = subprocess.check_output(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "-n",
            "iris",
            "get",
            "secret",
            "iris-task-env",
            "-o",
            "json",
        ]
    )
    data = json.loads(raw)["data"]
    environment = {key: base64.b64decode(value).decode() for key, value in data.items()}
    config = json.loads(environment["FSSPEC_S3"])
    config["endpoint_url"] = "https://cwobject.com"
    config["key"] = environment["AWS_ACCESS_KEY_ID"]
    config["secret"] = environment["AWS_SECRET_ACCESS_KEY"]
    return s3fs.S3FileSystem(**config)


def build_source_archive(helico_repo: Path, destination: Path) -> str:
    """Create a deterministic archive of the pinned Helico source tree."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=helico_repo, text=True
    ).strip()
    if sha != HELICO_SHA:
        raise ValueError(f"Helico checkout is {sha}, expected {HELICO_SHA}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
        cwd=helico_repo,
        text=True,
    )
    if status:
        raise ValueError("Helico src or pyproject.toml has uncommitted changes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "src", "pyproject.toml"],
            cwd=helico_repo,
        ).split(b"\0")
        paths = [helico_repo / item.decode() for item in tracked if item]
        for path in paths:
            info = archive.gettarinfo(
                str(path), arcname=str(Path("helico") / path.relative_to(helico_repo))
            )
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    return sha256_file(destination)


def remote_digest(fs: s3fs.S3FileSystem, remote: str) -> str | None:
    """Read an object's SHA-256 sidecar when present."""
    sidecar = f"{remote}.sha256"
    if not fs.exists(sidecar):
        return None
    return fs.cat_file(sidecar).decode().strip()


def upload_one(
    fs: s3fs.S3FileSystem, local: Path, remote: str, expected_sha256: str | None = None
) -> dict:
    """Upload one file idempotently and publish its digest sidecar."""
    digest = sha256_file(local)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{local}: digest {digest} != {expected_sha256}")
    if (
        fs.exists(remote)
        and fs.size(remote) == local.stat().st_size
        and remote_digest(fs, remote) == digest
    ):
        status = "skip"
    else:
        fs.put_file(str(local), remote)
        if fs.size(remote) != local.stat().st_size:
            raise OSError(f"size mismatch after uploading {remote}")
        fs.pipe_file(f"{remote}.sha256", f"{digest}\n".encode())
        status = "uploaded"
    return {
        "local": str(local),
        "remote": f"s3://{remote}",
        "bytes": local.stat().st_size,
        "sha256": digest,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("scratch/full_inputs"))
    parser.add_argument("--helico-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ccd-cache",
        type=Path,
        default=Path(
            os.environ.get("HELICO_DATA_DIR", Path.home() / ".cache/helico/data")
        )
        / "processed"
        / "ccd_cache.pkl",
    )
    parser.add_argument("--kubeconfig", type=Path, default=DEFAULT_KUBECONFIG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--inputs-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Stage immutable assets first and the complete input manifest last."""
    args = parse_args()
    if args.assets_only and args.inputs_only:
        raise ValueError("--assets-only and --inputs-only are mutually exclusive")
    fs = coreweave_s3(args.kubeconfig)
    records = []

    if not args.inputs_only:
        archive = Path("scratch/assets") / f"helico-{HELICO_SHA[:12]}-src.tar.gz"
        build_source_archive(args.helico_repo, archive)
        assets = [
            (
                args.checkpoint,
                f"{ASSET_PREFIX}/contacts-msafree-01-step-6000.pt",
                CHECKPOINT_SHA256,
            ),
            (args.ccd_cache, f"{ASSET_PREFIX}/ccd_cache.pkl", CCD_SHA256),
            (archive, f"{ASSET_PREFIX}/{archive.name}", SOURCE_ARCHIVE_SHA256),
        ]
        for local, remote, digest in assets:
            record = upload_one(fs, local, remote, digest)
            records.append(record)
            print(f"{record['status']}: {record['remote']} ({record['bytes']} bytes)")

    if not args.assets_only:
        manifest_path = args.input_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        targets = manifest["targets"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    upload_one,
                    fs,
                    args.input_dir / target["relative_path"],
                    f"{INPUT_PREFIX}/{target['relative_path']}",
                    target["file_sha256"],
                ): target["target"]
                for target in targets
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                record = future.result()
                records.append(record)
                print(
                    f"staged {completed}/{len(futures)} targets: "
                    f"{record['status']} {record['remote']}",
                    flush=True,
                )
        manifest_record = upload_one(fs, manifest_path, f"{INPUT_PREFIX}/manifest.json")
        records.append(manifest_record)
        print(f"{manifest_record['status']}: {manifest_record['remote']}")

    summary = {
        "s3_root": f"s3://{S3_ROOT}",
        "objects": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
