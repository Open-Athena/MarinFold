"""Compress and publish the Stage A alignment evidence to the public HF bucket."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF_PREFIX = (
    "hf://buckets/open-athena/MarinFold/"
    "data/evals/exp350_complex_holdout_survival/evidence"
)
ARMS = ("complex", "native", "helico")
COVERAGES = ("query", "target")


def compress_with_manifest(source: Path, destination: Path) -> dict:
    """Create a deterministic gzip while hashing and counting the raw input."""
    raw_hash = hashlib.sha256()
    lines = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("rb") as src,
        destination.open("wb") as raw_out,
        gzip.GzipFile(fileobj=raw_out, mode="wb", filename="", mtime=0) as out,
    ):
        while chunk := src.read(8 * 1024 * 1024):
            raw_hash.update(chunk)
            lines += chunk.count(b"\n")
            out.write(chunk)
    return {
        "object": destination.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": raw_hash.hexdigest(),
        "rows": lines,
        "gzip_bytes": destination.stat().st_size,
        "gzip_sha256": hashlib.file_digest(
            destination.open("rb"), "sha256"
        ).hexdigest(),
    }


def copy_with_manifest(source: Path, destination: Path) -> dict:
    """Copy a small provenance file and return its public-object metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "object": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
    }


def main() -> None:
    """Build a self-checking evidence bundle and optionally upload it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("/data/exp350"))
    ap.add_argument(
        "--pair-confirm-work",
        type=Path,
        default=Path("/data/exp350_pair_confirm"),
    )
    ap.add_argument(
        "--staging", type=Path, default=Path("/data/exp350/public_evidence")
    )
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    args.staging.mkdir(parents=True, exist_ok=True)
    objects = []
    for arm in ARMS:
        for coverage in COVERAGES:
            alignment = args.work / f"{arm}_{coverage}_alignments.tsv"
            objects.append(
                compress_with_manifest(alignment, args.staging / f"{alignment.name}.gz")
            )
            log = args.work / f"{arm}_{coverage}_search.log"
            objects.append(copy_with_manifest(log, args.staging / log.name))
        search_manifest = HERE / "data" / f"{arm}_search.json"
        objects.append(
            copy_with_manifest(search_manifest, args.staging / search_manifest.name)
        )
    for name in [
        "candidate_provenance.json",
        "complex_sequence_manifest.json",
        "helico_reference.json",
        "native_db_manifest.json",
        "search_diagnostics.json",
    ]:
        source = HERE / "data" / name
        objects.append(copy_with_manifest(source, args.staging / name))
    for coverage in COVERAGES:
        alignment = args.pair_confirm_work / f"complex_{coverage}_alignments.tsv"
        public_name = f"pair_confirmation_{coverage}_alignments.tsv.gz"
        objects.append(compress_with_manifest(alignment, args.staging / public_name))
        log = args.pair_confirm_work / f"complex_{coverage}_search.log"
        public_log = f"pair_confirmation_{coverage}_search.log"
        objects.append(copy_with_manifest(log, args.staging / public_log))
    pair_search = HERE / "data/pair_confirmation_search.json"
    objects.append(copy_with_manifest(pair_search, args.staging / pair_search.name))
    manifest = {
        "public_prefix": HF_PREFIX,
        "scope": "complete broad and pair-survivor confirmation alignments and search logs supporting exp350 Stage A",
        "objects": objects,
    }
    manifest_path = args.staging / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(manifest_path, HERE / "data/evidence_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.prepare_only:
        subprocess.run(
            ["hf", "buckets", "sync", str(args.staging), HF_PREFIX], check=True
        )


if __name__ == "__main__":
    main()
