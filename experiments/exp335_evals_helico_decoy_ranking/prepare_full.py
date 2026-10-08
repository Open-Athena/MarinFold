"""Prepare index-validated Helico inputs for the full AF2Rank benchmark.

The raw 5.18 GB PDB archive remains outside git. This script converts each
target independently into one deterministic gzip JSON payload so preparation
can run in parallel and resume target by target.
"""

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import gemmi
from pyconfind import cached_rotamer_library, load_library

from benchmark import load_af2rank_rows
from prepare_pilot import ca_coordinates, candidate_path, contact_pairs, map_digest

EXPECTED_AF2RANK_SHA256 = (
    "ddb3b91c27561212fa9152df4a4a436b9d01990cfe801569d7adc7f925fb75c9"
)
EXPECTED_TARGETS = 133
EXPECTED_DECOYS = 180_079
SCHEMA_VERSION = 2

_ROTAMER_LIBRARY = None


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_candidate_ids(candidate_ids: list[str]) -> str:
    """Hash an ordered candidate identifier list."""
    return hashlib.sha256(
        "".join(f"{item}\n" for item in candidate_ids).encode()
    ).hexdigest()


def digest_candidate_sources(
    decoy_dir: Path, target: str, candidate_ids: list[str]
) -> str:
    """Hash the ordered source PDB bytes that feed one target payload."""
    digest = hashlib.sha256()
    for decoy_id in candidate_ids:
        path = candidate_path(decoy_dir, target, decoy_id)
        digest.update(decoy_id.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def initialize_worker() -> None:
    """Load the pyconfind rotamer library once in each worker process."""
    global _ROTAMER_LIBRARY
    _ROTAMER_LIBRARY = load_library(cached_rotamer_library())


def deterministic_gzip_write(path: Path, payload: bytes) -> None:
    """Write gzip bytes atomically with a stable header timestamp."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as stream,
    ):
        stream.write(payload)
    temporary.replace(path)


def project_candidate_to_target(
    target_sequence: str,
    candidate_sequence: str,
    present_pairs: list[list[int]],
    candidate_ca: list[list[float]],
) -> tuple[list[list[int]], list[list[float]], dict]:
    """Project a terminally extended candidate onto exact target indices.

    The Rosetta archive contains at least one decoy with an extra resolved
    terminal residue. We retain it only when the complete target sequence occurs
    exactly once as a contiguous candidate substring. This gives a unique index
    map without inventing coordinates or silently accepting substitutions.
    """
    if len(candidate_ca) != len(candidate_sequence):
        raise ValueError(
            f"{len(candidate_ca)} C-alpha coordinates for "
            f"{len(candidate_sequence)} candidate residues"
        )
    if candidate_sequence == target_sequence:
        return (
            present_pairs,
            candidate_ca,
            {
                "method": "identity",
                "candidate_residues": len(candidate_sequence),
                "target_start": 0,
                "target_end": len(target_sequence),
                "dropped_candidate_residues": 0,
                "dropped_present_contacts": 0,
            },
        )

    starts = []
    offset = candidate_sequence.find(target_sequence)
    while offset >= 0:
        starts.append(offset)
        offset = candidate_sequence.find(target_sequence, offset + 1)
    if len(starts) != 1:
        raise ValueError(
            "candidate sequence is not a unique contiguous extension of the target "
            f"(candidate={len(candidate_sequence)}, target={len(target_sequence)}, "
            f"matches={starts})"
        )
    start = starts[0]
    end = start + len(target_sequence)
    projected_pairs = sorted(
        [left - start, right - start]
        for left, right in present_pairs
        if start <= left < end and start <= right < end
    )
    if len(projected_pairs) != len({tuple(pair) for pair in projected_pairs}):
        raise ValueError("projection created duplicate contact pairs")
    return (
        projected_pairs,
        candidate_ca[start:end],
        {
            "method": "unique_contiguous_target_subsequence",
            "candidate_residues": len(candidate_sequence),
            "target_start": start,
            "target_end": end,
            "dropped_candidate_residues": len(candidate_sequence)
            - len(target_sequence),
            "dropped_present_contacts": len(present_pairs) - len(projected_pairs),
        },
    )


def validate_existing(
    payload_path: Path,
    metadata_path: Path,
    *,
    source_fingerprint: str,
    candidate_ids_sha256: str,
    source_pdbs_sha256: str,
) -> dict | None:
    """Return valid resume metadata, or None when a target must be rebuilt."""
    if not payload_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("source_fingerprint") != source_fingerprint:
        return None
    if metadata.get("candidate_ids_sha256") != candidate_ids_sha256:
        return None
    if metadata.get("source_pdbs_sha256") != source_pdbs_sha256:
        return None
    if metadata.get("file_sha256") != sha256_file(payload_path):
        return None
    return metadata


def prepare_target(task: dict) -> dict:
    """Extract all candidate maps for one target and write its payload."""
    if _ROTAMER_LIBRARY is None:
        raise RuntimeError("worker rotamer library was not initialized")
    target = task["target"]
    candidate_ids = task["candidate_ids"]
    decoy_dir = Path(task["decoy_dir"])
    output_dir = Path(task["output_dir"])
    source_fingerprint = task["source_fingerprint"]
    candidate_ids_sha256 = digest_candidate_ids(candidate_ids)
    source_pdbs_sha256 = digest_candidate_sources(decoy_dir, target, candidate_ids)
    payload_path = output_dir / "targets" / f"{target}.json.gz"
    metadata_path = output_dir / "targets" / f"{target}.meta.json"
    existing = validate_existing(
        payload_path,
        metadata_path,
        source_fingerprint=source_fingerprint,
        candidate_ids_sha256=candidate_ids_sha256,
        source_pdbs_sha256=source_pdbs_sha256,
    )
    if existing is not None:
        return existing

    sequence = None
    candidates = []
    for decoy_id in candidate_ids:
        path = candidate_path(decoy_dir, target, decoy_id)
        structure = gemmi.read_structure(str(path))
        observed_sequence, observed_pairs = contact_pairs(structure, _ROTAMER_LIBRARY)
        observed_ca = ca_coordinates(structure)
        if sequence is None:
            sequence = observed_sequence
        try:
            pairs, candidate_ca, index_mapping = project_candidate_to_target(
                sequence, observed_sequence, observed_pairs, observed_ca
            )
        except ValueError as error:
            raise ValueError(f"{target}/{decoy_id}: {error}") from error
        candidates.append(
            {
                "decoy_id": decoy_id,
                "present_pairs": pairs,
                "candidate_ca": candidate_ca,
                "contact_map_sha256": map_digest(sequence, pairs),
                "index_mapping": index_mapping,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "target": target,
        "sequence": sequence,
        "candidates": candidates,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    input_sha256 = hashlib.sha256(encoded).hexdigest()
    deterministic_gzip_write(payload_path, encoded)
    metadata = {
        "target": target,
        "n_candidates": len(candidates),
        "n_residues": len(sequence),
        "candidate_ids_sha256": candidate_ids_sha256,
        "source_pdbs_sha256": source_pdbs_sha256,
        "source_fingerprint": source_fingerprint,
        "input_sha256": input_sha256,
        "file_sha256": sha256_file(payload_path),
        "file_bytes": payload_path.stat().st_size,
        "relative_path": str(payload_path.relative_to(output_dir)),
        "projected_candidates": sum(
            candidate["index_mapping"]["method"] != "identity"
            for candidate in candidates
        ),
        "dropped_candidate_residues": sum(
            candidate["index_mapping"]["dropped_candidate_residues"]
            for candidate in candidates
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--af2rank-csv", type=Path, required=True)
    parser.add_argument("--decoy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/full_inputs"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--limit-targets", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Prepare every target in parallel and write a complete input manifest."""
    args = parse_args()
    af2rank_sha256 = sha256_file(args.af2rank_csv)
    if af2rank_sha256 != EXPECTED_AF2RANK_SHA256:
        raise ValueError(f"unexpected corrected AF2Rank CSV digest: {af2rank_sha256}")
    rows = load_af2rank_rows(args.af2rank_csv)
    by_target: dict[str, list[str]] = {}
    for row in rows:
        decoy_id = row["decoy_id"]
        if decoy_id == "none":
            continue
        by_target.setdefault(row["target"], []).append(decoy_id)
    if len(by_target) != EXPECTED_TARGETS:
        raise ValueError(f"expected {EXPECTED_TARGETS} targets, found {len(by_target)}")
    decoy_count = sum(len(ids) - 1 for ids in by_target.values())
    if decoy_count != EXPECTED_DECOYS:
        raise ValueError(f"expected {EXPECTED_DECOYS} decoys, found {decoy_count}")

    targets = sorted(by_target)
    if args.limit_targets:
        targets = targets[: args.limit_targets]
    for target in targets:
        ids = by_target[target]
        if ids.count("native") != 1:
            raise ValueError(f"{target}: expected exactly one native row")
        by_target[target] = [
            "native",
            *sorted(item for item in ids if item != "native"),
        ]

    pyconfind_version = importlib.metadata.version("pyconfind")
    source_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "af2rank_sha256": af2rank_sha256,
                "pyconfind_version": pyconfind_version,
                "map_geometry": {
                    "native_only": True,
                    "contact_distance": 3.0,
                    "dcut": 25.0,
                    "clash_distance": 2.0,
                    "assembly": None,
                    "min_contact_degree": 0.001,
                    "min_sequence_separation": 6,
                },
                "candidate_index_mapping": (
                    "identity, or a unique contiguous target-sequence substring "
                    "within a terminally extended candidate"
                ),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    (args.output_dir / "targets").mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "target": target,
            "candidate_ids": by_target[target],
            "decoy_dir": str(args.decoy_dir),
            "output_dir": str(args.output_dir),
            "source_fingerprint": source_fingerprint,
        }
        for target in targets
    ]
    metadata = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker
    ) as pool:
        futures = {pool.submit(prepare_target, task): task["target"] for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            metadata.append(result)
            print(
                f"prepared {completed}/{len(tasks)} targets: {result['target']} "
                f"({result['n_candidates']} candidates, {result['file_bytes']} bytes)",
                flush=True,
            )

    metadata.sort(key=lambda item: item["target"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "af2rank_sha256": af2rank_sha256,
        "pyconfind_version": pyconfind_version,
        "n_targets": len(metadata),
        "n_candidates": sum(item["n_candidates"] for item in metadata),
        "projected_candidates": sum(item["projected_candidates"] for item in metadata),
        "dropped_candidate_residues": sum(
            item["dropped_candidate_residues"] for item in metadata
        ),
        "targets": metadata,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {manifest_path}: {manifest['n_targets']} targets, "
        f"{manifest['n_candidates']} candidates"
    )


if __name__ == "__main__":
    main()
