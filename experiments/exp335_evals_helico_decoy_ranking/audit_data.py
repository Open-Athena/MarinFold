"""Audit the exact Rosetta-decoy benchmark and reproduce paper baselines."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from benchmark import load_af2rank_rows, summarize_baselines

EXPECTED_ARCHIVE_BYTES = 5_179_742_249
EXPECTED_ARCHIVE_SHA256 = (
    "098a3d813b74588bfe126cb1ac5c27b3123ff70e187d636ffdb66b5dc4a57733"
)
EXPECTED_AF2RANK_BYTES = 24_695_656
EXPECTED_AF2RANK_SHA256 = (
    "ddb3b91c27561212fa9152df4a4a436b9d01990cfe801569d7adc7f925fb75c9"
)
EXPECTED_TARGETS = 133
EXPECTED_DECOYS = 180_079


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_keys(keys: Iterable[tuple[str, str]]) -> str:
    """Hash a sorted set of benchmark candidate identifiers."""
    digest = hashlib.sha256()
    for target, decoy_id in sorted(keys):
        digest.update(f"{target}\t{decoy_id}\n".encode())
    return digest.hexdigest()


def load_targets(paths: list[Path]) -> set[str]:
    """Load disjoint newline-delimited target lists."""
    targets: list[str] = []
    for path in paths:
        targets.extend(
            line.strip() for line in path.read_text().splitlines() if line.strip()
        )
    if len(targets) != len(set(targets)):
        raise ValueError("target files contain duplicate identifiers")
    return set(targets)


def load_reference_keys(path: Path) -> set[tuple[str, str]]:
    """Load ``target decoy_id metric value`` keys from an AF2Rank score file."""
    keys: set[tuple[str, str]] = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected four fields")
            key = (fields[0], fields[1])
            if key in keys:
                raise ValueError(f"{path}:{line_number}: duplicate key {key}")
            keys.add(key)
    return keys


def pdb_residue_summary(path: Path) -> tuple[int, int]:
    """Count distinct ATOM residues and chains in a benchmark native PDB."""
    residues: set[tuple[str, str, str]] = set()
    chains: set[str] = set()
    with path.open(errors="replace") as stream:
        for line in stream:
            if not line.startswith("ATOM  "):
                continue
            chain = line[21].strip() or "_"
            key = (chain, line[22:26].strip(), line[26].strip())
            residues.add(key)
            chains.add(chain)
    if not residues:
        raise ValueError(f"native PDB contains no ATOM residues: {path}")
    return len(residues), len(chains)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV with a stable column order."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--decoy-dir",
        type=Path,
        required=True,
        help="Directory containing one folder per target and a natives/ folder",
    )
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--af2rank-csv", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    """Validate all benchmark joins and write compact provenance artifacts."""
    args = parse_args()
    archive_size = args.archive.stat().st_size
    archive_sha256 = sha256_file(args.archive)
    if (
        archive_size != EXPECTED_ARCHIVE_BYTES
        or archive_sha256 != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError(
            f"unexpected archive identity: {archive_size} bytes, SHA-256 {archive_sha256}"
        )

    af2rank_size = args.af2rank_csv.stat().st_size
    af2rank_sha256 = sha256_file(args.af2rank_csv)
    if (
        af2rank_size != EXPECTED_AF2RANK_BYTES
        or af2rank_sha256 != EXPECTED_AF2RANK_SHA256
    ):
        raise ValueError(
            f"unexpected AF2Rank table identity: {af2rank_size} bytes, SHA-256 {af2rank_sha256}"
        )

    targets = load_targets(args.target_file)
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError(f"expected {EXPECTED_TARGETS} targets, found {len(targets)}")

    target_dirs = {
        path.name
        for path in args.decoy_dir.iterdir()
        if path.is_dir() and path.name != "natives"
    }
    if target_dirs != targets:
        raise ValueError(
            f"target-directory mismatch: extra={sorted(target_dirs - targets)}, "
            f"missing={sorted(targets - target_dirs)}"
        )

    pdb_keys = {
        (target, path.name)
        for target in targets
        for path in (args.decoy_dir / target).glob("*.pdb")
    }
    if len(pdb_keys) != EXPECTED_DECOYS:
        raise ValueError(
            f"expected {EXPECTED_DECOYS} decoy PDBs, found {len(pdb_keys)}"
        )

    native_dir = args.decoy_dir / "natives"
    missing_natives = sorted(
        target for target in targets if not (native_dir / f"{target}.pdb").is_file()
    )
    if missing_natives:
        raise ValueError(f"missing benchmark natives: {missing_natives}")

    reference_files = ["tmscore.txt", "rmsd.txt", "rosettascore.txt"]
    for filename in reference_files:
        keys = load_reference_keys(args.score_dir / filename)
        if keys != pdb_keys:
            raise ValueError(
                f"{filename} key mismatch: "
                f"PDB-only={len(pdb_keys - keys)}, table-only={len(keys - pdb_keys)}"
            )

    af2rank_rows = load_af2rank_rows(args.af2rank_csv)
    af2rank_decoy_keys = {
        (row["target"], row["decoy_id"])
        for row in af2rank_rows
        if row["decoy_id"] not in {"native", "none"}
    }
    if af2rank_decoy_keys != pdb_keys:
        raise ValueError(
            "corrected AF2Rank table key mismatch: "
            f"PDB-only={len(pdb_keys - af2rank_decoy_keys)}, "
            f"table-only={len(af2rank_decoy_keys - pdb_keys)}"
        )
    for control in ("native", "none"):
        control_targets = {
            row["target"] for row in af2rank_rows if row["decoy_id"] == control
        }
        if control_targets != targets:
            raise ValueError(
                f"AF2Rank {control} rows mismatch: extra={sorted(control_targets - targets)}, "
                f"missing={sorted(targets - control_targets)}"
            )

    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in af2rank_rows:
        if row["decoy_id"] not in {"native", "none"}:
            by_target[row["target"]].append(row)

    target_rows = []
    for target in sorted(targets):
        rows = by_target[target]
        tm_scores = [float(row["tmscore"]) for row in rows]
        gdt_ts_scores = [float(row["gdt_ts"]) for row in rows]
        n_residues, n_chains = pdb_residue_summary(native_dir / f"{target}.pdb")
        target_rows.append(
            {
                "target": target,
                "n_decoys": len(rows),
                "n_native_residues": n_residues,
                "n_native_chains": n_chains,
                "min_tmscore": min(tm_scores),
                "median_tmscore": statistics.median(tm_scores),
                "max_tmscore": max(tm_scores),
                "min_gdt_ts": min(gdt_ts_scores),
                "median_gdt_ts": statistics.median(gdt_ts_scores),
                "max_gdt_ts": max(gdt_ts_scores),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target_summary.csv", target_rows)
    baseline_rows = summarize_baselines(af2rank_rows)
    write_csv(args.output_dir / "baseline_summary.csv", baseline_rows)

    counts = [int(row["n_decoys"]) for row in target_rows]
    lengths = [int(row["n_native_residues"]) for row in target_rows]
    manifest = {
        "benchmark": "AF2Rank Rosetta decoy set",
        "sources": {
            "paper": "https://doi.org/10.1103/PhysRevLett.129.238101",
            "code": "https://github.com/jproney/AF2Rank",
            "archive": "https://files.ipd.uw.edu/pub/decoyset/decoys.zip",
            "corrected_results_folder": "https://drive.google.com/drive/folders/1Q0aCR_lk4R67XlX9IHl6Jk0-dUI19rhA",
        },
        "archive": {"bytes": archive_size, "sha256": archive_sha256},
        "af2rank_corrected_results": {
            "filename": args.af2rank_csv.name,
            "bytes": af2rank_size,
            "sha256": af2rank_sha256,
            "rows": len(af2rank_rows),
        },
        "coverage": {
            "targets": len(targets),
            "decoys": len(pdb_keys),
            "benchmark_natives": len(targets),
            "no_template_controls": len(targets),
            "decoys_per_target_min": min(counts),
            "decoys_per_target_median": statistics.median(counts),
            "decoys_per_target_max": max(counts),
            "native_residues_min": min(lengths),
            "native_residues_median": statistics.median(lengths),
            "native_residues_max": max(lengths),
        },
        "candidate_key_sha256": digest_keys(pdb_keys),
        "reference_tables": reference_files,
    }
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(manifest["coverage"], indent=2, sort_keys=True))
    for row in baseline_rows:
        print(
            f"{row['method']}: Spearman={row['mean_target_spearman_tmscore']:.6f}, "
            f"top-1 TM={row['mean_top1_tmscore']:.6f}"
        )


if __name__ == "__main__":
    main()
