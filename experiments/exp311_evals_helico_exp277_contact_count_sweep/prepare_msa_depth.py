"""Extract exp311 target depths from the A3Ms used by Helico exp14."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMPARISON = HERE / "data" / "gdt_ts_predictor_comparison_per_target.csv"
OUTPUT = HERE / "data" / "msa_depth.csv"
MANIFEST = HERE / "data" / "msa_depth_manifest.json"
SOURCE_COMMIT = "b10385d736673c81b10e70d1099962af6f2573c0"
SOURCE_PATH = "experiments/exp14_foldbench_held_out_monomers/data/msa_depth.csv"
SOURCE_URL = f"https://github.com/Open-Athena/helico/blob/{SOURCE_COMMIT}/{SOURCE_PATH}"
SOURCE_SHA256 = "94f4586a169f84b5129566bc239943fdd69b819660b212da497def62c39b4289"
EXPECTED_TARGETS = {"eval-val": 96, "eval-denovo": 19}
THRESHOLDS = (10, 100)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exp311_targets() -> set[tuple[str, str]]:
    """Load the target cohort already paired across all seven methods."""
    targets = set()
    with COMPARISON.open() as stream:
        for row in csv.DictReader(stream):
            targets.add((row["eval_set"], row["stem"]))
    counts = {
        eval_set: sum(dataset == eval_set for dataset, _ in targets)
        for eval_set in EXPECTED_TARGETS
    }
    if counts != EXPECTED_TARGETS:
        raise ValueError(f"unexpected exp311 target coverage: {counts}")
    return targets


def extract(source: Path, targets: set[tuple[str, str]]) -> list[dict]:
    """Extract one integer MSA depth for every exp311 target."""
    rows = []
    seen = set()
    with source.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["eval_set"], row["target_id"])
            if key not in targets:
                continue
            if key in seen:
                raise ValueError(f"duplicate MSA depth: {key}")
            seen.add(key)
            value = float(row["n_sequences"])
            if not math.isfinite(value) or value < 1 or not value.is_integer():
                raise ValueError(f"invalid MSA depth for {key}: {row['n_sequences']}")
            rows.append({
                "eval_set": row["eval_set"],
                "stem": row["target_id"],
                "msa_depth": int(value),
            })
    if seen != targets:
        raise ValueError(f"missing MSA depths: {sorted(targets - seen)}")
    return sorted(rows, key=lambda row: (row["eval_set"], row["stem"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Helico exp14 data/msa_depth.csv",
    )
    args = parser.parse_args()
    source_digest = sha256(args.source)
    if source_digest != SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, found {source_digest}"
        )

    rows = extract(args.source, exp311_targets())
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    subset_counts = {
        f"le_{threshold}": {
            eval_set: sum(row["eval_set"] == eval_set and row["msa_depth"] <= threshold for row in rows)
            for eval_set in EXPECTED_TARGETS
        }
        for threshold in THRESHOLDS
    }
    MANIFEST.write_text(json.dumps({
        "definition": "Number of sequences in the exact A3M supplied to Protenix-v2 + MSA, query included.",
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_path": SOURCE_PATH,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": args.source.stat().st_size,
        "target_counts": EXPECTED_TARGETS,
        "inclusive_subset_counts": subset_counts,
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} MSA depths to {OUTPUT}")
    print(json.dumps(subset_counts, indent=2))


if __name__ == "__main__":
    main()
