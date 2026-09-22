"""Extract the paired external-predictor GDT-TS rows used by exp311.

The source is Helico exp14's published per-target score table. The compact
output is committed so the comparison plots do not require network access.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SELECTIONS = HERE / "data" / "per_target_selection.csv"
OUTPUT = HERE / "data" / "predictor_baseline_gdt_ts.csv"
MANIFEST = HERE / "data" / "predictor_baseline_manifest.json"
SOURCE_URL = (
    "https://huggingface.co/buckets/timodonnell/helico-experiments/resolve/"
    "exp14_foldbench_held_out_monomers/scores/per_target.csv"
)
SOURCE_SHA256 = "1cba4eb14fbd92c842bc908f9114e20b0cba0176c79dd3b1f9735c899a3b62cb"
ARMS = {
    "esmfold": "ESMFold",
    "esmfold2": "ESMFold2",
    "protenix_v2_single_seq": "Protenix-v2 single sequence",
    "protenix_v2_msa": "Protenix-v2 + MSA",
}
EXPECTED_TARGETS = {"eval-val": 96, "eval-denovo": 19}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exp311_targets() -> set[tuple[str, str]]:
    """Load the exact target cohort from exp311's GDT-TS selections."""
    targets = set()
    with SELECTIONS.open() as stream:
        for row in csv.DictReader(stream):
            if row["metric"] == "gdt_ts":
                targets.add((row["eval_set"], row["stem"]))
    counts = {
        eval_set: sum(dataset == eval_set for dataset, _ in targets)
        for eval_set in EXPECTED_TARGETS
    }
    if counts != EXPECTED_TARGETS:
        raise ValueError(f"unexpected exp311 target coverage: {counts}")
    return targets


def extract(source: Path, targets: set[tuple[str, str]]) -> list[dict]:
    """Extract one valid GDT-TS value per target and external predictor."""
    rows = []
    seen = set()
    with source.open() as stream:
        for row in csv.DictReader(stream):
            target = (row["eval_set"], row["target_id"])
            if target not in targets or row["arm"] not in ARMS:
                continue
            if row["status"] != "ok" or not row["gdt_ts"]:
                raise ValueError(f"invalid baseline row: {row}")
            key = (*target, row["arm"])
            if key in seen:
                raise ValueError(f"duplicate baseline row: {key}")
            seen.add(key)
            value = float(row["gdt_ts"])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite baseline GDT-TS: {key}")
            rows.append({
                "eval_set": row["eval_set"],
                "stem": row["target_id"],
                "method_id": row["arm"],
                "method_label": ARMS[row["arm"]],
                "gdt_ts": value,
            })
    expected = {(*target, arm) for target in targets for arm in ARMS}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"baseline coverage mismatch; missing={missing}, extra={extra}")
    return sorted(rows, key=lambda row: (row["eval_set"], row["stem"], row["method_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Downloaded Helico exp14 scores/per_target.csv",
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
    MANIFEST.write_text(json.dumps({
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_bytes": args.source.stat().st_size,
        "metrics": ["gdt_ts"],
        "predictor_arms": ARMS,
        "target_counts": EXPECTED_TARGETS,
        "output_rows": len(rows),
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} paired baseline rows to {OUTPUT}")


if __name__ == "__main__":
    main()
