"""Freeze the context-complete contact and contact-conditioned Helico subset."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCE = DATA / "foldbench_complex_eval_targets.parquet"
TARGETS = DATA / "foldbench_complex_contact_eval_targets.parquet"
SUMMARY = DATA / "foldbench_complex_contact_eval.csv"
ELIGIBILITY = DATA / "foldbench_complex_contact_context.csv"
MANIFEST = DATA / "foldbench_complex_contact_eval_manifest.json"
PUBLIC_PREFIX = (
    "hf://buckets/open-athena/MarinFold/data/evals/"
    "exp350_foldbench_pair_holdout/contact_eval_v1"
)
MODEL_LABEL = "contacts-v1-exp343-m2-p06-complex-1.5B-step-280154"
DIAGNOSTIC_PREFIX = (
    "s3://marin-us-east-02a/MarinFold/"
    "exp350_evals_complex_holdout_survival/foldbench-pair-holdout-v1/"
    "rollout-diagnostic-full-context"
)
FULL_CONTEXT_UNFINISHED = {
    "8axi-assembly1": 12,
    "8b3w-assembly1": 7,
    "8btj-assembly1": 65,
    "8ok4-assembly1": 48,
    "8tn8-assembly1": 2,
    "8z4f-assembly1": 12,
    "9gsq-assembly1": 1,
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write deterministic CSV rows."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_dev_groups(rows: list[dict]) -> set[str]:
    """Choose 25% development targets by group using metadata only."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    items = []
    for group_id, members in grouped.items():
        items.append(
            (
                group_id,
                len(members),
                sum(row["complex_type"] == "homodimer" for row in members),
                sum(row["L"] for row in members),
            )
        )
    items.sort()
    total_count = len(rows)
    total_homo = sum(row["complex_type"] == "homodimer" for row in rows)
    total_length = sum(row["L"] for row in rows)
    target_count = round(total_count * 0.25)
    best_score = None
    best_groups: tuple[str, ...] | None = None

    def visit(
        index: int,
        count: int,
        homo: int,
        length: int,
        chosen: tuple[str, ...],
    ) -> None:
        nonlocal best_score, best_groups
        if count == target_count:
            score = (
                abs(homo * total_count - count * total_homo),
                abs(length * total_count - count * total_length),
                chosen,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_groups = chosen
            return
        if index == len(items) or count > target_count:
            return
        group_id, item_count, item_homo, item_length = items[index]
        visit(index + 1, count, homo, length, chosen)
        visit(
            index + 1,
            count + item_count,
            homo + item_homo,
            length + item_length,
            chosen + (group_id,),
        )

    visit(0, 0, 0, 0, ())
    if best_groups is None:
        raise ValueError(f"Cannot assign exactly {target_count} development targets")
    return set(best_groups)


def main() -> None:
    """Write the model-context eligibility audit and filtered target table."""
    source_table = pq.read_table(SOURCE)
    source_rows = source_table.to_pylist()
    source_stems = {row["stem"] for row in source_rows}
    unknown = set(FULL_CONTEXT_UNFINISHED) - source_stems
    if unknown:
        raise ValueError(f"Unknown context exclusions: {sorted(unknown)}")
    eligible = [
        dict(row)
        for row in source_rows
        if row["stem"] not in FULL_CONTEXT_UNFINISHED
    ]
    dev_groups = choose_dev_groups(eligible)
    for row in eligible:
        row["split"] = "dev" if row["group_id"] in dev_groups else "test"
    eligible.sort(key=lambda row: row["stem"])
    pq.write_table(pa.Table.from_pylist(eligible, schema=source_table.schema), TARGETS)

    contact_split = {row["stem"]: row["split"] for row in eligible}
    eligibility_rows = []
    for row in sorted(source_rows, key=lambda item: item["stem"]):
        unfinished = FULL_CONTEXT_UNFINISHED.get(row["stem"], 0)
        eligibility_rows.append(
            {
                "target_id": row["target_id"],
                "original_split": row["split"],
                "contact_split": contact_split.get(row["stem"], ""),
                "group_id": row["group_id"],
                "complex_type": row["complex_type"],
                "length": row["L"],
                "evaluated_rollouts": 100,
                "unfinished_full_context_rollouts": unfinished,
                "contact_eligible": unfinished == 0,
                "status": (
                    "included"
                    if unfinished == 0
                    else "excluded_full_8192_context_truncation"
                ),
                "model_label": MODEL_LABEL,
                "diagnostic_prefix": DIAGNOSTIC_PREFIX,
            }
        )
    write_csv(ELIGIBILITY, eligibility_rows)

    summary_rows = [
        {
            "target_id": row["target_id"],
            "split": row["split"],
            "group_id": row["group_id"],
            "complex_type": row["complex_type"],
            "length": row["L"],
            "n_gt": row["n_gt"],
            "n_resolved_pairs": row["n_resolved_pairs"],
        }
        for row in eligible
    ]
    write_csv(SUMMARY, summary_rows)

    selection = {
        "structural_targets": len(source_rows),
        "contact_eligible": len(eligible),
        "context_excluded": len(FULL_CONTEXT_UNFINISHED),
        "unfinished_rollouts": sum(FULL_CONTEXT_UNFINISHED.values()),
        "evaluated_rollouts": len(source_rows) * 100,
        "dev": sum(row["split"] == "dev" for row in eligible),
        "test": sum(row["split"] == "test" for row in eligible),
        "homodimer": sum(row["complex_type"] == "homodimer" for row in eligible),
        "heterodimer": sum(row["complex_type"] == "heterodimer" for row in eligible),
        "homology_groups": len({row["group_id"] for row in eligible}),
    }
    manifest = {
        "dataset": "foldbench_complex_pair_holdout_contact_eval_v1",
        "public_prefix": PUBLIC_PREFIX,
        "model_label": MODEL_LABEL,
        "diagnostic_prefix": DIAGNOSTIC_PREFIX,
        "diagnostic_jobs": [
            f"/bizon/exp350-foldbench-complex-s{index}of6-diagfull"
            for index in range(6)
        ],
        "eligibility_rule": (
            "100/100 rollouts must stop before the checkpoint's full remaining "
            "8192-token context"
        ),
        "selection": selection,
        "context_exclusions": FULL_CONTEXT_UNFINISHED,
        "files": {
            SOURCE.name: sha256(SOURCE),
            TARGETS.name: sha256(TARGETS),
            SUMMARY.name: sha256(SUMMARY),
            ELIGIBILITY.name: sha256(ELIGIBILITY),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
