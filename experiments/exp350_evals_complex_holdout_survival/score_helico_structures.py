"""Score cached Helico predictions on the exact frozen FoldBench protein pair."""

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
from helico.bench import _find_gt_path, download_foldbench
from helico.train import coords_to_pdb, pdb_chain_id_map

HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = HERE / "data/foldbench_complex_contact_eval_targets.parquet"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 350


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_name(run_dir: Path) -> tuple[str, str]:
    """Extract the frozen split and experimental arm from a cache directory."""
    for split in ("dev", "test"):
        prefix = f"exp350-{split}-"
        if run_dir.name.startswith(prefix):
            return split, run_dir.name.removeprefix(prefix)
    raise ValueError(
        f"Run directory name must start with exp350-dev- or exp350-test-: {run_dir}"
    )


def dockq_maps(
    native_chain_ids: list[str],
    predicted_chain_ids: list[str],
    model_chain_ids: dict[str, str],
    complex_type: str,
) -> list[dict[str, str]]:
    """Return the fixed mapping, plus the swapped homodimer mapping."""
    if len(native_chain_ids) != 2 or len(predicted_chain_ids) != 2:
        raise ValueError(
            f"Expected one native/predicted chain pair, got "
            f"{native_chain_ids}/{predicted_chain_ids}"
        )
    native_left, native_right = native_chain_ids
    predicted_left, predicted_right = predicted_chain_ids
    mappings = [
        {
            native_left: model_chain_ids[predicted_left],
            native_right: model_chain_ids[predicted_right],
        }
    ]
    if complex_type == "homodimer":
        mappings.append(
            {
                native_left: model_chain_ids[predicted_right],
                native_right: model_chain_ids[predicted_left],
            }
        )
    return mappings


def score_one(
    prediction_path: Path,
    target: dict,
    gt_dir: Path,
    structures_dir: Path,
    run_name: str,
) -> dict:
    """Score one confidence-selected Helico prediction with pair-specific DockQ."""
    with prediction_path.open("rb") as handle:
        prediction = pickle.load(handle)
    tokenized = prediction["tokenized"]
    target_id = target["target_id"]
    predicted_chain_ids = list(target["chain_ids"])
    native_chain_ids = list(target["author_chain_ids"])
    model_chain_ids = pdb_chain_id_map(tokenized.chain_ids)
    missing = set(predicted_chain_ids) - set(model_chain_ids)
    if missing:
        raise ValueError(
            f"{target_id}: predicted structure is missing frozen chains {sorted(missing)}"
        )

    pdb_string = coords_to_pdb(
        torch.as_tensor(prediction["pred_coords"]),
        torch.as_tensor(prediction["plddt"]),
        tokenized,
    )
    structure_path = structures_dir / run_name / f"{target_id}.pdb"
    structure_path.parent.mkdir(parents=True, exist_ok=True)
    structure_path.write_text(pdb_string + "\n")

    with tempfile.NamedTemporaryFile(
        suffix=".pdb", mode="w", delete=False
    ) as temporary:
        temporary.write(pdb_string)
        temporary_path = Path(temporary.name)
    try:
        model_structure = load_PDB(str(temporary_path))
        native_path = _find_gt_path(gt_dir, target_id)
        native_structure = load_PDB(str(native_path))
        scored_mappings = []
        for chain_map in dockq_maps(
            native_chain_ids,
            predicted_chain_ids,
            model_chain_ids,
            target["complex_type"],
        ):
            interfaces, total = run_on_all_native_interfaces(
                model_structure,
                native_structure,
                chain_map=chain_map,
            )
            if len(interfaces) != 1:
                raise ValueError(
                    f"{target_id}: expected one frozen interface for {chain_map}, "
                    f"got {sorted(interfaces)}"
                )
            scored_mappings.append(
                (float(total), next(iter(interfaces.values())), chain_map)
            )
        _, best, best_map = max(scored_mappings, key=lambda item: item[0])
    finally:
        os.unlink(temporary_path)

    dockq = float(best["DockQ"])
    if dockq >= 0.80:
        quality = "high"
    elif dockq >= 0.49:
        quality = "medium"
    elif dockq >= 0.23:
        quality = "acceptable"
    else:
        quality = "incorrect"
    return {
        "target_id": target_id,
        "group_id": target["group_id"],
        "complex_type": target["complex_type"],
        "chain_1": predicted_chain_ids[0],
        "chain_2": predicted_chain_ids[1],
        "author_chain_1": native_chain_ids[0],
        "author_chain_2": native_chain_ids[1],
        "dockq": dockq,
        "irmsd": float(best["iRMSD"]),
        "lrmsd": float(best["LRMSD"]),
        "fnat": float(best["fnat"]),
        "dockq_quality": quality,
        "acceptable_or_better": dockq >= 0.23,
        "selected_chain_map": json.dumps(best_map, sort_keys=True),
        "prediction_sha256": sha256(prediction_path),
        "structure_sha256": sha256(structure_path),
    }


def group_bootstrap_interval(rows: list[dict]) -> tuple[float, float]:
    """Bootstrap targets by frozen homology group and return a 95% interval."""
    by_group: dict[str, list[float]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(float(row["dockq"]))
    group_ids = sorted(by_group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [value for group_id in sampled for value in by_group[group_id]]
        means[index] = np.mean(values)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate one summary row per split and experimental arm."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["arm"]), []).append(row)
    output = []
    for (split, arm), arm_rows in sorted(grouped.items()):
        dockq = np.array([float(row["dockq"]) for row in arm_rows])
        low, high = group_bootstrap_interval(arm_rows)
        output.append(
            {
                "split": split,
                "arm": arm,
                "n_targets": len(arm_rows),
                "n_groups": len({row["group_id"] for row in arm_rows}),
                "mean_dockq": float(np.mean(dockq)),
                "median_dockq": float(np.median(dockq)),
                "group_bootstrap_95_low": low,
                "group_bootstrap_95_high": high,
                "acceptable_or_better_rate": float(
                    np.mean([row["acceptable_or_better"] for row in arm_rows])
                ),
                "mean_irmsd": float(np.mean([float(row["irmsd"]) for row in arm_rows])),
                "mean_lrmsd": float(np.mean([float(row["lrmsd"]) for row in arm_rows])),
                "mean_fnat": float(np.mean([float(row["fnat"]) for row in arm_rows])),
            }
        )
    return output


def paired_group_bootstrap_interval(rows: list[dict]) -> tuple[float, float]:
    """Bootstrap paired target differences by frozen homology group."""
    by_group: dict[str, list[float]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(float(row["dockq_delta"]))
    group_ids = sorted(by_group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [value for group_id in sampled for value in by_group[group_id]]
        means[index] = np.mean(values)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compare_test_arms(rows: list[dict]) -> list[dict]:
    """Return prespecified paired test comparisons for contact conditioning."""
    test_rows = [row for row in rows if row["split"] == "test"]
    by_arm = {
        arm: {row["target_id"]: row for row in test_rows if row["arm"] == arm}
        for arm in {row["arm"] for row in test_rows}
    }
    comparisons = [
        ("marinfold_all_L", "off"),
        ("marinfold_intra_L", "off"),
        ("marinfold_all_L", "marinfold_intra_L"),
        ("oracle", "off"),
    ]
    output = []
    for candidate, control in comparisons:
        if candidate not in by_arm or control not in by_arm:
            continue
        candidate_rows = by_arm[candidate]
        control_rows = by_arm[control]
        if set(candidate_rows) != set(control_rows):
            raise ValueError(
                f"Paired target mismatch for {candidate} versus {control}: "
                f"{sorted(set(candidate_rows) ^ set(control_rows))}"
            )
        paired = []
        for target_id in sorted(candidate_rows):
            candidate_row = candidate_rows[target_id]
            control_row = control_rows[target_id]
            if candidate_row["group_id"] != control_row["group_id"]:
                raise ValueError(f"Group mismatch for paired target {target_id}")
            paired.append(
                {
                    "target_id": target_id,
                    "group_id": candidate_row["group_id"],
                    "dockq_delta": (
                        float(candidate_row["dockq"]) - float(control_row["dockq"])
                    ),
                }
            )
        deltas = np.array([row["dockq_delta"] for row in paired])
        low, high = paired_group_bootstrap_interval(paired)
        output.append(
            {
                "split": "test",
                "candidate_arm": candidate,
                "control_arm": control,
                "n_targets": len(paired),
                "n_groups": len({row["group_id"] for row in paired}),
                "candidate_mean_dockq": float(
                    np.mean([float(row["dockq"]) for row in candidate_rows.values()])
                ),
                "control_mean_dockq": float(
                    np.mean([float(row["dockq"]) for row in control_rows.values()])
                ),
                "mean_paired_dockq_delta": float(np.mean(deltas)),
                "group_bootstrap_95_low": low,
                "group_bootstrap_95_high": high,
                "n_improved": int(np.sum(deltas > 0)),
                "n_tied": int(np.sum(deltas == 0)),
                "n_worse": int(np.sum(deltas < 0)),
                "acceptable_or_better_rate_delta": float(
                    np.mean(
                        [row["acceptable_or_better"] for row in candidate_rows.values()]
                    )
                    - np.mean(
                        [row["acceptable_or_better"] for row in control_rows.values()]
                    )
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV with stable field order."""
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Score one or more Helico cache directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path, required=True)
    args = parser.parse_args()

    targets = {row["target_id"]: row for row in pq.read_table(args.targets).to_pylist()}
    gt_dir = download_foldbench() / "examples" / "ground_truths"
    rows = []
    for run_dir in args.run_dir:
        split, arm = parse_run_name(run_dir)
        expected = {
            target_id
            for target_id, target in targets.items()
            if target["split"] == split
        }
        prediction_paths = sorted((run_dir / "predictions").glob("*.pkl"))
        observed = {path.stem for path in prediction_paths}
        if observed != expected:
            raise ValueError(
                f"{run_dir.name}: target mismatch; missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
        for prediction_path in prediction_paths:
            row = score_one(
                prediction_path,
                targets[prediction_path.stem],
                gt_dir,
                args.structures_dir,
                run_dir.name,
            )
            row = {"split": split, "arm": arm, "run_name": run_dir.name, **row}
            rows.append(row)
            print(f"{run_dir.name} {prediction_path.stem}: DockQ={row['dockq']:.4f}")

    write_csv(args.output, rows)
    summary = summarize(rows)
    write_csv(args.summary, summary)
    comparisons = compare_test_arms(rows)
    if comparisons:
        write_csv(args.comparisons, comparisons)
    for row in summary:
        print(
            f"{row['split']} {row['arm']}: mean DockQ={row['mean_dockq']:.4f} "
            f"[{row['group_bootstrap_95_low']:.4f}, {row['group_bootstrap_95_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
