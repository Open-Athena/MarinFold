#!/usr/bin/env python
"""Score every individual Helico structure against both fold references."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_helico_targets import SOURCE, prepare_structure
from score_helico_cross import prediction_coords, tm_score

HERE = Path(__file__).resolve().parent
ROOT = HERE / "_cache" / "helico_iid"


def superpose(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch-superpose C-alpha coordinates and return standard CA RMSD."""
    if len(predicted) < 3:
        return predicted, float("nan")
    pred_center, ref_center = predicted.mean(0), reference.mean(0)
    pred_zero, ref_zero = predicted - pred_center, reference - ref_center
    u, _, vt = np.linalg.svd(pred_zero.T @ ref_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = pred_zero @ rotation + ref_center
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))
    return aligned, rmsd


def gdt(distances: np.ndarray) -> float:
    """Helico-style GDT-TS after the supplied Kabsch superposition."""
    return float(np.mean([(distances < cutoff).mean() for cutoff in (1, 2, 4, 8)]))


def comparison_metrics(predicted: np.ndarray, reference: np.ndarray,
                       region: np.ndarray) -> dict[str, float]:
    """Global and switching-region metrics with explicit fit conventions."""
    globally_aligned, global_rmsd = superpose(predicted, reference)
    global_distances = np.linalg.norm(globally_aligned - reference, axis=1)
    result = {
        "tm_common": tm_score(predicted, reference),
        "gdt_common": gdt(global_distances),
        "rmsd_common": global_rmsd,
        "gdt_region_global_fit": gdt(global_distances[region]),
        "rmsd_region_global_fit": float(np.sqrt(np.mean(global_distances[region] ** 2))),
    }
    local_aligned, local_rmsd = superpose(predicted[region], reference[region])
    local_distances = np.linalg.norm(local_aligned - reference[region], axis=1)
    result.update({
        "gdt_region_local_fit": gdt(local_distances),
        "rmsd_region_local_fit": local_rmsd,
    })
    return result


def score_protein(pair_id: str, shard_index: int = 0,
                  shard_count: int = 1) -> tuple[list[dict], dict | None]:
    """Score one shard of a protein's structures while references stay in memory."""
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    target = truth.loc[pair_id]
    sequence = str(target.sequence)
    references = {}
    for fold in (1, 2):
        _, _, references[fold], _ = prepare_structure(str(target[f"fold{fold}"]), sequence)
    common = sorted(set(int(position) for position in target.common_positions)
                    & references[1].keys() & references[2].keys())
    lo, hi = int(target.fs_lo), int(target.fs_hi)
    region = np.array([lo <= position < hi for position in common])
    if len(common) < 20 or region.sum() < 3:
        raise ValueError(f"{pair_id}: insufficient common ({len(common)}) or region "
                         f"({int(region.sum())}) C-alpha positions")
    reference_arrays = {
        fold: np.array([references[fold][position] for position in common])
        for fold in (1, 2)
    }
    ref_metrics = comparison_metrics(reference_arrays[1], reference_arrays[2], region)
    separation = {"pair_id": pair_id, "n_common_ca": len(common),
                  "n_region_ca": int(region.sum()), **ref_metrics}
    rows = []
    structures = ROOT / "results" / "predictions" / "iid1000"
    paths = sorted(structures.glob(f"{pair_id}__*.pdb.gz"))[shard_index::shard_count]
    for path in paths:
        target_id = path.name.removesuffix(".pdb.gz")
        predicted_map = prediction_coords(path, sequence)
        positions = [position for position in common if position in predicted_map]
        if len(positions) != len(common):
            raise ValueError(f"{target_id}: {len(positions)}/{len(common)} common CAs")
        predicted = np.array([predicted_map[position] for position in common])
        row = {"target_id": target_id, "pair_id": pair_id,
               "n_common_ca": len(common), "n_region_ca": int(region.sum())}
        for fold in (1, 2):
            metrics = comparison_metrics(predicted, reference_arrays[fold], region)
            row.update({f"{name}_fold{fold}": value for name, value in metrics.items()})
        rows.append(row)
    return rows, separation if shard_index == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shards-per-protein", type=int, default=12)
    args = parser.parse_args()
    input_meta = pq.read_table(ROOT / "data" / "inputs.parquet").to_pandas()
    run = pd.read_csv(ROOT / "results" / "iid1000.csv")
    expected = set(input_meta.target_id)
    if set(run.target_id) != expected or len(run) != len(input_meta) or not run.status.eq("ok").all():
        raise ValueError("Helico iid run is incomplete")
    pair_ids = sorted(input_meta.pair_id.unique())
    all_rows, separation_by_pair = [], {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_protein, pair_id, shard_index, args.shards_per_protein):
                (pair_id, shard_index)
            for pair_id in pair_ids
            for shard_index in range(args.shards_per_protein)
        }
        for future in concurrent.futures.as_completed(futures):
            pair_id, shard_index = futures[future]
            rows, separation = future.result()
            all_rows.extend(rows)
            if separation is not None:
                separation_by_pair[pair_id] = separation
            print(f"scored {pair_id} shard {shard_index + 1}/{args.shards_per_protein}",
                  flush=True)
    scores = pd.DataFrame(all_rows)
    counts = scores.groupby("pair_id").size()
    if len(scores) != len(input_meta) or not counts.reindex(pair_ids).eq(1003).all():
        raise ValueError("expected 1,003 scored structures for each protein")
    if set(separation_by_pair) != set(pair_ids):
        raise ValueError("missing reference-separation scores")
    complete = input_meta.merge(run, on="target_id", validate="one_to_one", suffixes=("", "_helico"))
    complete = complete.merge(scores, on=["target_id", "pair_id"], validate="one_to_one")
    complete.sort_values(["pair_id", "rollout"]).to_parquet(
        ROOT / "scores.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(separation_by_pair.values()).sort_values("pair_id").to_csv(
        HERE / "data" / "helico_iid_reference_separation.csv", index=False
    )
    complete[complete.kind != "iid"].sort_values(["pair_id", "rollout"]).to_csv(
        HERE / "data" / "helico_iid_control_scores.csv", index=False
    )
    timings = complete[[
        "target_id", "pair_id", "kind", "rollout", "L", "n_contacts_helico",
        "n_contacts",
        "predict_seconds", "elapsed_seconds", "model_load_seconds", "gpu_name",
        "gpu_total_memory_gb", "gpu_compute_capability", "hostname", "platform",
        "torch_version",
    ]].rename(columns={
        "pair_id": "stem", "kind": "mode", "L": "n_residues",
        "n_contacts_helico": "n_pairs", "n_contacts": "n_pairs_requested",
        "predict_seconds": "elapsed_seconds",
        "elapsed_seconds": "target_seconds_after_model_load",
    })
    timings["total_seconds"] = (timings.target_seconds_after_model_load
                                + timings.model_load_seconds)
    timings["model_nickname"] = "helico-contacts-msafree-01-step-6000"
    timings["runner_tag"] = "modal"
    timings["n_samples"] = 1
    timings["n_cycles"] = 6
    protocol = json.loads((ROOT / "data" / "protocol.json").read_text())
    timings["timestamp_utc"] = protocol["run_started_utc"]
    timings.to_csv(HERE / "data" / "helico_iid_timings.csv", index=False)
    print(f"wrote {len(complete):,} cross-reference rows")


if __name__ == "__main__":
    main()
