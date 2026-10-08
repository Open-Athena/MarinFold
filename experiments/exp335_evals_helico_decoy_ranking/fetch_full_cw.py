"""Validate and consolidate completed CoreWeave result parts."""

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
from pathlib import Path

from full_worker_cw import RUN_FINGERPRINT, assign_tasks, output_uris
from stage_full_cw import DEFAULT_KUBECONFIG, coreweave_s3


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write non-empty dictionaries with stable Unix line endings."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_expected_parts(
    input_dir: Path, tasks: list[dict], *, allow_remote_only_inputs: bool
) -> list[tuple[dict, list[str] | None]]:
    """Load expected candidate IDs while parsing each target payload only once."""
    candidate_ids_by_path: dict[str, list[str]] = {}
    expected = []
    for task in tasks:
        relative_path = task["relative_path"]
        input_path = input_dir / relative_path
        if not input_path.is_file():
            if not allow_remote_only_inputs:
                raise FileNotFoundError(input_path)
            expected.append((task, None))
            continue
        if relative_path not in candidate_ids_by_path:
            with gzip.open(input_path, "rt") as stream:
                payload = json.load(stream)
            candidate_ids_by_path[relative_path] = [
                candidate["decoy_id"] for candidate in payload["candidates"]
            ]
        expected.append(
            (
                task,
                candidate_ids_by_path[relative_path][task["start"] : task["end"]],
            )
        )
    return expected


def fetch_part(
    fs, task: dict, candidate_ids: list[str] | None
) -> tuple[list[dict], list[dict]]:
    """Fetch and fully validate a metrics/timing object pair."""
    metrics_uri, timing_uri = output_uris(task)
    metrics_path = metrics_uri.removeprefix("s3://")
    timing_path = timing_uri.removeprefix("s3://")
    compressed = fs.cat_file(metrics_path)
    metrics_sha256 = hashlib.sha256(compressed).hexdigest()
    payload = json.loads(gzip.decompress(compressed))
    timing = json.loads(fs.cat_file(timing_path))
    observed_candidate_ids = payload.get("candidate_ids")
    if not isinstance(observed_candidate_ids, list):
        raise ValueError(f"{metrics_uri}: missing candidate IDs")
    if candidate_ids is None:
        candidate_ids = observed_candidate_ids
    if len(candidate_ids) != task["end"] - task["start"]:
        raise ValueError(f"{metrics_uri}: candidate count mismatch")
    expected = {
        "run_fingerprint": RUN_FINGERPRINT,
        "target_input_sha256": task["input_sha256"],
        "candidate_ids": candidate_ids,
    }
    for key, value in expected.items():
        if payload.get(key) != value or timing.get(key) != value:
            raise ValueError(
                f"{task['target']} {task['start']}:{task['end']}: {key} mismatch"
            )
    if timing["metrics_sha256"] != metrics_sha256:
        raise ValueError(f"{metrics_uri}: metrics digest mismatch")
    results = payload["results"]
    if [result["decoy_id"] for result in results] != candidate_ids:
        raise ValueError(f"{metrics_uri}: result order mismatch")
    if len(timing["timings"]) != len(results):
        raise ValueError(f"{timing_uri}: timing count mismatch")
    return results, timing["timings"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("scratch/full_inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/full_results"))
    parser.add_argument("--num-shards", type=int, default=96)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--kubeconfig", type=Path, default=DEFAULT_KUBECONFIG)
    parser.add_argument(
        "--allow-remote-only-inputs",
        action="store_true",
        help="derive candidate IDs from fingerprinted result objects when local target payloads are absent",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch all expected parts and write complete local result tables."""
    args = parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    tasks = [
        task for shard in assign_tasks(manifest, args.num_shards) for task in shard
    ]
    expected = load_expected_parts(
        args.input_dir,
        tasks,
        allow_remote_only_inputs=args.allow_remote_only_inputs,
    )
    fs = coreweave_s3(args.kubeconfig)
    result_parts = []
    timing_parts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_part, fs, task, candidate_ids): (task, candidate_ids)
            for task, candidate_ids in expected
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results, timings = future.result()
            result_parts.extend(results)
            timing_parts.extend(timings)
            if completed % 100 == 0 or completed == len(futures):
                print(f"fetched {completed}/{len(futures)} result parts", flush=True)

    candidate_rows = []
    sample_rows = []
    for result in result_parts:
        samples = result.pop("samples")
        result.pop("timing")
        candidate_rows.append(result)
        sample_rows.extend(
            {
                "target": result["target"],
                "decoy_id": result["decoy_id"],
                "map_mode": result["map_mode"],
                **sample,
            }
            for sample in samples
        )
    candidate_rows.sort(key=lambda row: (row["target"], row["decoy_id"]))
    sample_rows.sort(
        key=lambda row: (row["target"], row["decoy_id"], row["sample_idx"])
    )
    timing_parts.sort(key=lambda row: row["stem"])
    keys = [(row["target"], row["decoy_id"]) for row in candidate_rows]
    if len(keys) != manifest["n_candidates"] or len(keys) != len(set(keys)):
        raise ValueError(
            f"expected {manifest['n_candidates']} unique candidates, found {len(keys)}"
        )
    if len(sample_rows) != 3 * len(candidate_rows) or len(timing_parts) != len(
        candidate_rows
    ):
        raise ValueError("candidate/sample/timing coverage mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "candidate_metrics.csv", candidate_rows)
    write_csv(args.output_dir / "sample_metrics.csv", sample_rows)
    write_csv(args.output_dir / "timings.csv", timing_parts)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_fingerprint": RUN_FINGERPRINT,
                "n_parts": len(tasks),
                "n_candidates": len(candidate_rows),
                "n_samples": len(sample_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"wrote {len(candidate_rows)} candidates, {len(sample_rows)} samples, "
        f"and {len(timing_parts)} timings to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
