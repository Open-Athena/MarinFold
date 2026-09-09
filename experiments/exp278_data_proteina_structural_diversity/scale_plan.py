"""Freeze length/class quotas and balanced independent-worker assignments."""

import argparse
import csv
import heapq
import json
import math
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ["unconditional", "1.x.x.x", "2.x.x.x", "3.x.x.x"]


def make_plan(run_id: str, workers: int, target: int, case_seconds: int = 3600) -> dict:
    with (HERE / "data/cost-by-case.csv").open() as handle:
        measurements = list(csv.DictReader(handle))
    cases = []
    expected = 0
    strata = [
        (length, condition) for length in range(60, 501) for condition in CONDITIONS
    ]
    for stratum, (length, condition) in enumerate(strata):
        points = sorted(
            [row for row in measurements if row["condition"] == condition],
            key=lambda row: int(row["length"]),
        )
        xs = [int(row["length"]) for row in points]
        inverse_yield = float(
            np.interp(
                length, xs, [1 / float(row["retained_fraction"]) for row in points]
            )
        )
        seconds = float(
            np.interp(
                length,
                xs,
                [float(row["gpu_seconds_per_raw_candidate"]) for row in points],
            )
        )
        quota = target // len(strata) + (stratum < target % len(strata))
        batch = 32 if length <= 200 else 16
        batches = math.ceil(quota * inverse_yield / batch)
        batches_per_case = max(1, int(case_seconds / (seconds * batch)))
        expected += batches * batch / inverse_yield
        for start in range(0, batches, batches_per_case):
            n_batches = min(batches - start, batches_per_case)
            index = len(cases)
            cases.append(
                {
                    "id": f"c{index:06d}-l{length}-{condition}",
                    "length": length,
                    "condition": condition,
                    "model": "short" if length <= 250 else "long",
                    "noise": 0.45 if length <= 250 else 0.35,
                    "batch_size": batch,
                    "batches": n_batches,
                    "samples": n_batches * batch,
                    "seed": 1000000000 + index * 10000,
                    "estimated_gpu_seconds": n_batches * batch * seconds,
                    "pilot_inverse_yield": inverse_yield,
                }
            )
    if sum(case["samples"] for case in cases) > 6_000_000:
        raise ValueError("Plan exceeds the six-million-raw-candidate bound")
    # Greedy load balancing, followed by a reproducible within-worker shuffle,
    # prevents all workers starting with the same length or the same class.
    queue = [(0.0, worker) for worker in range(workers)]
    heapq.heapify(queue)
    assignments = [[] for _ in range(workers)]
    for case in sorted(
        cases, key=lambda row: row["estimated_gpu_seconds"], reverse=True
    ):
        load, worker = heapq.heappop(queue)
        assignments[worker].append({**case, "worker": worker})
        heapq.heappush(queue, (load + case["estimated_gpu_seconds"], worker))
    for worker, assigned in enumerate(assignments):
        random.Random(278 + worker).shuffle(assigned)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "workers": workers,
        "output": f"s3://marin-us-east-02a/MarinFold/exp278-proteina/{run_id}",
        "target_documents": target,
        "expected_retained_at_pilot_yield": expected,
        "raw_candidates": sum(case["samples"] for case in cases),
        "estimated_pure_gpu_hours": sum(case["estimated_gpu_seconds"] for case in cases)
        / 3600,
        "length_target": "uniform accepted lengths 60..500",
        "class_target": "equal expected accepted contributions from four requested arms",
        "precision": "reference float32 matmuls",
        "attempts_per_backbone": 1,
        "esm_revision": "75a3841ee059df2bf4d56688166c8fb459ddd97a",
        "checkpoint_boundary": "short through 250 aa, long from 251; validate crossover canaries before bulk",
        "cases": [case for assigned in assignments for case in assigned],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=768)
    parser.add_argument("--target", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.run_id, args.workers, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in plan.items() if key != "cases"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
