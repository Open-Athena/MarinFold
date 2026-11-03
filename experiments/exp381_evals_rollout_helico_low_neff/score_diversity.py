"""Score the frozen selection experiment with the unchanged structural scorer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freeze_diversity import digest
from score import METRICS, score_one

HERE = Path(__file__).resolve().parent


def main(smoke: bool, maxmin: bool = False) -> None:
    """Verify all identities and atom coverage before aggregating accuracy."""
    build = json.loads((HERE / "data/tmscore_build.json").read_text())
    assert digest(HERE / "scratch/TMscore") == build["binary_sha256"]
    tag = "maxmin" if maxmin else "diversity"
    task_file = "maxmin_tasks.json" if maxmin else "tasks.json"
    count = 200 if maxmin else 400
    smoke_count = 2 if maxmin else 4
    plan = json.loads((HERE / f"data/{tag}_plan.json").read_text())
    tasks = json.loads((HERE / f"inputs/diversity/{task_file}").read_text())
    manifest = json.loads((HERE / f"data/{tag}_selection_manifest.json").read_text())
    assert digest(HERE / f"inputs/diversity/{task_file}") == manifest["tasks_sha256"]
    if smoke:
        stem = json.loads((HERE / "inputs/diversity/targets.json").read_text())[0][
            "stem"
        ]
        tasks = [t for t in tasks if t["stem"] == stem and t["rollout"] == 0]
    rows, timings, versions, runners = [], [], set(), set()
    for task in tasks:
        folder = HERE / "scratch/diversity/results" / task["task_id"]
        row = json.loads((folder / "result.json").read_text())
        assert row["input_sha256"] == task["input_sha256"]
        assert (
            row["plan_sha256"]
            == manifest["plan_sha256"]
            == digest(HERE / f"data/{tag}_plan.json")
        )
        assert row["selection_manifest_sha256"] == digest(
            HERE / f"data/{tag}_selection_manifest.json"
        )
        assert row["helico_source_sha"] == plan["helico_source_sha"]
        for name, expected in row["output_sha256"].items():
            assert digest(folder / name) == expected
        scalar = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        metrics = score_one(folder)
        assert (
            metrics["ca_coverage"] == 1
            and metrics["n_reference_ca"] == row["n_residues"]
        )
        assert np.isfinite(row["ranking_score"]) and all(
            np.isfinite(metrics[m]) for m in METRICS
        )
        rows.append({**scalar, **metrics})
        timings.append(scalar)
        versions.add(json.dumps(row["versions"], sort_keys=True))
        runners.add(row["runner_sha256"])
    assert len(rows) == (smoke_count if smoke else count)
    assert len(versions) == len(runners) == 1
    original = json.loads((HERE / "data/run_manifest.json").read_text())
    assert json.loads(next(iter(versions))) == original["runtime_versions"]
    record = dict(
        n_predictions=len(rows),
        plan_sha256=manifest["plan_sha256"],
        selection_manifest_sha256=digest(HERE / f"data/{tag}_selection_manifest.json"),
        minimum_ca_coverage=min(r["ca_coverage"] for r in rows),
        minimum_atom_coverage=min(r["atom_coverage"] for r in rows),
        runtime_versions=json.loads(next(iter(versions))),
        runner_sha256=next(iter(runners)),
        scoring_source_sha256=digest(HERE / "score.py"),
        tmscore_build=build,
        inference_seconds=sum(r["elapsed_seconds"] for r in rows),
        unique_worker_setup_seconds=sum(
            {r["worker_id"]: r["model_load_seconds"] for r in rows}.values()
        ),
    )
    filename = f"{tag}_helico_smoke.json" if smoke else f"{tag}_run_manifest.json"
    (HERE / "data" / filename).write_text(json.dumps(record, indent=2) + "\n")
    if not smoke:
        pd.DataFrame(rows).to_csv(HERE / f"data/{tag}_metrics.csv", index=False)
        pd.DataFrame(timings).to_csv(
            HERE / f"data/{tag}_helico_timings.csv", index=False
        )
    print(
        f"Scored {len(rows)} structures; minimum CA coverage {record['minimum_ca_coverage']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--maxmin", action="store_true")
    args = parser.parse_args()
    main(args.smoke, args.maxmin)
