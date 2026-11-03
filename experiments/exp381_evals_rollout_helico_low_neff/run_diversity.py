"""Run the frozen 400 contact-selection predictions using the original Helico API."""

import json
from pathlib import Path

from run_helico import Predictor, app, preflight, sha256

HERE = Path(__file__).resolve().parent


@app.local_entrypoint()
def diversity(smoke: bool = False) -> None:
    """Fold selected contacts with matched original per-slot Helico seeds."""
    provenance = preflight(4 if smoke else 400, 1 if smoke else 8)
    selection = json.loads(
        (HERE / "data/diversity_selection_manifest.json").read_text()
    )
    task_path = HERE / "inputs/diversity/tasks.json"
    if (
        sha256(task_path) != selection["tasks_sha256"]
        or sha256(HERE / "data/diversity_plan.json") != selection["plan_sha256"]
    ):
        raise ValueError("Frozen selection inputs changed")
    publication = json.loads(
        (HERE / "data/diversity_selection_publication.json").read_text()
    )
    for relative, expected in publication["files_sha256"].items():
        if sha256(HERE / relative) != expected:
            raise ValueError("Published selection differs from local inputs")
    provenance.update(
        original_plan_sha256=provenance["plan_sha256"],
        plan_sha256=selection["plan_sha256"],
        selection_manifest_sha256=sha256(
            HERE / "data/diversity_selection_manifest.json"
        ),
        driver_sha256=sha256(__file__),
    )
    targets = json.loads((HERE / "inputs/diversity/targets.json").read_text())
    tasks = json.loads(task_path.read_text())
    if smoke:
        targets = targets[:1]
        tasks = [
            t for t in tasks if t["stem"] == targets[0]["stem"] and t["rollout"] == 0
        ]
    elif not (HERE / "data/diversity_helico_smoke.json").exists():
        raise ValueError("Score the four-prediction smoke gate first")
    output = HERE / "scratch/diversity/results"
    pending = [t for t in tasks if not (output / t["task_id"] / "result.json").exists()]
    # Small five-slot batches return partial results promptly, also balancing
    # long targets across the eight persistent GPU workers.
    jobs = [
        (
            target,
            [
                t
                for t in pending
                if t["stem"] == target["stem"] and t["rollout"] // 5 == batch
            ],
            provenance,
            "v3-diversity",
        )
        for target in sorted(targets, key=lambda t: -t["L_helico"])
        for batch in range(2)
    ]
    failures = []
    for result in Predictor.with_options(
        max_containers=1 if smoke else 8
    )().predict.starmap(
        [j for j in jobs if j[1]], order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, Exception):
            failures.append(repr(result))
            continue
        for relative, content in result["files"].items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(
            f"Downloaded {result['stem']}: {len(result['rows'])} predictions",
            flush=True,
        )
    if failures:
        raise RuntimeError(f"Failed tasks (durable partials retained): {failures}")
    for task in tasks:
        row = json.loads((output / task["task_id"] / "result.json").read_text())
        if (
            row["input_sha256"] != task["input_sha256"]
            or row["plan_sha256"] != provenance["plan_sha256"]
        ):
            raise ValueError("Saved result input mismatch")
    print("Completed", len(tasks), "diversity predictions")
