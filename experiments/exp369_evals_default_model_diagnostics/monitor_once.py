"""Read one compact Iris/S3 progress snapshot without submitting or changing jobs."""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import s3fs
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Record job states and durable complete-condition counts for one launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=HERE / "data/jobs_full.json")
    parser.add_argument("--iris-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    parser.add_argument("--history", type=Path, default=Path("/data/exp369/progress.jsonl"))
    args = parser.parse_args()
    launch = json.loads(args.jobs.read_text())
    plan = json.loads(args.plan.read_text())
    expected = {target["stem"]: sum(1 + 2 * len({dose["k"] for dose in grid if dose["k"]})
                                    for grid in target["doses_by_repeat"].values()) for target in plan["targets"]}
    fs = s3fs.S3FileSystem(profile="cw", client_kwargs={"endpoint_url": "https://cwobject.com"},
                          config_kwargs={"s3": {"addressing_style": "virtual"}})
    objects = fs.find(launch["out"].removeprefix("s3://"), detail=True)
    complete = Counter(name.split("/")[-2] for name in objects if name.endswith(".complete.json"))
    prefix = launch.get("job_prefix", launch["job_ids"][0].rsplit("-s", 1)[0])
    with open_iris_client(config_file=args.iris_config, workspace=None) as client:
        jobs = client.list_jobs(prefix=prefix, limit=100)
    states = {str(job.job_id): {"state": job.state.value, "error": job.error_message, "pending_reason": job.pending_reason}
              for job in jobs if str(job.job_id) in launch["job_ids"]}
    snapshot = {"timestamp_utc": datetime.now(UTC).isoformat(), "complete_conditions": sum(complete.values()),
                "expected_conditions": sum(expected.values()), "complete_proteins": sum(complete[stem] == n for stem, n in expected.items()),
                "expected_proteins": len(expected), "artifact_bytes": sum(item["size"] for item in objects.values()),
                "job_state_counts": dict(Counter(job["state"] for job in states.values())),
                "jobs": states, "conditions_by_protein": dict(complete)}
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a") as stream:
        stream.write(json.dumps(snapshot) + "\n")
    print(json.dumps({key: value for key, value in snapshot.items() if key not in ("jobs", "conditions_by_protein")}, indent=2))
    for name, status in states.items():
        if status["state"] not in ("running", "succeeded", "pending", "building"):
            print(json.dumps({"job": name, **status}))


if __name__ == "__main__":
    main()
