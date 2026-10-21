"""Redistribute unfinished dose groups after the original workers have stopped.

The frozen inference bundle, prompts, random seeds, output keys, and 100-sample
budgets are unchanged. Counts for a protein are partitioned into disjoint jobs;
each worker verifies and skips already-complete conditions. This command never
cancels jobs and refuses submission while any original worker is active.
"""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import s3fs
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.resources.state import TERMINAL_JOB_STATES

from dispatch_spiking import IMAGE, bootstrap

HERE = Path(__file__).resolve().parent


def partition_remaining(plan: dict, root: str, completed: set[str], max_jobs: int) -> list[dict]:
    """Balance disjoint count groups while keeping each job on one protein."""
    remaining = []
    for target in plan["targets"]:
        counts = {}
        for repeat, grid in target["doses_by_repeat"].items():
            for count in sorted({dose["k"] for dose in grid}):
                for policy in (["random"] if count == 0 else ["random", "sse_balanced"]):
                    marker = f"{root}/units/{target['stem']}/{policy}-r{repeat}-k{count:04d}.complete.json"
                    if marker not in completed:
                        counts[count] = counts.get(count, 0) + 1
        if counts:
            remaining.append({"stem": target["stem"], "L": target["L"], "counts": counts, "parts": 1})
    if len(remaining) > max_jobs:
        raise ValueError("Too many unfinished proteins for a tail redistribution")
    while sum(row["parts"] for row in remaining) < max_jobs:
        eligible = [row for row in remaining if row["parts"] < len(row["counts"])]
        if not eligible:
            break
        selected = max(eligible, key=lambda row: row["L"] * sum(row["counts"].values()) / row["parts"])
        selected["parts"] += 1
    jobs = []
    for row in remaining:
        groups = [[] for _ in range(row["parts"])]
        loads = [0] * row["parts"]
        for count, n_conditions in sorted(row["counts"].items(), key=lambda item: (-item[1], item[0])):
            part = min(range(row["parts"]), key=loads.__getitem__)
            groups[part].append(count)
            loads[part] += n_conditions
        if sorted(count for group in groups for count in group) != sorted(row["counts"]):
            raise ValueError("Tail partition lost or duplicated a count")
        jobs.extend({"stem": row["stem"], "L": row["L"], "counts": sorted(group), "remaining_conditions": load}
                    for group, load in zip(groups, loads, strict=True))
    return jobs


def main() -> None:
    """Preview by default; submit only after old jobs have all terminated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 12:
        raise ValueError("The approved allocation is at most twelve H100s")
    original = json.loads((HERE / "data/jobs_full.json").read_text())
    plan = json.loads(args.plan.read_text())
    fs = s3fs.S3FileSystem(profile="cw", client_kwargs={"endpoint_url": "https://cwobject.com"},
                          config_kwargs={"s3": {"addressing_style": "virtual"}})
    root = original["out"].removeprefix("s3://")
    completed = {name for name in fs.find(root) if name.endswith(".complete.json")}
    assignments = partition_remaining(plan, root, completed, args.max_jobs)
    summary = {"n_jobs": len(assignments), "remaining_conditions": sum(row["remaining_conditions"] for row in assignments),
               "bundle_sha256": original["bundle_sha256"], "out": original["out"], "assignments": assignments}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.submit or not assignments:
        return
    digest = original["bundle_sha256"]
    uri = original["out"].rsplit("/", 1)[0] + f"/inputs/{digest}.tar.gz"
    prefix = original["job_ids"][0].rsplit("-s", 1)[0]
    with open_iris_client(config_file=args.iris_config, workspace=None) as iris:
        statuses = iris.list_jobs(prefix=prefix, limit=100)
        originals = [job for job in statuses if str(job.job_id) in original["job_ids"]]
        if len(originals) != len(original["job_ids"]) or any(job.state not in TERMINAL_JOB_STATES for job in originals):
            raise ValueError("Original jobs must be confirmed terminal before redistributing their output keys")
        client = FrayIrisClient.from_iris_client(iris)
        ids = []
        for assignment in assignments:
            counts = ",".join(map(str, assignment["counts"]))
            group_hash = hashlib.sha256(counts.encode()).hexdigest()[:8]
            command = bootstrap(0, 1, uri, digest, original["out"], False).rstrip()
            command += f" --stem {shlex.quote(assignment['stem'])} --counts {shlex.quote(counts)}\n"
            request = JobRequest(name=f"exp369-tail-{digest[:8]}-{assignment['stem'].lower()}-{group_hash}",
                                 entrypoint=Entrypoint.from_binary("bash", ["-lc", command]),
                                 resources=ResourceConfig.with_gpu("H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="128g"),
                                 environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
                                 replicas=1, processes_per_task=1, priority=3,
                                 max_retries_failure=0, max_retries_preemption=10)
            job = client.submit(request)
            ids.append(str(job.job_id))
            job_prefix = ids[0].rsplit("/", 1)[0] + f"/exp369-tail-{digest[:8]}"
            (HERE / "data/jobs_tail.json").write_text(json.dumps(summary | {"job_ids": ids, "job_prefix": job_prefix}, indent=2) + "\n")
            print(job.job_id, flush=True)


if __name__ == "__main__":
    main()
