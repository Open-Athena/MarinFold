"""Execute the fixed checkpoint-evaluation schedule inside a persistent Iris job.

This driver dispatches a fixed number of co-located GPU shards for each committed
request, waits for every child, and fails loudly on incomplete measurements. It
does not make training placement, recovery, or convergence decisions.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import fsspec
from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)
from iris.client.client import get_iris_ctx

import wandb
from aggregate_eval import aggregate, read_json, write_json
from common import ROOT

IMAGE = "vllm/vllm-openai@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40"


def evaluate_request(request: dict, shards: int, reference_e8: bool) -> dict:
    """Evaluate exactly one immutable export and join its complete target universe."""
    output = request.get(
        "output", f"{ROOT}/eval-val/{request['run_id']}/step-{request['step']}"
    )
    client = current_client()
    identity = hashlib.sha256((request["checkpoint"] + output).encode()).hexdigest()[
        :12
    ]
    jobs = []
    for shard in range(shards):
        command = [
            "bootstrap.sh",
            "--checkpoint",
            request["checkpoint"],
            "--format",
            request["format"],
            "--out",
            output,
            "--shard",
            str(shard),
            "--shards",
            str(shards),
        ]
        if reference_e8:
            command += ["--reference-e8", "--max-model-len", "8192"]
        child = client.submit(
            JobRequest(
                name=f"exp347-eval-{identity}-s{shard:02d}",
                entrypoint=Entrypoint.from_binary("bash", command),
                resources=ResourceConfig.with_gpu(
                    "H100",
                    count=1,
                    cpu=8,
                    ram="64Gi",
                    disk="128Gi",
                    image=IMAGE,
                ),
                environment=create_environment(docker_image=IMAGE, setup_scripts=[]),
                priority=3,
                max_retries_failure=0,
                max_retries_preemption=100,
            ),
            adopt_existing=True,
        )
        jobs.append(child)
    write_json(
        output + "/dispatch.json",
        {
            "request": request,
            "jobs": [j.job_id for j in jobs],
            "shards": shards,
            "image": IMAGE,
        },
    )
    failures = []
    for job in jobs:
        status = job.wait(raise_on_failure=False)
        if status != JobStatus.SUCCEEDED:
            failures.append(
                {
                    "job": job.job_id,
                    "status": str(status),
                    "logs": list(job.logs(max_lines=50)),
                }
            )
    if failures:
        write_json(output + "/driver_failure.json", {"failures": failures})
        raise RuntimeError(f"Incomplete evaluation: {failures}")
    return {
        **aggregate(
            output, Path("eval_val.jsonl"), request["checkpoint"], request["format"]
        ),
        "step": request["step"],
        "tokens": request["tokens"],
        "training_run": request["run_id"],
        "final": request["final"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--evaluation-run", required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--reference-e8", action="store_true")
    args = parser.parse_args()
    # Off-cluster current_client() silently selects LocalClient. Never let a
    # monitoring process accidentally try to execute GPU workers on a laptop.
    if get_iris_ctx() is None:
        raise RuntimeError("The evaluation driver must run inside Iris")
    if not 1 <= args.shards <= 12:
        raise ValueError("Expected 1-12 explicitly reserved evaluation GPUs")
    fs, _ = fsspec.core.url_to_fs(ROOT)
    with wandb.init(
        entity="open-athena",
        project="MarinFold",
        id=args.evaluation_run,
        name=args.evaluation_run,
        resume="must",
        job_type="eval",
        group="exp347-qwen-base-contacts",
    ) as run:
        run.define_metric("checkpoint/tokens")
        run.define_metric("eval_val/*", step_metric="checkpoint/tokens")
        completed = set()
        while True:
            if args.request_file:
                requests = [json.loads(args.request_file.read_text())]
            else:
                paths = fs.glob(f"{ROOT}/eval_requests/{args.training_run}/step-*.json")
                requests = sorted(
                    (read_json("s3://" + p) for p in paths), key=lambda r: r["tokens"]
                )
            for request in requests:
                if request["run_id"] != args.training_run:
                    raise ValueError("Request belongs to another training run")
                if request["step"] in completed:
                    continue
                result_uri = f"{ROOT}/eval_results/{args.training_run}/step-{request['step']}.json"
                if fs.exists(result_uri):
                    result = read_json(result_uri)
                    if (
                        not result["complete"]
                        or result["checkpoint"] != request["checkpoint"]
                    ):
                        raise ValueError("Inconsistent committed evaluation result")
                else:
                    result = evaluate_request(request, args.shards, args.reference_e8)
                    write_json(result_uri, result)
                run.log(
                    {
                        "checkpoint/tokens": result["tokens"],
                        "checkpoint/step": result["step"],
                        "eval_val/n_units": result["n_units"],
                        **{
                            f"eval_val/{k}_r_precision": v
                            for k, v in result["r_precision"].items()
                        },
                    }
                )
                run.summary["latest_result"] = result
                print(json.dumps(result), flush=True)
                completed.add(request["step"])
                if request["final"]:
                    return
            if args.request_file:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
