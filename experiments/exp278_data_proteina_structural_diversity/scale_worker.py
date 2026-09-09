"""Execute one independent, preemptible GPU worker's durable case queue.

Each Iris root job owns exactly one worker index. Cases, backbone batches and
sequence attempts have immutable identities. Restarting a worker reuses saved
backbones and designed sequences and skips completed cases; no gang exists.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import fsspec

from scale_common import paused, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    root = manifest["output"]
    control = root + "/control.json"
    fs, path = fsspec.core.url_to_fs(root)
    cases = [case for case in manifest["cases"] if case["worker"] == args.worker]
    if not cases:
        raise ValueError("Worker has no assigned cases")
    status_uri = f"{root}/workers/{args.worker:04d}.json"
    completed_raw = completed_folded = completed_quality = completed_cases = 0

    def status(phase: str, case: dict | None) -> None:
        write_json(
            status_uri,
            {
                "worker": args.worker,
                "phase": phase,
                "case": case,
                "completed_raw": completed_raw,
                "completed_folded": completed_folded,
                "completed_quality": completed_quality,
                "completed_cases": completed_cases,
                "planned_cases": len(cases),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    for case in cases:
        if paused(control):
            status("paused", case)
            return
        generated = f"{root}/cases/{case['id']}/generated"
        folded = f"{root}/cases/{case['id']}/folded"
        if not fs.exists(generated.removeprefix("s3://") + "/complete.json"):
            status("sampling", case)
            command = [
                sys.executable,
                str(Path(__file__).with_name("sample_worker.py")),
                "--model",
                case["model"],
                "--length",
                str(case["length"]),
                "--batch-size",
                str(case["batch_size"]),
                "--batches",
                str(case["batches"]),
                "--seed",
                str(case["seed"]),
                "--cath",
                case["condition"],
                "--noise",
                str(case["noise"]),
                "--compile",
                "--output",
                generated,
                "--control",
                control,
            ]
            subprocess.run(command, check=True)
            if paused(control):
                status("paused", case)
                return
        generated_summary = read_json(generated + "/complete.json")
        if generated_summary["count"] != case["samples"]:
            raise ValueError("Completed generation count differs from manifest")
        if not fs.exists(folded.removeprefix("s3://") + "/complete.json"):
            status("refolding", case)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("fold_worker.py")),
                    "--input",
                    generated,
                    "--output",
                    folded,
                    "--esm-revision",
                    manifest["esm_revision"],
                    "--seed",
                    "278",
                    "--control",
                    control,
                ],
                check=True,
            )
            if paused(control):
                status("paused", case)
                return
        folded_summary = read_json(folded + "/complete.json")
        if folded_summary["candidates"] != case["samples"]:
            raise ValueError("Completed folding count differs from manifest")
        completed_raw += generated_summary["count"]
        completed_folded += folded_summary["candidates"]
        completed_quality += folded_summary["quality_pass"]
        completed_cases += 1
        status("case_complete", case)
        print(
            json.dumps(
                {
                    "event": "case_complete",
                    "case": case["id"],
                    "completed_cases": completed_cases,
                }
            ),
            flush=True,
        )
    status("complete", None)


if __name__ == "__main__":
    main()
