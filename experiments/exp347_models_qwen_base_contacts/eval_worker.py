"""One resumable GPU shard of the Qwen eval-val rollout diagnostic.

All 100 samples must terminate before metrics are committed. Each protein gets
its own inference timer, vote matrix, raw completions, and atomic completion
marker. A failed/capped evaluation preserves diagnostics and cannot masquerade
as a complete R-precision measurement.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import FORMATS
from eval_contract import (
    N_ROLLOUTS,
    contact_votes,
    generation_budget,
    load_metric_reference,
    rollout_prompt,
    score_votes,
)


def write_json(uri: str, value: dict | list) -> None:
    """Write one small JSON artifact through the injected object-store filesystem."""
    with fsspec.open(uri, "w") as handle:
        json.dump(value, handle)


def stage_model(uri: str, destination: Path) -> dict:
    """Stage a co-located export and record its complete file identity."""
    if not uri.startswith("s3://marin-us-east-02a/"):
        raise ValueError(
            "Evaluation weights must be in the co-located CoreWeave bucket"
        )
    fs, path = fsspec.core.url_to_fs(uri)
    destination.mkdir(parents=True, exist_ok=True)
    files = fs.find(path, detail=True)
    identity = []
    for remote, info in sorted(files.items()):
        relative = remote.removeprefix(path + "/")
        local = destination / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(remote, str(local))
        if local.stat().st_size != info["size"]:
            raise ValueError(f"Incomplete model staging: {remote}")
        identity.append(
            {"name": relative, "size": info["size"], "etag": info.get("ETag")}
        )
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        if not (destination / name).exists():
            raise ValueError(f"Incomplete model export: missing {name}")
    return {"uri": uri, "files": identity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metric-reference", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=347)
    parser.add_argument("--reference-e8", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.targets.read_text().splitlines()]
    expected_count, expected_set = (
        (554, "legacy-e8-reference") if args.reference_e8 else (97, "eval-val")
    )
    if len(records) != expected_count or any(
        r["eval_set"] != expected_set for r in records
    ):
        raise ValueError(f"Expected exactly {expected_count} {expected_set} units")
    if len({(r["dataset"], r["stem"]) for r in records}) != expected_count:
        raise ValueError("Duplicate evaluation unit")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard assignment")
    records.sort(key=lambda r: (r["L"], r["dataset"], r["stem"]))
    assigned = [r for i, r in enumerate(records) if i % args.shards == args.shard]
    if args.limit:
        assigned = assigned[: args.limit]
    fs, _ = fsspec.core.url_to_fs(args.out)
    pending = []
    for record in assigned:
        unit = f"{record['dataset']}__{record['stem']}"
        marker = f"{args.out}/{unit}/complete.json"
        if not fs.exists(marker):
            pending.append(record)
            continue
        with fsspec.open(marker) as handle:
            completed = json.load(handle)
        if (
            completed["checkpoint"] != args.checkpoint
            or completed["document_format"] != args.format
        ):
            raise ValueError(f"Evaluation output identity changed: {marker}")
    if not pending:
        return
    started = time.perf_counter()
    directory = Path("/tmp/exp347-eval-model")
    identity = stage_model(args.checkpoint, directory)
    model_stage_seconds = time.perf_counter() - started
    reference = load_metric_reference(args.metric_reference)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    load_start = time.perf_counter()
    model = LLM(
        model=str(directory),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=100,
        max_num_batched_tokens=4096,
        seed=args.seed,
    )
    load_seconds = time.perf_counter() - load_start
    props = torch.cuda.get_device_properties(0)
    provenance = {
        "checkpoint": identity,
        "targets_sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
        "metric_sha256": hashlib.sha256(args.metric_reference.read_bytes()).hexdigest(),
        "document_format": args.format,
        "recipe": "100 rollouts; T=1; top_p=.95; top_k=-1; live-contact frequency",
        "resampling": "canonical n-terminus/order"
        if args.format == "contacts_v1"
        else "fixed ordinary sequence; independent samples",
        "budget": "6L+128 domain-token capacity translated to native tokenizer; see eval_contract.py",
        "iris_job_id": os.environ.get("IRIS_JOB_ID"),
        "max_model_len": args.max_model_len,
    }
    write_json(f"{args.out}/provenance-shard-{args.shard}.json", provenance)
    for record in pending:
        unit_start = time.perf_counter()
        pairs = [rollout_prompt(record, args.format, i) for i in range(N_ROLLOUTS)]
        prompts, starts = zip(*pairs, strict=True)
        budget = generation_budget(tokenizer, record["L"], args.format)
        lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        if args.reference_e8:
            budget = min(budget, args.max_model_len - max(lengths))
        if budget <= 0 or max(lengths) + budget > args.max_model_len:
            raise ValueError(
                f"Insufficient context for {record['stem']}: {max(lengths)}+{budget}"
            )
        seed = int(
            hashlib.sha256(
                f"{args.seed}:{record['dataset']}:{record['stem']}".encode()
            ).hexdigest()[:8],
            16,
        )
        stop = "<end>" if args.format == "contacts_v1" else "\nEND"
        parameters = [
            SamplingParams(
                temperature=1.0,
                top_p=0.95,
                top_k=-1,
                max_tokens=budget,
                stop=[stop],
                include_stop_str_in_output=True,
                skip_special_tokens=False,
                seed=(seed + i) % (2**31),
            )
            for i in range(N_ROLLOUTS)
        ]
        inference_start = time.perf_counter()
        outputs = model.generate(list(prompts), parameters, use_tqdm=False)
        elapsed = time.perf_counter() - inference_start
        if len(outputs) != N_ROLLOUTS:
            raise ValueError("Sampling did not return all requested rollouts")
        completions = [o.outputs[0] for o in outputs]
        unfinished = sum(o.finish_reason != "stop" for o in completions)
        missing_format_end = sum(stop not in o.text for o in completions)
        unit = f"{args.out}/{record['dataset']}__{record['stem']}"
        raw = "".join(
            json.dumps(
                {
                    "rollout": i,
                    "start": starts[i],
                    "prompt": prompts[i],
                    "text": o.text,
                    "finish_reason": o.finish_reason,
                    "tokens": len(o.token_ids),
                }
            )
            + "\n"
            for i, o in enumerate(completions)
        ).encode()
        with fsspec.open(unit + "/completions.jsonl.gz", "wb") as handle:
            handle.write(gzip.compress(raw))
        votes = contact_votes(
            [o.text for o in completions], list(starts), record["L"], args.format
        )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, score=votes)
        with fsspec.open(unit + "/votes.npz", "wb") as handle:
            handle.write(buffer.getvalue())
        timing = {
            "dataset": record["dataset"],
            "stem": record["stem"],
            "n_residues": record["L"],
            "n_pairs": max(record["L"] - 6, 0) * max(record["L"] - 5, 0) // 2,
            "mode": f"{args.format}-rollout100",
            "elapsed_seconds": elapsed,
            "model_load_seconds": load_seconds,
            "model_stage_seconds": model_stage_seconds,
            "total_seconds": model_stage_seconds
            + load_seconds
            + time.perf_counter()
            - unit_start,
            "model_nickname": args.checkpoint,
            "runner_tag": "iris-coreweave",
            "gpu_name": props.name,
            "gpu_total_memory_gb": props.total_memory / 2**30,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "n_rollouts": N_ROLLOUTS,
            "unfinished_rollouts": unfinished,
            "missing_format_end": missing_format_end,
            "generated_tokens": sum(len(o.token_ids) for o in completions),
            "prompt_tokens_max": max(lengths),
            "max_tokens": budget,
            "max_model_len": args.max_model_len,
        }
        write_json(unit + "/timings.json", timing)
        if unfinished:
            write_json(unit + "/failure.json", timing)
            raise RuntimeError(
                f"{record['stem']}: {unfinished}/100 rollouts hit their cap; outputs preserved"
            )
        metrics = score_votes(votes, record, reference)
        write_json(unit + "/metrics.json", metrics)
        write_json(
            unit + "/complete.json",
            {
                "checkpoint": args.checkpoint,
                "stem": record["stem"],
                "n_rollouts": N_ROLLOUTS,
                "document_format": args.format,
                "tokens": timing["generated_tokens"],
                "unfinished_rollouts": 0,
                "metrics": metrics,
                "timing": timing,
            },
        )
        print(
            json.dumps(
                {
                    "stem": record["stem"],
                    "elapsed": elapsed,
                    "R": {
                        m["range"]: m["precision"] for m in metrics if m["cut"] == "R"
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
