"""Extend the historical randomized pools without changing their generation recipe.

The source bundle includes exp373's frozen worker and contact parser: this is a
replay of that experiment's generator, with a different target list and range.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)
from marinfold.inference._tokenizer import model_source_path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from contacts import parse_contacts, sample_seed
from worker import generate, publish, stage

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Generate indices 100..999, durably recording timing and stop status."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["default", "mpnn2-lr5e-4"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE / "data/diversity_plan.json").read_text())
    targets = sorted(
        json.loads((HERE / "inputs/diversity/targets.json").read_text()),
        key=lambda t: (t["L"], t["stem"]),
    )
    targets = (
        targets[-1:]
        if args.smoke
        else [t for i, t in enumerate(targets) if i % args.shards == args.shard]
    )
    root = f"{plan['artifact_root']}/raw/{args.model}"
    pending = []
    for target in targets:
        uri = f"{root}/{target['stem']}-t1.0-randomized.complete.json"
        result = subprocess.run(
            [
                "/tmp/exp373-hf/bin/python",
                str(HERE / "publish_file.py"),
                "--exists",
                uri,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != "true":
            pending.append(target)
    if not pending:
        return
    spec = plan["models"][args.model]
    started = time.monotonic()
    model_dir = Path("/tmp/exp381-model")
    verified = stage(spec, model_dir)
    stage_seconds = time.monotonic() - started
    started = time.monotonic()
    effective = model_source_path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(effective)
    config = json.loads((Path(effective) / "config.json").read_text())
    assert (
        config["rope_theta"] == 500000
        and config["rope_scaling"]["rope_type"] == "llama3"
    )
    assert len(tokenizer) == config["vocab_size"] == 2845
    model = LLM(
        model=str(effective),
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=128,
        seed=373,
    )
    model_load = time.monotonic() - started
    device = torch.cuda.get_device_properties(0)
    hardware = dict(
        gpu_name=device.name,
        gpu_total_memory_gb=device.total_memory / 2**30,
        gpu_compute_capability=f"{device.major}.{device.minor}",
        hostname=socket.gethostname(),
        platform=platform.platform(),
        torch_version=str(torch.__version__),
        runner_tag="iris",
        model_nickname=spec["nickname"],
        checkpoint_step=spec["step"],
        model_load_seconds=model_load,
        model_stage_seconds=stage_seconds,
        job_id=os.environ.get("IRIS_JOB_ID", ""),
    )
    publish(
        dict(
            model=spec,
            verified_files=verified,
            hardware=hardware,
            effective_config=config,
            runtime_versions={
                p: importlib.metadata.version(p)
                for p in ["vllm", "torch", "transformers", "tokenizers", "marinfold"]
            },
        ),
        f"{root}/worker-{'smoke' if args.smoke else args.shard}.json",
    )
    for target in pending:
        started = time.monotonic()
        indices = list(range(plan["rollout_start"], plan["rollout_stop"]))
        documents = [
            build_document(
                f"{target['stem']}:r{k}",
                residues_from_sequence(target["input_seq"]),
                [],
                config=GenerationConfig(),
            )
            for k in indices
        ]
        prompts = [
            d.document.split("<begin_statements>")[0] + "<begin_statements>"
            for d in documents
        ]
        params = [
            SamplingParams(
                temperature=1.0,
                top_p=0.95,
                top_k=-1,
                max_tokens=min(
                    8192 - len(tokenizer(p, add_special_tokens=False).input_ids),
                    6 * target["L"] + 128,
                ),
                stop_token_ids=[tokenizer.convert_tokens_to_ids("<end>")],
                skip_special_tokens=False,
                seed=sample_seed(target["stem"], "randomized", k // 100, k % 100),
            )
            for k, p in zip(indices, prompts, strict=True)
        ]
        infer_started = time.monotonic()
        records = generate(model, tokenizer, prompts, params)
        elapsed = time.monotonic() - infer_started
        for k, record, document in zip(indices, records, documents, strict=True):
            record.update(
                rollout=k,
                block=k // 100,
                prompt_id=f"{target['stem']}:r{k}",
                offset=document.n_term_index,
                **parse_contacts(record["text"], document.n_term_index, target["L"]),
            )
        timing = dict(
            **hardware,
            stem=target["stem"],
            n_residues=target["L"],
            n_pairs=max(0, (target["L"] - 6) * (target["L"] - 5) // 2),
            mode="randomized",
            temperature=1.0,
            elapsed_seconds=elapsed,
            n_rollouts=len(records),
            rollout_start=indices[0],
            rollout_stop=indices[-1] + 1,
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
        raw_uri = f"{root}/{target['stem']}-t1.0-randomized.json.gz"
        publish(
            dict(
                target=target,
                model=args.model,
                temperature=1.0,
                mode="randomized",
                prompts={
                    r["prompt_id"]: p for r, p in zip(records, prompts, strict=True)
                },
                timing=timing,
                samples=records,
            ),
            raw_uri,
            compressed=True,
        )
        timing["total_seconds"] = (
            time.monotonic() - started + stage_seconds + model_load
        )
        unfinished = sum(r["finish_reason"] != "stop" for r in records)
        publish(
            dict(
                raw_uri=raw_uri,
                count=len(records),
                unfinished=unfinished,
                extended=sum(r["continuation_seed"] is not None for r in records),
                timing=timing,
            ),
            raw_uri.replace(".json.gz", ".complete.json"),
        )
        print(
            "COMPLETE",
            args.model,
            target["stem"],
            elapsed,
            "unfinished",
            unfinished,
            flush=True,
        )
        if unfinished or not any(r["contacts"] for r in records):
            raise ValueError("Require completed samples containing contacts")


if __name__ == "__main__":
    main()
