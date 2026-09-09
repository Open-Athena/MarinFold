# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run frozen contact-prompt interventions on one local or CoreWeave GPU."""

import argparse
import gzip
import hashlib
import io
import json
import platform
import random
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from common import parse_rollout, realization, seed_statement
from rank_pairwise import pcontact_matrix

MODEL_URI = (
    "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
    "checkpoints/protein/prot-exp232-cw-cv1-decontam-s02-m2-p06-aug/"
    "2026.08.14.2/hf/step-145199"
)
MODEL_FILES = {
    "config.json": (1557, "d8e904f8170ddf00d74c864f31d258a4"),
    "model-00001-of-00002.safetensors": (
        4979485528,
        "2e38a75033f4df3a73a4be9bc2ceeefe-95",
    ),
    "model-00002-of-00002.safetensors": (
        906042048,
        "f444bd62152329ef71c7c46e7ee1c3cd-18",
    ),
    "model.safetensors.index.json": (20882, "bc0a5fd2c9aae096abae4caf9040c79c"),
    "tokenizer.json": (64407, "c4b3a16978e30eb150cca4fd8934b6ae"),
    "tokenizer_config.json": (290, "336f4e2ca951fa13a20cb1c4b68b2040"),
}


def write_bytes(uri: str, payload: bytes) -> None:
    """Write through the filesystem configured by the execution host."""
    fs, path = fsspec.core.url_to_fs(uri)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "wb") as stream:
        stream.write(payload)


def write_json(uri: str, value: dict) -> None:
    """Publish a JSON record after its associated data have been written."""
    write_bytes(uri, (json.dumps(value, indent=2) + "\n").encode())


def stage_model(source: str) -> str:
    """Use local weights or stage the pinned export entirely within CoreWeave."""
    if "://" not in source:
        return source
    if source != MODEL_URI:
        raise ValueError("Only the pinned co-located CoreWeave checkpoint is supported")
    fs, root = fsspec.core.url_to_fs(source)
    destination = Path("/tmp/exp254-m2-p06-step145199")
    destination.mkdir(exist_ok=True)
    for name, (size, etag) in MODEL_FILES.items():
        info = fs.info(f"{root}/{name}")
        if info["size"] != size or info["ETag"].strip('"') != etag:
            raise ValueError(f"Checkpoint identity mismatch for {name}")
        path = destination / name
        if not path.exists() or path.stat().st_size != size:
            fs.get_file(f"{root}/{name}", str(path))
    return str(destination)


def prompt(prefix: str, positions: list[int], pairs: list[list[int]], key: str) -> str:
    """Serialize a fixed set into the realization, shuffling order and orientation."""
    generator = random.Random(key)
    ordered = list(pairs)
    generator.shuffle(ordered)
    suffix = "".join(
        seed_statement(positions[i], positions[j], generator) for i, j in ordered
    )
    # Round-trip validation catches wrong coordinate maps and malformed tokens.
    inverse = {position: i for i, position in enumerate(positions)}
    expected = {tuple(pair) for pair in pairs}
    if set(parse_rollout(suffix, inverse)) != expected:
        raise ValueError("Supplied contact serialization does not round-trip")
    return prefix + suffix


def run(args: argparse.Namespace) -> None:
    """Evaluate complete per-protein units and resume only verified completions."""
    # These optional accelerator imports keep CPU prompt tests lightweight.
    import torch
    import transformers
    import vllm
    from marinfold.document_structures.contacts_v1 import residues_from_sequence
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    raw_plan = args.plan.read_bytes()
    plan = json.loads(raw_plan)
    plan_sha = hashlib.sha256(raw_plan).hexdigest()
    if plan["eval_set"] != "eval-val" or len(plan["targets"]) != 97:
        raise ValueError("This protocol requires exactly 97 eval-val inputs")
    shard, shards = map(int, args.shard.split("/"))
    if not 0 <= shard < shards:
        raise ValueError("Shard must satisfy 0 <= index < count")
    targets = sorted(plan["targets"], key=lambda t: (t["L"], t["stem"]))
    targets = [
        target for index, target in enumerate(targets) if index % shards == shard
    ]
    if args.stem:
        targets = [target for target in targets if target["stem"] == args.stem]
        if not targets:
            raise ValueError("Requested smoke target is not in the selected shard")
    fs, root = fsspec.core.url_to_fs(args.out)
    pending = []
    for target in targets:
        marker = f"{root}/units/{target['stem']}.complete.json"
        if fs.exists(marker):
            with fs.open(marker) as stream:
                complete = json.load(stream)
            if complete["plan_sha256"] != plan_sha:
                raise ValueError("Cannot resume results from a different plan")
            for suffix, expected in complete["files"].items():
                with fs.open(f"{root}/units/{target['stem']}.{suffix}", "rb") as stream:
                    payload = stream.read()
                if (
                    len(payload) != expected["bytes"]
                    or hashlib.sha256(payload).hexdigest() != expected["sha256"]
                ):
                    raise ValueError(
                        f"Corrupt completed output: {target['stem']}.{suffix}"
                    )
        else:
            pending.append(target)
    if not pending:
        print("All selected units already complete", flush=True)
        return
    setup_start = time.monotonic()
    model = stage_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.encode("<contact>", add_special_tokens=False) != [
        tokenizer.convert_tokens_to_ids("<contact>")
    ]:
        raise ValueError("Contact marker is not a single dedicated token")
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_fraction,
        enable_prefix_caching=True,
        generation_config="vllm",
        max_num_seqs=128,
        max_logprobs=len(tokenizer),
        seed=254,
    )
    load_seconds = time.monotonic() - setup_start
    device = torch.cuda.get_device_properties(0)
    worker_meta = {
        "model_nickname": "prot-exp232-cw-cv1-decontam-s02-m2-p06-aug-step-145199",
        "model_source": args.model,
        "model_load_seconds": load_seconds,
        "runner_tag": "iris" if args.out.startswith("s3://") else "local",
        "gpu_name": device.name,
        "gpu_total_memory_gb": device.total_memory / 2**30,
        "gpu_compute_capability": f"{device.major}.{device.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "plan_sha256": plan_sha,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    prob_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        logprobs=len(tokenizer),
        seed=254,
    )
    for target in pending:
        started = time.monotonic()
        stem, length = target["stem"], target["L"]
        residues = residues_from_sequence(target["input_seq"])
        arrays, raw, timings = {}, {}, []
        for rep in range(plan["n_repeats"]):
            frames = [
                realization(stem, residues, f"conditional-v1-rep{rep}-r{r}")
                for r in range(plan["n_rollouts"])
            ]
            for arm in plan["arms"]:
                given = target["contexts"][rep][arm]
                given_set = {tuple(pair) for pair in given}
                prompts = [
                    prompt(prefix, positions, given, f"{stem}:{rep}:{arm}:{r}")
                    for r, (prefix, positions) in enumerate(frames)
                ]
                token_counts = [
                    len(tokenizer.encode(text, add_special_tokens=False))
                    for text in prompts
                ]
                budget = 6 * length + 128
                if max(token_counts) + budget > 8192:
                    raise ValueError(
                        f"{stem}/{arm}: full prefix plus completion allowance exceeds context"
                    )
                arm_start = time.monotonic()
                # Paired request RNG across conditions; iid_repeat independently
                # estimates sampling variation without a prompt intervention.
                offset = 10_000_000 if arm == "iid_repeat" else 0
                params = [
                    SamplingParams(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=-1,
                        max_tokens=budget,
                        stop_token_ids=[end_id],
                        skip_special_tokens=False,
                        seed=254_000_000 + rep * 1_000_000 + r + offset,
                    )
                    for r in range(plan["n_rollouts"])
                ]
                generation_start = time.monotonic()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.monotonic() - generation_start
                if len(outputs) != plan["n_rollouts"]:
                    raise ValueError("Engine returned an incomplete rollout set")
                votes = np.zeros((length, length), dtype=np.int16)
                saved, copied, novel, unfinished, token_total = [], 0, 0, 0, 0
                for output, (_, positions) in zip(outputs, frames, strict=True):
                    completion = output.outputs[0]
                    pairs = parse_rollout(
                        completion.text,
                        {position: i for i, position in enumerate(positions)},
                    )
                    if pairs:
                        ii, jj = np.array(pairs).T
                        votes[ii, jj] += 1
                        votes[jj, ii] += 1
                    copied += len(set(pairs) & given_set)
                    novel += len(set(pairs) - given_set)
                    unfinished += completion.finish_reason != "stop"
                    token_total += len(completion.token_ids)
                    saved.append(
                        {
                            "text": completion.text,
                            "contacts": pairs,
                            "finish_reason": completion.finish_reason,
                            "tokens": len(completion.token_ids),
                        }
                    )
                key = f"r{rep}__{arm}"
                arrays[f"{key}__votes"] = votes
                raw[key] = saved
                probability_seconds = 0.0
                if arm != "iid_repeat":
                    pstart = time.monotonic()
                    arrays[f"{key}__prob"] = pcontact_matrix(
                        llm, prob_params, tokenizer, prompts[0], frames[0][1]
                    ).astype(np.float32)
                    probability_seconds = time.monotonic() - pstart
                timings.append(
                    dict(
                        stem=stem,
                        n_residues=length,
                        n_pairs=length * (length - 1) // 2,
                        mode=arm,
                        replicate=rep,
                        n_rollouts=plan["n_rollouts"],
                        n_given=len(given),
                        elapsed_seconds=elapsed,
                        probability_probe_seconds=probability_seconds,
                        total_seconds=time.monotonic() - arm_start,
                        prompt_tokens=max(token_counts),
                        max_new_tokens=budget,
                        generated_tokens=token_total,
                        copied_context_pairs=copied,
                        novel_generated_pairs=novel,
                        unfinished_rollouts=unfinished,
                        timestamp_utc=datetime.now(UTC).isoformat(),
                        **worker_meta,
                    )
                )
                print(
                    f"{stem} rep={rep} {arm} n_given={len(given)} tokens={token_total} "
                    f"rollout_s={elapsed:.1f} probe_s={probability_seconds:.1f} unfinished={unfinished}",
                    flush=True,
                )
                if unfinished:
                    write_bytes(
                        f"{args.out}/failures/{stem}-{key}.json.gz",
                        gzip.compress(json.dumps(saved).encode(), mtime=0),
                    )
                    raise ValueError(
                        f"{stem}/{key}: unfinished rollouts; saved for diagnosis"
                    )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        outputs = {
            "npz": buffer.getvalue(),
            "raw.json.gz": gzip.compress(json.dumps(raw).encode(), mtime=0),
            "timings.csv": pd.DataFrame(timings).to_csv(index=False).encode(),
        }
        for suffix, payload in outputs.items():
            write_bytes(f"{args.out}/units/{stem}.{suffix}", payload)
        write_json(
            f"{args.out}/units/{stem}.complete.json",
            {
                "stem": stem,
                "plan_sha256": plan_sha,
                "elapsed_seconds": time.monotonic() - started,
                "files": {
                    suffix: {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                    for suffix, payload in outputs.items()
                },
            },
        )
        print(f"COMPLETE {stem} elapsed={time.monotonic() - started:.1f}s", flush=True)


def main() -> None:
    """Run one resumable shard of the frozen experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_URI)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--stem")
    parser.add_argument("--gpu-fraction", type=float, default=0.85)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
