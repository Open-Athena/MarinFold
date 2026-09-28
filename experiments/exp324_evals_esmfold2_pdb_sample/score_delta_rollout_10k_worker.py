# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample delta-stream suffixes for exp324 10k targets and write sparse vote parts."""

import argparse
import os
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from delta_stream_rollout import DOC_END_TOKEN_ID, parse_contact_suffix, sequence_prefix

SCORE_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("i", pa.int16()), ("j", pa.int16()), ("votes", pa.int16()),
])

TIMING_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("n_residues", pa.int32()),
    ("n_pairs", pa.int64()), ("mode", pa.string()), ("elapsed_seconds", pa.float64()),
    ("model_load_seconds", pa.float64()), ("total_seconds", pa.float64()),
    ("model_nickname", pa.string()), ("runner_tag", pa.string()),
    ("gpu_name", pa.string()), ("gpu_total_memory_gb", pa.float64()),
    ("gpu_compute_capability", pa.string()), ("hostname", pa.string()),
    ("platform", pa.string()), ("torch_version", pa.string()),
    ("timestamp_utc", pa.string()), ("n_rollouts", pa.int32()),
    ("generated_tokens", pa.int64()), ("shard", pa.int32()), ("num_shards", pa.int32()),
    ("temperature", pa.float64()), ("top_p", pa.float64()),
    ("prompt_tokens", pa.int32()), ("max_tokens", pa.int32()),
])


def read_parquet(uri: str) -> pa.Table:
    with fsspec.open(uri, "rb") as handle:
        return pq.read_table(handle)


def write_parquet(table: pa.Table, uri: str) -> None:
    with fsspec.open(uri, "wb") as handle:
        pq.write_table(table, handle, compression="zstd")


def stage_model(source: str, destination: Path) -> tuple[Path, float]:
    if "://" not in source:
        return Path(source), 0.0
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fs, root = fsspec.core.url_to_fs(source)
    files = [entry for entry in fs.ls(root, detail=True) if entry["type"] == "file"]
    if not files:
        raise ValueError(f"no model files at {source}")
    for entry in files:
        fs.get_file(entry["name"], str(destination / os.path.basename(entry["name"])))
    return destination, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    shard_index, num_shards = (int(value) for value in args.shard.split("/"))
    records = read_parquet(args.targets).to_pylist()
    records.sort(key=lambda record: (record["L"], record["stem"]))
    records = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    if args.limit:
        records = records[: args.limit]

    import torch
    from vllm import LLM, SamplingParams

    model_dir, stage_seconds = stage_model(args.model, Path("/tmp/delta-model"))
    load_started = time.monotonic()
    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9, generation_config="vllm", max_num_seqs=512)
    model_load_seconds = time.monotonic() - load_started
    props = torch.cuda.get_device_properties(0)
    hardware = {
        "gpu_name": props.name,
        "gpu_total_memory_gb": props.total_memory / 2**30,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    output_dir = f"{args.out.rstrip('/')}/{args.label}"
    malformed = total = 0
    started = time.monotonic()
    for offset in range(0, len(records), args.chunk):
        group = records[offset : offset + args.chunk]
        prompts = []
        sampling = []
        for record in group:
            prefix = sequence_prefix(record["input_seq"])
            max_new = min(8192 - len(prefix), 6 * int(record["L"]) + 128)
            for rollout in range(args.n_rollouts):
                prompts.append({"prompt_token_ids": prefix})
                sampling.append(SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=-1, max_tokens=max_new, stop_token_ids=[DOC_END_TOKEN_ID], ignore_eos=True, seed=shard_index * 10_000_019 + (offset + len(prompts)) * 1009 + rollout))
        inference_started = time.monotonic()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        inference_seconds = time.monotonic() - inference_started
        per_protein_seconds = inference_seconds / len(group)
        rows = {name: [] for name in SCORE_SCHEMA.names}
        timings = []
        cursor = 0
        for record in group:
            length = int(record["L"])
            votes = np.zeros((length, length), dtype=np.int16)
            for output in outputs[cursor : cursor + args.n_rollouts]:
                total += 1
                token_ids = list(output.outputs[0].token_ids)
                if not token_ids or token_ids[-1] != DOC_END_TOKEN_ID:
                    token_ids.append(DOC_END_TOKEN_ID)
                try:
                    pairs = parse_contact_suffix(token_ids, length, strict=False)
                except ValueError:
                    malformed += 1
                    continue
                for left, right in pairs:
                    votes[left, right] += 1
            record_outputs = outputs[cursor : cursor + args.n_rollouts]
            cursor += args.n_rollouts
            ii, jj = np.nonzero(np.triu(votes, k=1))
            rows["dataset"] += [record["dataset"]] * len(ii)
            rows["stem"] += [record["stem"]] * len(ii)
            rows["L"] += [length] * len(ii)
            rows["i"] += ii.astype(np.int16).tolist()
            rows["j"] += jj.astype(np.int16).tolist()
            rows["votes"] += votes[ii, jj].tolist()
            timings.append({
                "dataset": record["dataset"], "stem": record["stem"], "n_residues": length,
                "n_pairs": max(length - 6, 0) * max(length - 5, 0) // 2,
                "mode": "rollout_resample", "elapsed_seconds": per_protein_seconds,
                "model_load_seconds": model_load_seconds, "total_seconds": stage_seconds + model_load_seconds + per_protein_seconds,
                "model_nickname": args.label, "runner_tag": "iris-coreweave", **hardware,
                "timestamp_utc": datetime.now(UTC).isoformat(), "n_rollouts": args.n_rollouts,
                "generated_tokens": sum(len(output.outputs[0].token_ids) for output in record_outputs),
                "shard": shard_index, "num_shards": num_shards, "temperature": args.temperature, "top_p": args.top_p,
                "prompt_tokens": len(sequence_prefix(record["input_seq"])),
                "max_tokens": min(8192 - len(sequence_prefix(record["input_seq"])), 6 * length + 128),
            })
        part = f"shard-{shard_index:03d}-part-{offset // args.chunk:04d}.parquet"
        write_parquet(pa.table(rows, schema=SCORE_SCHEMA), f"{output_dir}/scores/{part}")
        write_parquet(pa.Table.from_pylist(timings, schema=TIMING_SCHEMA), f"{output_dir}/timings/{part}")
        print(f"[{offset + len(group)}/{len(records)}] malformed={malformed}/{total} -> {output_dir}/scores/{part}", flush=True)
    print(f"DONE {len(records)} proteins malformed={malformed}/{total} elapsed={(time.monotonic() - started) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
