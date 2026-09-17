"""Sample V2 delta-stream suffixes and write exp82-compatible vote triplets."""

import argparse
import os
import time
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from delta_stream_rollout import DOC_END_TOKEN_ID, parse_contact_suffix, sequence_prefix

SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("i", pa.int16()), ("j", pa.int16()), ("votes", pa.int16()),
])


def read_parquet(uri: str):
    with fsspec.open(uri, "rb") as handle:
        return pq.read_table(handle)


def write_parquet(table, uri: str) -> None:
    with fsspec.open(uri, "wb") as handle:
        pq.write_table(table, handle, compression="zstd")


def stage_model(source: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(source)
    files = [entry for entry in fs.ls(root, detail=True) if entry["type"] == "file"]
    if not files:
        raise ValueError(f"no model files at {source}")
    for entry in files:
        fs.get_file(entry["name"], str(destination / os.path.basename(entry["name"])))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    shard_index, num_shards = (int(value) for value in args.shard.split("/"))
    records = read_parquet(args.targets).to_pylist()
    records.sort(key=lambda record: record["L"])
    records = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    if args.limit:
        records = records[: args.limit]

    from vllm import LLM, SamplingParams

    model = stage_model(args.model, Path("/tmp/delta-model"))
    llm = LLM(
        model=str(model), dtype="bfloat16", max_model_len=8192,
        gpu_memory_utilization=0.9, generation_config="vllm", max_num_seqs=512,
    )
    output_dir = f"{args.out.rstrip('/')}/{args.label}"
    malformed = total = 0
    started = time.time()
    for offset in range(0, len(records), args.chunk):
        group = records[offset : offset + args.chunk]
        prompts = []
        sampling = []
        for record in group:
            prefix = sequence_prefix(record["input_seq"])
            max_new = min(8192 - len(prefix), 6 * int(record["L"]) + 128)
            for rollout in range(args.n_rollouts):
                prompts.append({"prompt_token_ids": prefix})
                sampling.append(SamplingParams(
                    temperature=args.temperature, top_p=args.top_p, top_k=-1,
                    max_tokens=max_new, stop_token_ids=[DOC_END_TOKEN_ID], ignore_eos=True,
                    seed=shard_index * 10_000_019 + (offset + len(prompts)) * 1009 + rollout,
                ))
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        rows = {name: [] for name in ("dataset", "stem", "L", "i", "j", "votes")}
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
                except ValueError as error:
                    malformed += 1
                    if malformed <= 5:
                        print(f"malformed {record['dataset']}__{record['stem']}: {error}; ids={token_ids[:80]}", flush=True)
                    continue
                for left, right in pairs:
                    votes[left, right] += 1
            cursor += args.n_rollouts
            ii, jj = np.nonzero(np.triu(votes, k=1))
            rows["dataset"] += [record["dataset"]] * len(ii)
            rows["stem"] += [record["stem"]] * len(ii)
            rows["L"] += [length] * len(ii)
            rows["i"] += ii.astype(np.int16).tolist()
            rows["j"] += jj.astype(np.int16).tolist()
            rows["votes"] += votes[ii, jj].tolist()
        destination = f"{output_dir}/shard-{shard_index:03d}-part-{offset // args.chunk:04d}.parquet"
        write_parquet(pa.table(rows, schema=SCHEMA), destination)
        print(f"[{offset + len(group)}/{len(records)}] malformed={malformed}/{total} -> {destination}", flush=True)
    print(f"DONE {len(records)} proteins malformed={malformed}/{total} elapsed={(time.time()-started)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
