"""Sample contacts-v1 MarinFold rollouts for exp324 10k targets and write sparse vote parts."""

import argparse
import os
import platform
import re
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEGIN = "<begin_statements>"
NUM_POSITIONS = 2_000
MINIMUM_SEPARATION = 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

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
    ("generated_tokens", pa.int64()), ("stopped_rollouts", pa.int32()),
    ("unfinished_rollouts", pa.int32()), ("parsed_contacts", pa.int64()),
    ("valid_contacts", pa.int64()), ("complete", pa.bool_()),
    ("shard", pa.int32()), ("num_shards", pa.int32()),
    ("seed", pa.int64()), ("temperature", pa.float64()), ("top_p", pa.float64()),
    ("top_k", pa.int32()), ("prompt_tokens", pa.int32()), ("max_tokens", pa.int32()),
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


def candidate_pair_count(length: int) -> int:
    remaining = max(length - MINIMUM_SEPARATION, 0)
    return remaining * (remaining + 1) // 2


def gpu_metadata() -> dict[str, str | float]:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_total_memory_gb": props.total_memory / 2**30,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }


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
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contact-mult", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--fixed-residue-position-embeddings", choices=["auto", "force"], default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    shard_index, num_shards = (int(value) for value in args.shard.split("/"))
    records = read_parquet(args.targets).to_pylist()
    records.sort(key=lambda record: (record["L"], record["stem"]))
    records = [record for index, record in enumerate(records) if index % num_shards == shard_index]
    if args.limit:
        records = records[: args.limit]

    from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence
    from marinfold.inference._model_source import model_source_path
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_dir, stage_seconds = stage_model(args.model, Path("/tmp/marinfold-model"))
    fixed_mode = None if args.fixed_residue_position_embeddings == "auto" else args.fixed_residue_position_embeddings
    effective_model_dir = Path(model_source_path(model_dir, fixed_residue_position_embeddings=fixed_mode))
    tokenizer = AutoTokenizer.from_pretrained(str(effective_model_dir))
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    if end_token_id is None or end_token_id < 0:
        raise ValueError("model tokenizer has no <end> token")
    load_started = time.monotonic()
    llm = LLM(model=str(effective_model_dir), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=False, generation_config="vllm", max_num_seqs=512, seed=args.seed)
    model_load_seconds = time.monotonic() - load_started
    hardware = gpu_metadata()
    output_dir = f"{args.out.rstrip('/')}/{args.label}"
    started = time.monotonic()
    total_unfinished = total_rollouts = 0
    for offset in range(0, len(records), args.chunk):
        group = records[offset : offset + args.chunk]
        prompts: list[str] = []
        sampling: list[SamplingParams] = []
        per_record = []
        for record in group:
            residues = residues_from_sequence(record["input_seq"])
            first = len(prompts)
            position_maps: list[dict[int, int]] = []
            for rollout in range(args.n_rollouts):
                document = build_document(f"{record['stem']}:r{rollout}", residues, [], config=GenerationConfig())
                prompts.append(document.document[: document.document.index(BEGIN) + len(BEGIN)])
                position_maps.append({(document.n_term_index + index) % NUM_POSITIONS: index for index in range(document.seq_len)})
            prompt_tokens = len(tokenizer(prompts[first], add_special_tokens=False).input_ids)
            max_tokens = min(8192 - prompt_tokens, args.contact_mult * int(record["L"]) + 128)
            per_record.append({"record": record, "first": first, "position_maps": position_maps, "prompt_tokens": prompt_tokens, "max_tokens": max_tokens})
            sampling.extend(SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=max_tokens, stop_token_ids=[end_token_id], skip_special_tokens=False, seed=args.seed * 1_000_003 + first + rollout) for rollout in range(args.n_rollouts))
        inference_started = time.monotonic()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        inference_seconds = time.monotonic() - inference_started
        rows = {name: [] for name in SCORE_SCHEMA.names}
        timings = []
        for item in per_record:
            record = item["record"]
            length = int(record["L"])
            record_outputs = outputs[item["first"] : item["first"] + args.n_rollouts]
            unfinished = sum(1 for output in record_outputs if output.outputs[0].finish_reason != "stop")
            total_unfinished += unfinished
            total_rollouts += args.n_rollouts
            votes = np.zeros((length, length), dtype=np.int16)
            parsed_contacts = valid_contacts = 0
            for output, position_map in zip(record_outputs, item["position_maps"], strict=True):
                if output.outputs[0].finish_reason != "stop":
                    continue
                seen: set[tuple[int, int]] = set()
                matches = CONTACT_RE.findall(output.outputs[0].text)
                parsed_contacts += len(matches)
                for p1, p2 in matches:
                    i = position_map.get(int(p1))
                    j = position_map.get(int(p2))
                    if i is None or j is None or i == j:
                        continue
                    pair = (min(i, j), max(i, j))
                    if abs(i - j) < MINIMUM_SEPARATION or pair in seen:
                        continue
                    seen.add(pair)
                    votes[pair] += 1
                    valid_contacts += 1
            ii, jj = np.nonzero(np.triu(votes, k=1))
            rows["dataset"] += [record["dataset"]] * len(ii)
            rows["stem"] += [record["stem"]] * len(ii)
            rows["L"] += [length] * len(ii)
            rows["i"] += ii.astype(np.int16).tolist()
            rows["j"] += jj.astype(np.int16).tolist()
            rows["votes"] += votes[ii, jj].tolist()
            timings.append({
                "dataset": record["dataset"], "stem": record["stem"], "n_residues": length,
                "n_pairs": candidate_pair_count(length), "mode": "rollout_resample",
                "elapsed_seconds": inference_seconds / len(group), "model_load_seconds": model_load_seconds,
                "total_seconds": stage_seconds + model_load_seconds + inference_seconds / len(group),
                "model_nickname": args.label, "runner_tag": "iris-coreweave", **hardware,
                "timestamp_utc": datetime.now(UTC).isoformat(), "n_rollouts": args.n_rollouts,
                "generated_tokens": sum(len(output.outputs[0].token_ids) for output in record_outputs),
                "stopped_rollouts": args.n_rollouts - unfinished, "unfinished_rollouts": unfinished,
                "parsed_contacts": parsed_contacts, "valid_contacts": valid_contacts, "complete": unfinished == 0,
                "shard": shard_index, "num_shards": num_shards, "seed": args.seed,
                "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "prompt_tokens": item["prompt_tokens"], "max_tokens": item["max_tokens"],
            })
        part = f"shard-{shard_index:03d}-part-{offset // args.chunk:04d}.parquet"
        write_parquet(pa.table(rows, schema=SCORE_SCHEMA), f"{output_dir}/scores/{part}")
        write_parquet(pa.Table.from_pylist(timings, schema=TIMING_SCHEMA), f"{output_dir}/timings/{part}")
        print(f"[{offset + len(group)}/{len(records)}] L={group[0]['L']}-{group[-1]['L']} unfinished={total_unfinished}/{total_rollouts} -> {output_dir}/scores/{part}", flush=True)
    print(f"DONE {len(records)} proteins unfinished={total_unfinished}/{total_rollouts} elapsed={(time.monotonic() - started) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
