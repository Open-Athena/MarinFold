# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One resumable CoreWeave H100 shard of the exp82 rollout+resample eval.

The sampling and vote-matrix construction match exp82's
``score_rollout_worker.py``. This copy adds completion markers and per-protein
timing records so retries are atomic at a chunk boundary and the exp199 result
retains the predictor timing/provenance required by repository policy.
"""

import argparse
import base64
import json
import os
import platform
import re
import socket
import sys
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

SCORE_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("L", pa.int32()),
        ("i", pa.int16()),
        ("j", pa.int16()),
        ("votes", pa.int16()),
    ]
)

TIMING_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("n_residues", pa.int32()),
        ("n_pairs", pa.int64()),
        ("mode", pa.string()),
        ("elapsed_seconds", pa.float64()),
        ("model_load_seconds", pa.float64()),
        ("total_seconds", pa.float64()),
        ("model_nickname", pa.string()),
        ("runner_tag", pa.string()),
        ("gpu_name", pa.string()),
        ("gpu_total_memory_gb", pa.float64()),
        ("gpu_compute_capability", pa.string()),
        ("hostname", pa.string()),
        ("platform", pa.string()),
        ("torch_version", pa.string()),
        ("timestamp_utc", pa.string()),
        ("n_rollouts", pa.int32()),
        ("generated_tokens", pa.int64()),
        ("stopped_rollouts", pa.int32()),
        ("unfinished_rollouts", pa.int32()),
        ("parsed_contacts", pa.int64()),
        ("valid_contacts", pa.int64()),
        ("complete", pa.bool_()),
        ("shard", pa.int32()),
        ("num_shards", pa.int32()),
        ("seed", pa.int64()),
        ("temperature", pa.float64()),
        ("top_p", pa.float64()),
        ("top_k", pa.int32()),
        ("prompt_tokens", pa.int32()),
        ("max_tokens", pa.int32()),
    ]
)


def stage_model(
    source: str, destination: Path, manifest: dict
) -> tuple[Path, float, int]:
    """Stage an S3 model directory to local disk for vLLM."""

    if source.startswith("gs://"):
        raise ValueError(
            "GCS model sources are forbidden for this CoreWeave evaluation"
        )
    if "://" not in source:
        return Path(source), 0.0, 0
    if not source.startswith("s3://"):
        raise ValueError(f"expected an S3 model mirror, received {source!r}")
    destination.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    filesystem, root = fsspec.core.url_to_fs(source)
    expected_files = {entry["name"]: entry["size"] for entry in manifest["files"]}
    files = [
        entry for entry in filesystem.ls(root, detail=True) if entry["type"] == "file"
    ]
    files = [entry for entry in files if not entry["name"].endswith("identity.json")]
    actual_files = {os.path.basename(entry["name"]): entry["size"] for entry in files}
    if actual_files != expected_files:
        raise ValueError(
            f"model file set does not match its verified HF identity: {actual_files} != {expected_files}"
        )
    for entry in files:
        filesystem.get_file(
            entry["name"], str(destination / os.path.basename(entry["name"]))
        )
    size = sum(entry["size"] for entry in files)
    elapsed = time.monotonic() - start
    print(
        f"[worker] staged {len(files)} files ({size / 2**30:.2f} GiB) "
        f"from {source} in {elapsed:.1f}s",
        flush=True,
    )
    return destination, elapsed, size


def read_parquet(uri: str, columns: list[str] | None = None) -> pa.Table:
    """Read one parquet artifact through fsspec."""

    with fsspec.open(uri, "rb") as file:
        return pq.read_table(file, columns=columns)


def write_parquet(table: pa.Table, uri: str) -> None:
    """Write one compressed parquet artifact through fsspec."""

    with fsspec.open(uri, "wb") as file:
        pq.write_table(table, file, compression="zstd")


def write_json(data: dict, uri: str) -> None:
    """Write one JSON artifact through fsspec."""

    with fsspec.open(uri, "wt") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def load_targets(path: str) -> list[dict]:
    """Load and length-sort the immutable exp89 evaluation units."""

    records = read_parquet(path).to_pylist()
    records.sort(key=lambda record: (record["L"], record["dataset"], record["stem"]))
    return records


def completed_units(
    output_directory: str, shard: int, num_shards: int
) -> tuple[set[str], int]:
    """Return units committed by completion markers and the next part index."""

    filesystem, _ = fsspec.core.url_to_fs(output_directory)
    pattern = (
        f"{output_directory.rstrip('/')}/complete/"
        f"shard-{shard:03d}-of-{num_shards:03d}-part-*.json"
    )
    markers = sorted(filesystem.glob(pattern))
    units: set[str] = set()
    for marker in markers:
        with filesystem.open(marker, "rt") as file:
            record = json.load(file)
        if record["unfinished_rollouts"] != 0:
            raise ValueError(
                f"invalid completion marker with unfinished rollouts: {marker}"
            )
        for unit in record["units"]:
            key = f"{unit['dataset']}__{unit['stem']}"
            if key in units:
                raise ValueError(f"duplicate completed unit {key} in shard {shard}")
            units.add(key)
    return units, len(markers)


def gpu_metadata() -> dict[str, str | float]:
    """Return stable worker hardware and runtime fields."""

    import torch

    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": properties.name,
        "gpu_total_memory_gb": properties.total_memory / 2**30,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }


def candidate_pair_count(length: int) -> int:
    """Count upper-triangle pairs separated by at least six residues."""

    remaining = max(length - MINIMUM_SEPARATION, 0)
    return remaining * (remaining + 1) // 2


def parse_arguments() -> argparse.Namespace:
    """Parse one worker invocation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest-b64", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--contact-mult", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-frac", type=float, default=0.90)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    """Run one interleaved shard and commit atomic chunk markers."""

    arguments = parse_arguments()
    shard, num_shards = (int(value) for value in arguments.shard.split("/"))
    output_directory = f"{arguments.out.rstrip('/')}/{arguments.label}"

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
        residues_from_sequence,
    )
    from marinfold.inference._tokenizer import model_source_path
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_targets(arguments.targets)
    assigned = [
        record for index, record in enumerate(records) if index % num_shards == shard
    ]
    if arguments.limit is not None:
        assigned = assigned[: arguments.limit]
    completed, part = completed_units(output_directory, shard, num_shards)
    pending = [
        record
        for record in assigned
        if f"{record['dataset']}__{record['stem']}" not in completed
    ]
    print(
        f"[worker] shard {shard}/{num_shards}: assigned={len(assigned)} "
        f"completed={len(completed)} pending={len(pending)} n_rollouts={arguments.n_rollouts} "
        f"temperature={arguments.temperature} top_p={arguments.top_p} "
        f"top_k={arguments.top_k} seed={arguments.seed}",
        flush=True,
    )
    if not pending:
        return 0

    worker_start = time.monotonic()
    model_directory, model_stage_seconds, _ = stage_model(
        arguments.model,
        Path("/tmp/marinfold_model"),
        json.loads(base64.b64decode(arguments.model_manifest_b64)),
    )
    effective_model_directory = Path(model_source_path(model_directory))
    tokenizer = AutoTokenizer.from_pretrained(str(effective_model_directory))
    with (effective_model_directory / "config.json").open() as file:
        effective_config = json.load(file)
    rope_scaling = effective_config.get("rope_scaling") or {}
    if effective_config.get("rope_theta") != 500_000:
        raise ValueError(
            "the effective model config did not preserve rope_theta=500000"
        )
    if rope_scaling.get("rope_type") != "llama3":
        raise ValueError(
            "the effective model config did not preserve llama3 rope scaling"
        )
    if len(tokenizer) != effective_config.get("vocab_size"):
        raise ValueError(
            f"tokenizer/config vocabulary mismatch: {len(tokenizer)} != "
            f"{effective_config.get('vocab_size')}"
        )
    print(
        f"[worker] effective model overlay={effective_model_directory} "
        f"tokenizer={type(tokenizer).__name__} vocab={len(tokenizer)} "
        f"rope_theta={effective_config['rope_theta']} rope_type={rope_scaling['rope_type']}",
        flush=True,
    )
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    if end_token_id is None or end_token_id < 0:
        raise ValueError("the checkpoint tokenizer has no <end> token")

    load_start = time.monotonic()
    model = LLM(
        model=str(effective_model_directory),
        dtype="bfloat16",
        max_model_len=8_192,
        gpu_memory_utilization=arguments.gpu_frac,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=arguments.max_num_seqs,
        seed=arguments.seed,
    )
    model_load_seconds = time.monotonic() - load_start
    hardware = gpu_metadata()

    total_rollouts = 0
    total_unfinished = 0
    for offset in range(0, len(pending), arguments.chunk):
        group = pending[offset : offset + arguments.chunk]
        prompts: list[str] = []
        sampling_parameters: list[SamplingParams] = []
        per_record: list[dict] = []

        for record in group:
            residues = residues_from_sequence(record["input_seq"])
            first = len(prompts)
            position_maps: list[dict[int, int]] = []
            for rollout in range(arguments.n_rollouts):
                document = build_document(
                    f"{record['stem']}:r{rollout}",
                    residues,
                    [],
                    config=GenerationConfig(),
                )
                prompts.append(
                    document.document[: document.document.index(BEGIN) + len(BEGIN)]
                )
                position_maps.append(
                    {
                        (document.n_term_index + index) % NUM_POSITIONS: index
                        for index in range(document.seq_len)
                    }
                )
            prompt_tokens = len(
                tokenizer(prompts[first], add_special_tokens=False).input_ids
            )
            max_tokens = min(
                8_192 - prompt_tokens, arguments.contact_mult * record["L"] + 128
            )
            per_record.append(
                {
                    "record": record,
                    "first": first,
                    "position_maps": position_maps,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens,
                }
            )
            sampling_parameters.extend(
                SamplingParams(
                    temperature=arguments.temperature,
                    top_p=arguments.top_p,
                    top_k=arguments.top_k,
                    max_tokens=max_tokens,
                    stop_token_ids=[end_token_id],
                    skip_special_tokens=False,
                    seed=arguments.seed * 1_000_003 + first + rollout,
                )
                for rollout in range(arguments.n_rollouts)
            )

        inference_start = time.monotonic()
        outputs = model.generate(prompts, sampling_parameters, use_tqdm=False)
        inference_seconds = time.monotonic() - inference_start
        score_rows = {name: [] for name in SCORE_SCHEMA.names}
        timing_rows: list[dict] = []
        marker_units: list[dict] = []
        group_unfinished = 0
        unfinished_details: list[dict] = []

        for item in per_record:
            record = item["record"]
            first = item["first"]
            position_maps = item["position_maps"]
            record_outputs = outputs[first : first + arguments.n_rollouts]
            unfinished_outputs = [
                (rollout, output)
                for rollout, output in enumerate(record_outputs)
                if output.outputs[0].finish_reason != "stop"
            ]
            unfinished = len(unfinished_outputs)
            group_unfinished += unfinished
            total_unfinished += unfinished
            total_rollouts += arguments.n_rollouts
            unfinished_details.extend(
                {
                    "dataset": record["dataset"],
                    "stem": record["stem"],
                    "rollout": rollout,
                    "sampling_seed": arguments.seed * 1_000_003 + first + rollout,
                    "finish_reason": output.outputs[0].finish_reason,
                    "generated_tokens": len(output.outputs[0].token_ids),
                    "max_tokens": item["max_tokens"],
                }
                for rollout, output in unfinished_outputs
            )
            length = record["L"]
            votes = np.zeros((length, length), np.int32)
            parsed_contacts = 0
            valid_contacts = 0

            for output, position_map in zip(record_outputs, position_maps, strict=True):
                seen: set[tuple[int, int]] = set()
                matches = CONTACT_RE.findall(output.outputs[0].text)
                parsed_contacts += len(matches)
                for first_position, second_position in matches:
                    first_index = position_map.get(int(first_position))
                    second_index = position_map.get(int(second_position))
                    if (
                        first_index is None
                        or second_index is None
                        or first_index == second_index
                    ):
                        continue
                    pair = (
                        min(first_index, second_index),
                        max(first_index, second_index),
                    )
                    if (
                        abs(first_index - second_index) < MINIMUM_SEPARATION
                        or pair in seen
                    ):
                        continue
                    seen.add(pair)
                    votes[pair] += 1
                    valid_contacts += 1

            row_indices, column_indices = np.nonzero(np.triu(votes, k=1))
            score_rows["dataset"].extend([record["dataset"]] * len(row_indices))
            score_rows["stem"].extend([record["stem"]] * len(row_indices))
            score_rows["L"].extend([length] * len(row_indices))
            score_rows["i"].extend(row_indices.astype(np.int16).tolist())
            score_rows["j"].extend(column_indices.astype(np.int16).tolist())
            score_rows["votes"].extend(
                votes[row_indices, column_indices].astype(np.int16).tolist()
            )

            generated_tokens = sum(
                len(output.outputs[0].token_ids) for output in record_outputs
            )
            timing_rows.append(
                {
                    "dataset": record["dataset"],
                    "stem": record["stem"],
                    "n_residues": length,
                    "n_pairs": candidate_pair_count(length),
                    "mode": "rollout_resample",
                    "elapsed_seconds": inference_seconds,
                    "model_load_seconds": model_load_seconds,
                    "total_seconds": model_stage_seconds
                    + model_load_seconds
                    + inference_seconds,
                    "model_nickname": arguments.label,
                    "runner_tag": "iris-coreweave",
                    **hardware,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "n_rollouts": arguments.n_rollouts,
                    "generated_tokens": generated_tokens,
                    "stopped_rollouts": arguments.n_rollouts - unfinished,
                    "unfinished_rollouts": unfinished,
                    "parsed_contacts": parsed_contacts,
                    "valid_contacts": valid_contacts,
                    "complete": unfinished == 0,
                    "shard": shard,
                    "num_shards": num_shards,
                    "seed": arguments.seed,
                    "temperature": arguments.temperature,
                    "top_p": arguments.top_p,
                    "top_k": arguments.top_k,
                    "prompt_tokens": item["prompt_tokens"],
                    "max_tokens": item["max_tokens"],
                }
            )
            marker_units.append(
                {
                    "dataset": record["dataset"],
                    "stem": record["stem"],
                    "L": length,
                    "n_rollouts": arguments.n_rollouts,
                }
            )

        part_stem = f"shard-{shard:03d}-of-{num_shards:03d}-part-{part:04d}"
        if group_unfinished:
            failure_uri = f"{output_directory}/failures/{part_stem}.json"
            write_json(
                {
                    "units": marker_units,
                    "unfinished_rollouts": group_unfinished,
                    "total_rollouts": len(group) * arguments.n_rollouts,
                    "unfinished_details": unfinished_details,
                },
                failure_uri,
            )
            raise RuntimeError(
                f"{group_unfinished} rollout(s) hit the token cap; diagnostics at {failure_uri}"
            )

        score_uri = f"{output_directory}/scores/{part_stem}.parquet"
        timing_uri = f"{output_directory}/timings/{part_stem}.parquet"
        marker_uri = f"{output_directory}/complete/{part_stem}.json"
        write_parquet(pa.table(score_rows, schema=SCORE_SCHEMA), score_uri)
        write_parquet(
            pa.Table.from_pylist(timing_rows, schema=TIMING_SCHEMA), timing_uri
        )
        write_json(
            {
                "units": marker_units,
                "total_rollouts": len(group) * arguments.n_rollouts,
                "unfinished_rollouts": 0,
                "score_uri": score_uri,
                "timing_uri": timing_uri,
            },
            marker_uri,
        )
        part += 1
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(
            f"[worker] {offset + len(group)}/{len(pending)} proteins "
            f"L={group[0]['L']}-{group[-1]['L']} {inference_seconds:.1f}s "
            f"{generated_tokens / inference_seconds:.0f} tok/s "
            f"unfinished={total_unfinished}/{total_rollouts} -> {marker_uri} "
            f"elapsed={(time.monotonic() - worker_start) / 60:.1f}m",
            flush=True,
        )

    print(
        f"[worker] DONE shard {shard}/{num_shards}: proteins={len(pending)} "
        f"unfinished={total_unfinished}/{total_rollouts} "
        f"elapsed={(time.monotonic() - worker_start) / 60:.1f}m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
