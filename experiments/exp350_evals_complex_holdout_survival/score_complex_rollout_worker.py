# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One resumable CoreWeave H100 shard of the complex rollout+resample eval.

The sampling recipe matches exp82, while prompt layout and contact filtering are
chain-aware: each chain has independent termini and the six-residue separation
filter applies only within a chain. Completion markers and per-target timings
make retries atomic and preserve the predictor metadata required by policy.
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
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEGIN = "<begin_statements>"
NUM_POSITIONS = 2_000
MINIMUM_SEPARATION = 6
MODEL_CONTEXT = 8_192
DEFAULT_CONTACT_MULT = 0
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

ROLLOUT_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("stem", pa.string()),
        ("rollout", pa.int32()),
        ("sampling_seed", pa.int64()),
        ("finish_reason", pa.string()),
        ("generated_tokens", pa.int32()),
        ("parsed_contacts", pa.int32()),
        ("contacts", pa.list_(pa.list_(pa.int16(), 2))),
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
    """Load and length-sort the frozen complex evaluation units."""

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
        if record["unfinished_rollouts"] != 0 and not record.get(
            "accepted_unfinished", False
        ):
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


def candidate_pair_count(chain_lengths: list[int]) -> int:
    """Count all eligible intra-chain and inter-chain sequence pairs."""

    intra = 0
    for length in chain_lengths:
        remaining = max(length - MINIMUM_SEPARATION, 0)
        intra += remaining * (remaining + 1) // 2
    inter = sum(
        length * other
        for index, length in enumerate(chain_lengths)
        for other in chain_lengths[index + 1 :]
    )
    return intra + inter


def same_chain_too_close(first: int, second: int, chain_lengths: list[int]) -> bool:
    """Return whether a pair violates the within-chain separation rule."""

    offset = 0
    for length in chain_lengths:
        stop = offset + length
        if offset <= first < stop and offset <= second < stop:
            return abs(first - second) < MINIMUM_SEPARATION
        offset = stop
    return False


def generation_token_budget(
    prompt_tokens: int,
    length: int,
    contact_mult: int = DEFAULT_CONTACT_MULT,
) -> int:
    """Return a non-truncating complex rollout budget within model context."""
    available_context = MODEL_CONTEXT - prompt_tokens
    if contact_mult <= 0:
        return available_context
    return min(available_context, contact_mult * length + 128)


def parse_rollout_contacts(
    text: str, position_map: dict[int, int], chain_lengths: list[int]
) -> tuple[list[tuple[int, int]], int]:
    """Decode unique contacts in emission order into canonical chain coordinates."""
    matches = CONTACT_RE.findall(text)
    contacts: dict[tuple[int, int], None] = {}
    for first_position, second_position in matches:
        first = position_map.get(int(first_position))
        second = position_map.get(int(second_position))
        if first is None or second is None or first == second:
            continue
        if same_chain_too_close(first, second, chain_lengths):
            continue
        contacts[(min(first, second), max(first, second))] = None
    return list(contacts), len(matches)


def rollout_position_map(document) -> dict[int, int]:
    """Map randomized multi-chain position tokens to concatenated indices."""

    mapping = {}
    offset = 0
    for n_term, length in zip(
        document.n_term_indices, document.chain_lengths, strict=True
    ):
        for local_index in range(length):
            position = (n_term + local_index) % NUM_POSITIONS
            if position in mapping:
                raise ValueError(f"position-token collision at p{position}")
            mapping[position] = offset + local_index
        offset += length
    if len(mapping) != document.seq_len:
        raise ValueError("position map does not cover the full complex")
    return mapping


def remove_linker_positions(
    mapping: dict[int, int], boundary: int, linker_length: int = 10
) -> dict[int, int]:
    """Drop linker tokens and restore original A+B coordinates."""
    return {
        token: index if index < boundary else index - linker_length
        for token, index in mapping.items()
        if not boundary <= index < boundary + linker_length
    }


def parse_arguments() -> argparse.Namespace:
    """Parse one worker invocation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest-b64", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--input-mode", choices=("multichain", "linker10"), default="multichain"
    )
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--contact-mult", type=int, default=DEFAULT_CONTACT_MULT)
    parser.add_argument("--accept-unfinished", action="store_true")
    parser.add_argument("--save-rollouts", action="store_true")
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
            residues = []
            chains = list(
                zip(record["chain_ids"], record["chain_sequences"], strict=True)
            )
            if arguments.input_mode == "linker10":
                chains = [("A", ("G" * 10).join(record["chain_sequences"]))]
            for chain_id, sequence in chains:
                for residue in residues_from_sequence(sequence, chain=chain_id):
                    residues.append(replace(residue, seq_index=len(residues)))
            first = len(prompts)
            position_maps: list[dict[int, int]] = []
            for rollout in range(arguments.n_rollouts):
                document = build_document(
                    f"{record['stem']}:r{rollout}",
                    residues,
                    [],
                    config=GenerationConfig(max_chains=2),
                )
                if document is None:
                    raise ValueError(f"{record['stem']}: prompt exceeds context")
                prompts.append(
                    document.document[: document.document.index(BEGIN) + len(BEGIN)]
                )
                mapping = rollout_position_map(document)
                if arguments.input_mode == "linker10":
                    mapping = remove_linker_positions(
                        mapping, record["chain_lengths"][0]
                    )
                position_maps.append(mapping)
            prompt_tokens = len(
                tokenizer(prompts[first], add_special_tokens=False).input_ids
            )
            max_tokens = generation_token_budget(
                prompt_tokens, record["L"], arguments.contact_mult
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
        rollout_rows: list[dict] = []
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
            chain_lengths = list(map(int, record["chain_lengths"]))
            votes = np.zeros((length, length), np.int32)
            parsed_contacts = 0
            valid_contacts = 0

            for rollout, (output, position_map) in enumerate(
                zip(record_outputs, position_maps, strict=True)
            ):
                sample = output.outputs[0]
                contacts, n_parsed = parse_rollout_contacts(
                    sample.text, position_map, chain_lengths
                )
                if arguments.save_rollouts:
                    rollout_rows.append(
                        {
                            "dataset": record["dataset"],
                            "stem": record["stem"],
                            "rollout": rollout,
                            "sampling_seed": arguments.seed * 1_000_003
                            + first
                            + rollout,
                            "finish_reason": sample.finish_reason,
                            "generated_tokens": len(sample.token_ids),
                            "parsed_contacts": n_parsed,
                            "contacts": contacts,
                        }
                    )
                if sample.finish_reason != "stop":
                    continue
                parsed_contacts += n_parsed
                valid_contacts += len(contacts)
                for pair in contacts:
                    votes[pair] += 1

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
                    "n_pairs": candidate_pair_count(chain_lengths),
                    "mode": "rollout_resample_" + arguments.input_mode,
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
                    "usable_rollouts": arguments.n_rollouts - unfinished,
                    "unfinished_rollouts": unfinished,
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
            if not arguments.accept_unfinished:
                raise RuntimeError(
                    f"{group_unfinished} rollout(s) hit the token cap; diagnostics at {failure_uri}"
                )

        score_uri = f"{output_directory}/scores/{part_stem}.parquet"
        timing_uri = f"{output_directory}/timings/{part_stem}.parquet"
        marker_uri = f"{output_directory}/complete/{part_stem}.json"
        rollout_uri = None
        if arguments.save_rollouts:
            rollout_uri = f"{output_directory}/rollouts/{part_stem}.parquet"
            write_parquet(
                pa.Table.from_pylist(rollout_rows, schema=ROLLOUT_SCHEMA), rollout_uri
            )
        write_parquet(pa.table(score_rows, schema=SCORE_SCHEMA), score_uri)
        write_parquet(
            pa.Table.from_pylist(timing_rows, schema=TIMING_SCHEMA), timing_uri
        )
        write_json(
            {
                "units": marker_units,
                "total_rollouts": len(group) * arguments.n_rollouts,
                "usable_rollouts": len(group) * arguments.n_rollouts - group_unfinished,
                "unfinished_rollouts": group_unfinished,
                "accepted_unfinished": bool(group_unfinished),
                "unfinished_details": unfinished_details,
                "score_uri": score_uri,
                "timing_uri": timing_uri,
                "rollout_uri": rollout_uri,
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
