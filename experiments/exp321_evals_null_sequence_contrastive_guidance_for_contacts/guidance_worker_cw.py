#!/usr/bin/env python
"""Generate paired native/null contacts-v1 rollouts on one CoreWeave H100."""

import argparse
import hashlib
import platform
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from guidance_policy import (
    AA_THREE,
    expects_position_token,
    guided_sample_logits,
    null_sequence,
    parse_contacts,
    stable_seed,
    validate_prompt_pair,
)

BEGIN = "<begin_statements>"
NUM_POSITIONS = 2000
CONTEXT = 8192


@dataclass(frozen=True)
class PromptPair:
    """One document realization and its token-aligned null counterpart."""

    rollout: int
    native_ids: list[int]
    null_ids: list[int]
    position_to_index: dict[int, int]
    max_new: int
    prompt_sha256: str
    null_prompt_sha256: str


def make_prompt_pair(
    stem: str,
    sequence: str,
    rollout: int,
    null_kind: str,
    tokenizer: Any,
    build_document: Any,
    residues_from_sequence: Any,
    generation_config: Any,
) -> PromptPair:
    """Build native and null prompts with identical non-amino-acid tokens."""
    entry_id = f"{stem}:r{rollout}"
    background_sequence = null_sequence(sequence, null_kind, stem)
    native_document = build_document(
        entry_id, residues_from_sequence(sequence), [], config=generation_config()
    )
    null_document = build_document(
        entry_id, residues_from_sequence(background_sequence), [], config=generation_config()
    )
    native_prefix = native_document.document[
        : native_document.document.index(BEGIN) + len(BEGIN)
    ]
    null_prefix = null_document.document[
        : null_document.document.index(BEGIN) + len(BEGIN)
    ]
    native_ids = list(tokenizer.encode(native_prefix, add_special_tokens=False))
    null_ids = list(tokenizer.encode(null_prefix, add_special_tokens=False))
    amino_acid_ids = {
        int(tokenizer.convert_tokens_to_ids(f"<{name}>")) for name in AA_THREE
    }
    validate_prompt_pair(native_ids, null_ids, amino_acid_ids, len(sequence))
    position_to_index = {}
    for residue_index in range(len(sequence)):
        position = (native_document.n_term_index + residue_index) % NUM_POSITIONS
        token_id = int(tokenizer.convert_tokens_to_ids(f"<p{position}>"))
        position_to_index[token_id] = residue_index
    if len(position_to_index) != len(sequence):
        raise ValueError(f"{stem}: repeated position token")
    max_new = min(CONTEXT - len(native_ids), 6 * len(sequence) + 128)
    if max_new < 1:
        raise ValueError(f"{stem}: no generation budget remains")
    return PromptPair(
        rollout=rollout,
        native_ids=native_ids,
        null_ids=null_ids,
        position_to_index=position_to_index,
        max_new=max_new,
        prompt_sha256=hashlib.sha256(native_prefix.encode()).hexdigest(),
        null_prompt_sha256=hashlib.sha256(null_prefix.encode()).hexdigest(),
    )


def _row_is_guided(
    generated: list[int],
    scope: str,
    contact_id: int,
    position_ids: set[int],
) -> bool:
    if scope == "all":
        return True
    if scope == "positions":
        return expects_position_token(generated, contact_id, position_ids)
    raise ValueError(f"unknown guidance scope: {scope}")


def decode_batch(
    model: Any,
    pairs: list[PromptPair],
    *,
    stem: str,
    mode: str,
    contact_id: int,
    end_id: int,
    gamma: float,
    scope: str,
    top_p: float,
    temperature: float,
    pure_ratio: bool,
) -> list[dict]:
    """Decode native streams while forcing every sampled token into null streams."""
    import torch

    if not pairs:
        return []
    prompt_lengths = {len(pair.native_ids) for pair in pairs}
    if len(prompt_lengths) != 1:
        raise ValueError("one decode batch must have equal-length prompts")
    prompt_length = prompt_lengths.pop()
    if any(len(pair.null_ids) != prompt_length for pair in pairs):
        raise ValueError("null prompt length mismatch")
    max_new_values = {pair.max_new for pair in pairs}
    if len(max_new_values) != 1:
        raise ValueError("one decode batch must have one token budget")
    max_new = max_new_values.pop()
    batch = len(pairs)
    device = next(model.parameters()).device
    prompt_rows = [pair.native_ids for pair in pairs] + [pair.null_ids for pair in pairs]
    input_ids = torch.tensor(prompt_rows, dtype=torch.long, device=device)
    generated: list[list[int]] = [[] for _ in pairs]
    token_log_ratios: list[list[float]] = [[] for _ in pairs]
    token_native_logprobs: list[list[float]] = [[] for _ in pairs]
    finished_flags = [False] * batch
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed(f"{stem}:batch:{pairs[0].rollout}:sample"))
    position_id_sets = [set(pair.position_to_index) for pair in pairs]

    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        for step in range(max_new):
            native_logits = logits[:batch]
            null_logits = logits[batch:]
            guide_rows = torch.tensor(
                [
                    _row_is_guided(
                        generated[row], scope, contact_id,
                        position_id_sets[row],
                    )
                    for row in range(batch)
                ],
                dtype=torch.bool,
                device=device,
            )
            sample_logits = guided_sample_logits(
                native_logits,
                null_logits,
                guide_rows,
                gamma=gamma,
                top_p=top_p,
                temperature=temperature,
                pure_ratio=pure_ratio,
            )
            probabilities = torch.softmax(sample_logits, dim=-1)
            sampled = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(-1)
            finished = torch.tensor(finished_flags, dtype=torch.bool, device=device)
            sampled = torch.where(finished, torch.full_like(sampled, end_id), sampled)
            native_logprob = torch.log_softmax(native_logits.float(), dim=-1).gather(
                -1, sampled[:, None]
            ).squeeze(-1)
            null_logprob = torch.log_softmax(null_logits.float(), dim=-1).gather(
                -1, sampled[:, None]
            ).squeeze(-1)
            sampled_values = sampled.tolist()
            native_values = native_logprob.tolist()
            ratio_values = (native_logprob - null_logprob).tolist()
            for row in range(batch):
                if finished_flags[row]:
                    continue
                token = int(sampled_values[row])
                generated[row].append(token)
                token_native_logprobs[row].append(float(native_values[row]))
                token_log_ratios[row].append(float(ratio_values[row]))
                if token == end_id:
                    finished_flags[row] = True
            if all(finished_flags):
                break
            paired_next = torch.cat([sampled, sampled])[:, None]
            output = model(
                input_ids=paired_next,
                past_key_values=cache,
                cache_position=torch.tensor([prompt_length + step], device=device),
                use_cache=True,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1, :]

    rows = []
    for row, pair in enumerate(pairs):
        contacts, contact_log_ratios, malformed_contacts = parse_contacts(
            generated[row], token_log_ratios[row], contact_id, pair.position_to_index
        )
        rows.append({
            "rollout": pair.rollout,
            "mode": mode,
            "contacts": contacts,
            "contact_log_ratios": contact_log_ratios,
            "n_contacts": len(contacts),
            "n_tokens": len(generated[row]),
            "max_new": pair.max_new,
            "finished": bool(generated[row] and generated[row][-1] == end_id),
            "malformed_contacts": malformed_contacts,
            "native_nll": float(-sum(token_native_logprobs[row])),
            "mean_token_log_ratio": (
                float(sum(token_log_ratios[row]) / len(token_log_ratios[row]))
                if token_log_ratios[row] else float("nan")
            ),
            "prompt_sha256": pair.prompt_sha256,
            "null_prompt_sha256": pair.null_prompt_sha256,
        })
    return rows


def write_parquet(uri: str, rows: list[dict]) -> None:
    """Write a compressed result table through fsspec."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    with fsspec.open(uri, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle, compression="zstd")


def _select_targets(targets: list[dict], selection: str) -> list[dict]:
    if selection == "dev":
        return [row for row in targets if row["split"] == "dev" and bool(row["primary"])]
    if selection == "eval-holdout":
        return [row for row in targets if row["cohort"] == "eval-val" and row["split"] == "test"]
    if selection == "eval-all":
        return [row for row in targets if row["cohort"] == "eval-val"]
    if selection == "foldswitch-test":
        return [
            row for row in targets
            if row["cohort"] == "foldswitch" and row["split"] == "test"
            and bool(row["primary"])
        ]
    raise ValueError(f"unknown selection: {selection}")


def main() -> None:
    """Load one checkpoint and run one interleaved target shard."""
    import fsspec
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
        residues_from_sequence,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument(
        "--selection",
        choices=["dev", "eval-holdout", "eval-all", "foldswitch-test"],
        required=True,
    )
    parser.add_argument("--null", choices=["polyala", "polylys", "shuffle"], required=True)
    parser.add_argument("--scope", choices=["positions", "all"], default="positions")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--pure-ratio", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    shard, n_shards = (int(value) for value in args.shard.split("/"))
    with fsspec.open(args.targets, "rt") as handle:
        targets = pd.read_csv(handle).to_dict("records")
    targets = _select_targets(targets, args.selection)
    targets.sort(key=lambda row: (int(row["L"]), row["target_id"]))
    mine = [row for index, row in enumerate(targets) if index % n_shards == shard]
    if args.limit is not None:
        mine = mine[:args.limit]
    if not mine:
        print(f"[exp321] no targets for shard {shard}/{n_shards}", flush=True)
        return
    print(
        f"[exp321] shard {shard}/{n_shards}: {len(mine)} targets mode={args.mode} ",
        f"gamma={args.gamma} null={args.null} scope={args.scope}",
        flush=True,
    )

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda").eval()
    model_load = time.perf_counter() - load_start
    contact_id = int(tokenizer.convert_tokens_to_ids("<contact>"))
    end_id = int(tokenizer.convert_tokens_to_ids("<end>"))
    if contact_id == tokenizer.unk_token_id or end_id == tokenizer.unk_token_id:
        raise ValueError("checkpoint tokenizer lacks <contact> or <end>")
    gpu = torch.cuda.get_device_properties(0)
    metadata = {
        "model_nickname": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "runner_tag": "iris-cw-rno2a",
        "gpu_name": gpu.name,
        "gpu_total_memory_gb": gpu.total_memory / 1e9,
        "gpu_compute_capability": f"{gpu.major}.{gpu.minor}",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    for target_number, target in enumerate(mine, 1):
        stem = target["stem"]
        sequence = target["sequence"]
        cohort = target["cohort"]
        base = f"{args.out.rstrip('/')}/{args.mode}/{cohort}/{stem}"
        raw_uri = f"{base}.parquet"
        timing_uri = f"{base}.timing.parquet"
        filesystem, raw_path = fsspec.core.url_to_fs(raw_uri)
        timing_path = fsspec.core.url_to_fs(timing_uri)[1]
        if filesystem.exists(raw_path) and filesystem.exists(timing_path):
            print(f"[exp321] skip {stem}", flush=True)
            continue
        prompt_pairs = [
            make_prompt_pair(
                stem,
                sequence,
                rollout,
                args.null,
                tokenizer,
                build_document,
                residues_from_sequence,
                GenerationConfig,
            )
            for rollout in range(args.n_rollouts)
        ]
        started = time.perf_counter()
        rows = []
        for first in range(0, len(prompt_pairs), args.batch_size):
            rows.extend(
                decode_batch(
                    model,
                    prompt_pairs[first:first + args.batch_size],
                    stem=stem,
                    mode=args.mode,
                    contact_id=contact_id,
                    end_id=end_id,
                    gamma=args.gamma,
                    scope=args.scope,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    pure_ratio=args.pure_ratio,
                )
            )
        elapsed = time.perf_counter() - started
        for row in rows:
            row.update({
                "cohort": cohort,
                "dataset": target["dataset"],
                "stem": stem,
                "L": len(sequence),
                "gamma": args.gamma,
                "null_kind": args.null,
                "guidance_scope": args.scope,
                "pure_ratio": args.pure_ratio,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })
        write_parquet(raw_uri, rows)
        written = time.perf_counter()
        timing = [{
            "stem": stem,
            "cohort": cohort,
            "n_residues": len(sequence),
            "n_pairs": len(sequence) * (len(sequence) - 1) // 2,
            "mode": args.mode,
            "elapsed_seconds": elapsed,
            "model_load_seconds": model_load / len(mine),
            "total_seconds": written - started + model_load / len(mine),
            "n_rollouts": args.n_rollouts,
            "n_finished": sum(bool(row["finished"]) for row in rows),
            "n_malformed": sum(int(row["malformed_contacts"]) for row in rows),
            "generated_tokens": sum(int(row["n_tokens"]) for row in rows),
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "null_kind": args.null,
            "guidance_scope": args.scope,
            "pure_ratio": args.pure_ratio,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }]
        write_parquet(timing_uri, timing)
        print(
            f"[exp321] {target_number}/{len(mine)} {stem} L={len(sequence)} ",
            f"finished={timing[0]['n_finished']}/{len(rows)} time={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
