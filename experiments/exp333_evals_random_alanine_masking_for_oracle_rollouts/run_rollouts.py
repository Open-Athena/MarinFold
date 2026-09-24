"""Generate contacts-v1 rollouts from independently alanine-masked sequences."""

import argparse
import hashlib
import platform
import re
import socket
import time
from datetime import UTC, datetime
from typing import Any

from masking_policy import (
    alanine_mask,
    stable_seed,
    validate_nested_masks,
    validate_prompt_mutation,
)

BEGIN = "<begin_statements>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NUM_POSITIONS = 2000
MIN_SEPARATION = 6
CONTEXT = 8192


def mode_name(fraction: float) -> str:
    """Return a stable path-safe arm name."""
    return f"mask_p{round(fraction * 1000):04d}"


def parse_rollout(
    text: str, position_to_sequence: dict[int, int]
) -> tuple[list[list[int]], int, int, int]:
    """Parse unique valid contacts and malformed/out-of-range diagnostics."""
    contacts: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    syntactic = CONTACT_RE.findall(text)
    malformed = text.count("<contact>") - len(syntactic)
    out_of_range = 0
    too_close = 0
    for raw_left, raw_right in syntactic:
        left = position_to_sequence.get(int(raw_left))
        right = position_to_sequence.get(int(raw_right))
        if left is None or right is None:
            out_of_range += 1
            continue
        pair = (min(left, right), max(left, right))
        if pair[1] - pair[0] < MIN_SEPARATION:
            too_close += 1
            continue
        if pair not in seen:
            seen.add(pair)
            contacts.append([pair[0], pair[1]])
    return contacts, malformed, out_of_range, too_close


def selected_targets(targets: Any, selection: str) -> Any:
    """Select the preregistered natural cohort."""
    if selection == "dev":
        return targets[(targets.cohort == "eval-val") & (targets.split == "dev")]
    if selection == "heldout":
        return targets[(targets.cohort == "eval-val") & (targets.split == "test")]
    if selection == "all":
        return targets[targets.cohort == "eval-val"]
    raise ValueError(f"unknown selection: {selection}")


def make_prompt(
    stem: str,
    sequence: str,
    rollout: int,
    fraction: float,
    build_document: Any,
    residues_from_sequence: Any,
    generation_config: Any,
) -> tuple[str, dict[int, int], list[int], str, str]:
    """Build one masked prompt while retaining native-sequence coordinates."""
    mutated, positions = alanine_mask(sequence, stem, rollout, fraction)
    entry_id = f"{stem}:r{rollout}"
    document = build_document(
        entry_id,
        residues_from_sequence(mutated),
        [],
        config=generation_config(),
    )
    if document is None:
        raise ValueError(f"{stem}: document builder rejected sequence")
    prefix = document.document[: document.document.index(BEGIN) + len(BEGIN)]
    native_document = build_document(
        entry_id,
        residues_from_sequence(sequence),
        [],
        config=generation_config(),
    )
    if native_document is None:
        raise ValueError(f"{stem}: document builder rejected native sequence")
    native_prefix = native_document.document[
        : native_document.document.index(BEGIN) + len(BEGIN)
    ]
    validate_prompt_mutation(native_prefix, prefix, len(positions))
    position_to_sequence = {
        (document.n_term_index + index) % NUM_POSITIONS: index
        for index in range(document.seq_len)
    }
    if len(position_to_sequence) != len(sequence):
        raise ValueError(f"{stem}: repeated position token")
    return prefix, position_to_sequence, positions, mutated, native_prefix


def write_parquet(uri: str, rows: list[dict[str, Any]]) -> None:
    """Write a compressed table through fsspec."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    with fsspec.open(uri, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle, compression="zstd")


def main() -> None:
    """Load the checkpoint once and generate resumable target/mutation arms."""
    import fsspec
    import pandas as pd
    import torch
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig,
        build_document,
        residues_from_sequence,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--selection", choices=["dev", "heldout", "all"], required=True)
    parser.add_argument("--fractions", default="0,0.05,0.10,0.20,0.40,1.0")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-fraction", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    fractions = sorted({float(value) for value in args.fractions.split(",")})
    if not fractions or fractions[0] < 0.0 or fractions[-1] > 1.0:
        raise ValueError("fractions must be a non-empty subset of [0, 1]")
    shard, n_shards = (int(value) for value in args.shard.split("/"))
    with fsspec.open(args.targets, "rt") as handle:
        targets = selected_targets(pd.read_csv(handle), args.selection)
    targets = targets.sort_values(["L", "stem"])
    mine = targets.iloc[[index for index in range(len(targets)) if index % n_shards == shard]]
    if args.limit is not None:
        mine = mine.iloc[: args.limit]
    if mine.empty:
        print(f"[exp333] no targets for shard {shard}/{n_shards}", flush=True)
        return

    pending: list[tuple[Any, float, str, str]] = []
    for target in mine.itertuples():
        for fraction in fractions:
            mode = mode_name(fraction)
            base = f"{args.out.rstrip('/')}/{mode}/eval-val/{target.stem}"
            raw_uri = f"{base}.parquet"
            timing_uri = f"{base}.timing.parquet"
            filesystem, raw_path = fsspec.core.url_to_fs(raw_uri)
            timing_path = fsspec.core.url_to_fs(timing_uri)[1]
            if not (filesystem.exists(raw_path) and filesystem.exists(timing_path)):
                pending.append((target, fraction, raw_uri, timing_uri))
    if not pending:
        print(f"[exp333] shard {shard}/{n_shards} already complete", flush=True)
        return

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    end_id = int(tokenizer.convert_tokens_to_ids("<end>"))
    if end_id == tokenizer.unk_token_id:
        raise ValueError("checkpoint tokenizer lacks <end>")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=CONTEXT,
        gpu_memory_utilization=args.gpu_fraction,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=args.max_num_seqs,
        seed=333,
    )
    model_load_seconds = time.perf_counter() - load_started
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
    load_share = model_load_seconds / len(pending)

    for number, (target, fraction, raw_uri, timing_uri) in enumerate(pending, start=1):
        job_started = time.perf_counter()
        prompts: list[str] = []
        position_maps: list[dict[int, int]] = []
        masks: list[list[int]] = []
        mutated_sequences: list[str] = []
        native_prompt_hashes: list[str] = []
        sampling: list[Any] = []
        for rollout in range(args.n_rollouts):
            prompt, position_map, positions, mutated, native_prompt = make_prompt(
                target.stem,
                target.sequence,
                rollout,
                fraction,
                build_document,
                residues_from_sequence,
                GenerationConfig,
            )
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            max_new = min(CONTEXT - prompt_tokens, 6 * int(target.L) + 128)
            if max_new < 1:
                raise ValueError(f"{target.stem}: no completion budget remains")
            prompts.append(prompt)
            position_maps.append(position_map)
            masks.append(positions)
            mutated_sequences.append(mutated)
            native_prompt_hashes.append(hashlib.sha256(native_prompt.encode()).hexdigest())
            sampling.append(
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=-1,
                    max_tokens=max_new,
                    stop_token_ids=[end_id],
                    skip_special_tokens=False,
                    seed=stable_seed(f"{target.stem}:r{rollout}:sample"),
                )
            )
        validate_nested_masks(
            target.sequence,
            [
                (value, alanine_mask(target.sequence, target.stem, 0, value)[1])
                for value in fractions
            ],
        )

        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        rows = []
        mode = mode_name(fraction)
        for rollout, output, position_map, positions, mutated, prompt, native_hash, params in zip(
            range(args.n_rollouts),
            outputs,
            position_maps,
            masks,
            mutated_sequences,
            prompts,
            native_prompt_hashes,
            sampling,
            strict=True,
        ):
            completion = output.outputs[0]
            contacts, malformed, out_of_range, too_close = parse_rollout(
                completion.text, position_map
            )
            rows.append(
                {
                    "rollout": rollout,
                    "mode": mode,
                    "contacts": contacts,
                    "mutated_positions": positions,
                    "mutated_sequence_sha256": hashlib.sha256(mutated.encode()).hexdigest(),
                    "n_mutated": len(positions),
                    "n_mutable": sum(residue != "A" for residue in target.sequence),
                    "native_alanine_fraction": target.sequence.count("A") / len(target.sequence),
                    "n_contacts": len(contacts),
                    "n_tokens": len(completion.token_ids),
                    "max_new": params.max_tokens,
                    "finished": completion.finish_reason == "stop",
                    "malformed_contacts": malformed,
                    "out_of_range_contacts": out_of_range,
                    "too_close_contacts": too_close,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "native_prompt_sha256": native_hash,
                    "cohort": "eval-val",
                    "dataset": target.dataset,
                    "stem": target.stem,
                    "L": int(target.L),
                    "mutation_fraction": fraction,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        write_parquet(raw_uri, rows)
        written = time.perf_counter()
        timing = [
            {
                "stem": target.stem,
                "cohort": "eval-val",
                "n_residues": int(target.L),
                "n_pairs": int(target.L) * (int(target.L) - 1) // 2,
                "mode": mode,
                "elapsed_seconds": elapsed,
                "model_load_seconds": load_share,
                "total_seconds": written - job_started + load_share,
                "model_load_accounting": "amortized_across_pending_target_modes",
                "n_rollouts": len(rows),
                "n_finished": sum(bool(row["finished"]) for row in rows),
                "n_malformed": sum(int(row["malformed_contacts"]) for row in rows),
                "generated_tokens": sum(int(row["n_tokens"]) for row in rows),
                "batch_size": args.max_num_seqs,
                "mutation_fraction": fraction,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                **metadata,
            }
        ]
        write_parquet(timing_uri, timing)
        print(
            f"[exp333] {number}/{len(pending)} {mode} {target.stem} L={target.L} "
            f"finished={timing[0]['n_finished']}/{len(rows)} "
            f"tokens={timing[0]['generated_tokens']} time={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
