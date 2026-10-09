"""Run one length-interleaved shard on the existing CoreWeave checkpoint.

One generation supplies all nested short-budget conditions. Their elapsed
seconds therefore refer to shared generation, never invented per-cap timings.
Raw completion token IDs allow auditing or rebuilding every vote matrix.
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
import transformers
import vllm
from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence
from marinfold.inference._tokenizer import model_source_path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from compute_metrics import metric_rows, resolved_pairs, true_matrix
from contacts import limits, sequence_pairs, snapshots

MODEL_NICKNAME = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
MODEL_URI = "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344"
BEGIN = "<begin_statements>"


def write_bytes(uri: str, content: bytes) -> None:
    """Persist one artifact through the worker's injected S3 filesystem."""
    with fsspec.open(uri, "wb") as handle:
        handle.write(content)


def write_json(uri: str, value: dict) -> None:
    """Persist JSON, preserving undefined metrics as null."""
    write_bytes(uri, (json.dumps(value, allow_nan=False) + "\n").encode())


def stage_model(local_model: str | None = None) -> tuple[str, float]:
    """Verify source ETags and stage the existing checkpoint in-region."""
    started = time.monotonic()
    manifest = json.loads(Path("publication_manifest.json").read_text())
    if local_model is not None:
        destination = Path(local_model)
        for name, entry in manifest["files"].items():
            file = destination / name
            with file.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if file.stat().st_size != entry["size"] or digest != entry["sha256"]:
                raise ValueError(f"local checkpoint identity mismatch: {name}")
        return model_source_path(destination), time.monotonic() - started
    destination = Path("/tmp/exp354-model")
    destination.mkdir(exist_ok=True)
    filesystem, root = fsspec.core.url_to_fs(MODEL_URI)
    for entry in manifest["source"]["checkpoint_files"]:
        remote = f"{root}/{entry['name']}"
        info = filesystem.info(remote)
        if info["size"] != entry["size"] or info["ETag"].strip('"') != entry["digest"]:
            raise ValueError(f"checkpoint identity mismatch: {entry['name']}")
        filesystem.get_file(remote, str(destination / entry["name"]))
    return model_source_path(destination), time.monotonic() - started


def make_prompts(record: dict, count: int, tokenizer) -> tuple[list[list[int]], list[int]]:
    """Use the same deterministic realization IDs as the reference evaluator."""
    residues = residues_from_sequence(record["input_seq"])
    texts, starts = [], []
    for rollout in range(count):
        document = build_document(f"{record['stem']}:r{rollout}", residues, [], config=GenerationConfig())
        texts.append(document.document[: document.document.index(BEGIN) + len(BEGIN)])
        starts.append(document.n_term_index)
    prompts = tokenizer(texts, add_special_tokens=False).input_ids
    assert len({len(prompt) for prompt in prompts}) == 1
    return prompts, starts


def generate(model, tokenizer, prompts: list[list[int]], *, cap: int | None,
             budget: int, seed: int) -> tuple[list[list[int]], list[bool], float, int]:
    """Sample until EOS or the emitted-contact cutoff, with a hard context guard.

The usual three-token contact format needs one call. Think tokens, malformed
triples, or retractions can require extra segments; no EOS suppression or
logit modification is used. Segments never read the evaluation targets.
    """
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    generated = [[] for _ in prompts]
    stopped = [False] * len(prompts)
    active = list(range(len(prompts)))
    elapsed = 0.0
    actual_tokens = 0
    segment = 0
    while active:
        parameters, inputs = [], []
        for index in active:
            remaining = budget - len(generated[index])
            if remaining <= 0:
                raise RuntimeError(f"short rollout {index} cannot reach contact budget {cap}")
            if cap is None:
                chunk = remaining
            else:
                _, count = snapshots(tokenizer.convert_ids_to_tokens(generated[index]), [cap])
                chunk = min(remaining, max(1, 3 * (cap - count)))
            inputs.append({"prompt_token_ids": prompts[index] + generated[index]})
            parameters.append(SamplingParams(temperature=1.0, top_p=0.95, top_k=-1,
                max_tokens=chunk, stop_token_ids=[end_id], skip_special_tokens=False,
                seed=seed + index + segment * 1_000_003))
        begin = time.monotonic()
        outputs = model.generate(inputs, parameters, use_tqdm=False)
        elapsed += time.monotonic() - begin
        next_active = []
        for index, output in zip(active, outputs, strict=True):
            completion = output.outputs[0]
            piece = list(completion.token_ids)
            actual_tokens += len(piece)
            generated[index].extend(piece)
            stopped[index] = completion.finish_reason == "stop"
            if cap is not None:
                state, count = snapshots(tokenizer.convert_ids_to_tokens(generated[index]), [cap])
                if count >= cap:
                    generated[index] = generated[index][:state[cap]["tokens"]]
                elif not stopped[index]:
                    next_active.append(index)
        active = next_active
        segment += 1
    return generated, stopped, elapsed, actual_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", help="Verified local published export; omit for CoreWeave S3")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path("targets.jsonl").read_text().splitlines()]
    assert len(records) == 97 and len({r["stem"] for r in records}) == 97
    records.sort(key=lambda r: (r["L"], r["stem"]))
    records = records[args.shard::args.num_shards]
    if args.limit:
        records = records[:args.limit]
    filesystem, _ = fsspec.core.url_to_fs(args.out)
    pending = [r for r in records if not filesystem.exists(f"{args.out}/complete/{r['stem']}.json")]
    if not pending:
        return
    model_path, stage_seconds = stage_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = json.loads((Path(model_path) / "config.json").read_text())
    assert config["rope_theta"] == 500_000
    assert config["rope_scaling"]["rope_type"] == "llama3"
    assert len(tokenizer) == config["vocab_size"]
    begin = time.monotonic()
    model = LLM(model=model_path, dtype="bfloat16", max_model_len=8192,
                gpu_memory_utilization=0.90, enable_prefix_caching=False,
                generation_config="vllm", max_num_seqs=256, seed=args.seed)
    load_seconds = time.monotonic() - begin
    properties = torch.cuda.get_device_properties(0)
    hardware = dict(gpu_name=properties.name, gpu_total_memory_gb=properties.total_memory / 2**30,
                    gpu_compute_capability=f"{properties.major}.{properties.minor}",
                    hostname=socket.gethostname(), platform=platform.platform(), torch_version=torch.__version__,
                    vllm_version=vllm.__version__, transformers_version=transformers.__version__)
    for record in pending:
        begin = time.monotonic()
        length = record["L"]
        caps = limits(length)
        prompts, starts = make_prompts(record, 1000, tokenizer)
        budget = min(8192 - len(prompts[0]), 6 * length + 128)
        seed = args.seed * 1_000_003 + int(hashlib.sha256(record["stem"].encode()).hexdigest()[:7], 16)
        short, short_stopped, short_elapsed, short_generated = generate(
            model, tokenizer, prompts, cap=max(caps.values()), budget=budget, seed=seed)
        full, full_stopped, full_elapsed, full_generated = generate(
            model, tokenizer, prompts[:100], cap=None, budget=budget, seed=seed)
        matrices = {f"short_{label}_n{n}": np.zeros((length, length), np.int16)
                    for label in caps for n in (100, 1000)}
        matrices["full_n100"] = np.zeros((length, length), np.int16)
        token_counts = dict.fromkeys(matrices, 0)
        early_counts = dict.fromkeys(matrices, 0)
        contact_counts = dict.fromkeys(matrices, 0)
        for index, (token_ids, start) in enumerate(zip(short, starts, strict=True)):
            states, _ = snapshots(tokenizer.convert_ids_to_tokens(token_ids), list(caps.values()))
            for label, cap in caps.items():
                state = states[cap]
                pairs = sequence_pairs(state["pairs"], start, length)
                for n in (100, 1000):
                    if index >= n:
                        continue
                    mode = f"short_{label}_n{n}"
                    token_counts[mode] += state["tokens"]
                    early_counts[mode] += state["emitted"] < cap
                    contact_counts[mode] += len(pairs)
                    for i, j in pairs:
                        matrices[mode][i, j] += 1
                        matrices[mode][j, i] += 1
        for token_ids, start, stopped in zip(full, starts[:100], full_stopped, strict=True):
            token_counts["full_n100"] += len(token_ids)
            if not stopped:
                continue
            states, _ = snapshots(tokenizer.convert_ids_to_tokens(token_ids), [budget + 1])
            pairs = sequence_pairs(states[budget + 1]["pairs"], start, length)
            contact_counts["full_n100"] += len(pairs)
            for i, j in pairs:
                matrices["full_n100"][i, j] += 1
                matrices["full_n100"][j, i] += 1
        truth = true_matrix(length, record["contacts"])
        pi, pj, separation = resolved_pairs(np.asarray(record["resolved"], dtype=np.int64))
        rows, timings = [], []
        for mode, matrix in matrices.items():
            metrics = metric_rows(matrix, truth, pi, pj, separation, length, with_precision=True)
            for row in metrics:
                if not np.isfinite(row["precision"]):
                    row["precision"] = None
            rows.extend({"stem": record["stem"], "mode": mode, "n_residues": length, **row} for row in metrics)
            is_full = mode == "full_n100"
            elapsed = full_elapsed if is_full else short_elapsed
            n = int(mode.rsplit("n", 1)[1])
            timings.append(dict(stem=record["stem"], n_residues=length,
                n_pairs=max(length - 6, 0) * max(length - 5, 0) // 2,
                mode=mode, elapsed_seconds=elapsed, model_load_seconds=load_seconds,
                total_seconds=time.monotonic() - begin + stage_seconds + load_seconds,
                model_nickname=MODEL_NICKNAME, runner_tag="local" if args.model else "iris-coreweave", **hardware,
                timestamp_utc=datetime.now(UTC).isoformat(), n_rollouts=n,
                usable_rollouts=sum(full_stopped) if is_full else n,
                unfinished_rollouts=100 - sum(full_stopped) if is_full else 0,
                early_eos=early_counts[mode], valid_contact_votes=contact_counts[mode],
                generated_tokens=token_counts[mode],
                actual_shared_generated_tokens=full_generated if is_full else short_generated,
                timing_scope="full_n100" if is_full else "shared_short_n1000_max_cap",
                input_tokens=len(prompts[0]) * n, seed=seed, max_tokens=budget,
                contact_cap=None if is_full else caps[mode.split("_")[1]],
                temperature=1.0, top_p=0.95, top_k=-1))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **matrices)
        write_bytes(f"{args.out}/scores/{record['stem']}.npz", buffer.getvalue())
        traces = dict(stem=record["stem"], n_term=starts, short=short, full=full,
                      short_eos=short_stopped, full_eos=full_stopped)
        write_bytes(f"{args.out}/traces/{record['stem']}.json.gz", gzip.compress(json.dumps(traces).encode()))
        total_seconds = time.monotonic() - begin + stage_seconds + load_seconds
        for timing in timings:
            timing["total_seconds"] = total_seconds
        result = dict(stem=record["stem"], metrics=rows, timings=timings,
                      checkpoint=MODEL_URI, stage_seconds=stage_seconds,
                      worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(f"{args.out}/complete/{record['stem']}.json", result)
        print(json.dumps(dict(event="protein_complete", stem=record["stem"], L=length,
                              short_seconds=short_elapsed, full_seconds=full_elapsed,
                              full_unfinished=100 - sum(full_stopped))), flush=True)
    print("SHARD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
