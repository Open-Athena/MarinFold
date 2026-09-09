"""Batched vLLM construction of bootstrap or on-policy whole-trajectory datasets.

Each part contains every candidate for each of its targets. A part manifest is
written last and checked on resume, preventing a refresh from reusing samples
from different weights or sampling settings. Timing CSVs report batch latency
per input explicitly; they do not pretend concurrent requests ran sequentially.
"""

import argparse
import csv
import importlib
import itertools
import os
import platform
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fsspec
import torch
from marinfold.document_structures.contacts_v1_multi import (
    BEGIN,
    END,
    FINAL,
    MULTI,
    parse_history,
    truncate_history,
)
from marinfold.inference._tokenizer import model_source_path
from transformers import AutoTokenizer

from common import code_identity, identity, read_json, rows, seed_for, stage_model, write_json, write_rows
from corpus import contact_score, tokenize_exact


def decode_pairs(pairs: tuple, positions: list[int]) -> list[tuple[int, int]]:
    """Map ring indices to sequence indices, rejecting physically invalid pairs."""
    indices = {p: i for i, p in enumerate(positions)}
    result = []
    for a, b in pairs:
        if a not in indices or b not in indices or abs(indices[a] - indices[b]) < 6:
            raise ValueError("out-of-sequence or short-range generated contact")
        result.append(tuple(sorted((indices[a], indices[b]))))
    return result


def scored_candidate(target: dict, tokens: list[str], *, forced: bool, budget: int,
                     candidate_id: int, generator: str) -> dict[str, Any]:
    """Preserve malformed outputs as explicit invalid candidates for auditing."""
    record = {"target_id": target["target_id"], "header": target["header"],
              "reference": target["reference"], "positions": target["positions"],
              "n_residues": target["n_residues"], "forced": forced, "budget": budget,
              "candidate_id": candidate_id, "generator": generator, "generated": tokens,
              "history": [], "prediction": [], "valid": False, "error": "",
              "score": {"precision": 0.0, "recall": 0.0, "f1": -1.0}}
    truth = parse_history([FINAL, *target["reference"], END]).final
    reference = decode_pairs(truth, target["positions"])
    try:
        parsed = parse_history(tokens)
        for hypothesis in parsed.hypotheses:
            decode_pairs(hypothesis, target["positions"])
        prediction = decode_pairs(parsed.final, target["positions"])
    except ValueError as exc:
        # Invalid *model outputs* are data, not swallowed infrastructure errors.
        record["error"] = str(exc)
        return record
    record.update(history=tokens[:parsed.final_index], prediction=prediction, valid=True,
                  score=contact_score(prediction, reference))
    return record


def output_tokens(output: Any, tokenizer: Any, stop_id: int) -> list[str]:
    """Normalize vLLM's stop-token convention without accepting length truncation."""
    completion = output.outputs[0]
    ids = list(completion.token_ids)
    if completion.finish_reason == "stop" and completion.stop_reason == stop_id:
        if not ids or ids[-1] != stop_id:
            ids.append(stop_id)
    return tokenizer.convert_ids_to_tokens(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True, help="prepared target manifest")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["bootstrap", "synthesis"], default="synthesis")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--bootstrap-hypotheses", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 256, 1024, 2048])
    parser.add_argument("--forced-fraction", type=float, default=0.5)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-targets", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/exp281-models"))
    parser.add_argument("--enforce-eager", action="store_true", help="skip compilation for tiny-model smoke tests")
    args = parser.parse_args()
    if not 0 <= args.forced_fraction <= 1 or args.candidates < 1:
        raise ValueError("invalid generation mixture or candidate count")
    if not 0 <= args.shard_index < args.shard_count or min(args.budgets) < 0:
        raise ValueError("invalid shard or token budget")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["code_hash"] = code_identity()
    targets_manifest = read_json(args.targets)
    signature = identity({"config": config, "targets": targets_manifest})
    load_start = time.monotonic()
    model_path = model_source_path(stage_model(args.model, args.cache))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # The optional vLLM dependency is loaded only by the accelerator worker.
    vllm = importlib.import_module("vllm")
    with socket.socket() as port:
        port.bind(("", 0))
        os.environ["VLLM_PORT"] = str(port.getsockname()[1])
    engine = vllm.LLM(model=str(model_path), dtype="bfloat16", max_model_len=args.context,
                      gpu_memory_utilization=0.85, enable_prefix_caching=True, enforce_eager=args.enforce_eager)
    load_seconds = time.monotonic() - load_start
    end_id = tokenizer.convert_tokens_to_ids(END)
    if args.phase == "synthesis":
        tokenize_exact(tokenizer, [MULTI, FINAL])
    device = torch.cuda.get_device_properties(0)
    worker = {"model_load_seconds": load_seconds, "model_nickname": args.model,
              "runner_tag": "iris" if os.environ.get("IRIS_JOB_ID") else "local",
              "gpu_name": device.name, "gpu_total_memory_gb": device.total_memory / 1e9,
              "gpu_compute_capability": f"{device.major}.{device.minor}",
              "hostname": socket.gethostname(), "platform": platform.platform(),
              "torch_version": torch.__version__, "n_samples_per_seed": args.candidates}
    paths = targets_manifest["shards"][args.split]
    if args.shard_count > len(paths):
        raise ValueError("more generation workers than target shards")
    for file_index, path in enumerate(paths):
        if file_index % args.shard_count != args.shard_index:
            continue
        iterator = rows(path)
        part = 0
        while chunk := list(itertools.islice(iterator, args.batch_targets)):
            prefix = f"{args.output}/shard-{file_index:05d}-part-{part:05d}"
            part += 1
            fs, success = fsspec.core.url_to_fs(prefix + ".json")
            if fs.exists(success):
                if read_json(prefix + ".json")["signature"] != signature:
                    raise ValueError("output belongs to another generation configuration")
                continue
            prompts, parameters, specs = [], [], []
            for target in chunk:
                rng = random.Random(seed_for(args.seed, target["target_id"], "budget"))
                forced = rng.random() < args.forced_fraction
                reserve = 6 * target["n_residues"] + 128
                available = args.context - len(target["header"]) - reserve - 1
                if available < 0 or len(target["reference"]) + 1 > reserve:
                    raise ValueError(f"target cannot fit fixed length-based answer reserve: {target['target_id']}")
                budget = min(rng.choice(args.budgets), available) if forced else available
                for candidate_id in range(args.candidates):
                    header = [MULTI, *target["header"][1:]]
                    spec = (target, forced, budget, candidate_id, reserve)
                    if args.phase == "bootstrap":
                        for hypothesis in range(args.bootstrap_hypotheses):
                            prompts.append({"prompt_token_ids": tokenize_exact(tokenizer, [*target["header"], BEGIN])})
                            parameters.append(vllm.SamplingParams(
                                max_tokens=reserve, temperature=args.temperature, top_p=args.top_p, top_k=-1,
                                stop_token_ids=[end_id], seed=seed_for(args.seed, target["target_id"], candidate_id, hypothesis)))
                        specs.append(spec)
                    else:
                        prompt = [*header, FINAL] if forced and budget == 0 else [*header, BEGIN]
                        prompts.append({"prompt_token_ids": tokenize_exact(tokenizer, prompt)})
                        parameters.append(vllm.SamplingParams(
                            max_tokens=reserve if forced and budget == 0 else (max(1, budget - 1) if forced else budget + reserve),
                            temperature=args.temperature, top_p=args.top_p, top_k=-1,
                            stop_token_ids=[end_id], bad_words=[FINAL, END] if forced and budget > 0 else None,
                            seed=seed_for(args.seed, target["target_id"], candidate_id, "history")))
                        specs.append(spec)
            started = time.monotonic()
            outputs = engine.generate(prompts, parameters, use_tqdm=False)
            elapsed = time.monotonic() - started
            histories, second_prompts, second_parameters = [], [], []
            for index, (target, forced, budget, candidate_id, reserve) in enumerate(specs):
                if args.phase == "bootstrap":
                    tokens = []
                    for h in range(args.bootstrap_hypotheses):
                        draft = output_tokens(outputs[index * args.bootstrap_hypotheses + h], tokenizer, end_id)
                        if not draft or draft[-1] != END:
                            raise ValueError("bootstrap hypothesis failed to terminate")
                        parse_history([FINAL, *draft[:-1], END])
                        tokens.extend([BEGIN, *draft[:-1]])
                    histories.append(truncate_history(tokens, budget))
                elif forced and budget > 0:
                    tokens = [BEGIN, *output_tokens(outputs[index], tokenizer, end_id)]
                    history = truncate_history(tokens, budget)
                    histories.append(history)
                    second_prompts.append({"prompt_token_ids": tokenize_exact(tokenizer, [MULTI, *target["header"][1:], *history, FINAL])})
                    second_parameters.append(vllm.SamplingParams(
                        max_tokens=reserve, temperature=args.temperature, top_p=args.top_p, top_k=-1,
                        stop_token_ids=[end_id], seed=seed_for(args.seed, target["target_id"], candidate_id, "final")))
                else:
                    histories.append(None)
            second_started = time.monotonic()
            finals = iter(engine.generate(second_prompts, second_parameters, use_tqdm=False)) if second_prompts else iter([])
            elapsed += time.monotonic() - second_started
            records = []
            for index, (target, forced, budget, candidate_id, reserve) in enumerate(specs):
                if args.phase == "bootstrap":
                    tokens = [*histories[index], FINAL, *target["reference"], END]
                elif histories[index] is not None:
                    tokens = [*histories[index], FINAL, *output_tokens(next(finals), tokenizer, end_id)]
                else:
                    tokens = [FINAL if forced else BEGIN, *output_tokens(outputs[index], tokenizer, end_id)]
                record = scored_candidate(target, tokens, forced=forced, budget=budget,
                                          candidate_id=candidate_id, generator=args.model)
                record["bootstrap"] = args.phase == "bootstrap"
                if record["bootstrap"]:
                    # The reference closure taught in warm-up is not a sampled
                    # answer and must never masquerade as perfect model accuracy.
                    record["generated"] = list(histories[index])
                    record["prediction"] = []
                    record["score"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                records.append(record)
            write_rows(prefix + ".parquet", records)
            total = time.monotonic() - started
            timings = [{**worker, "stem": t["target_id"], "n_residues": t["n_residues"],
                        "n_pairs": len(t["reference"]) // 3, "mode": args.phase,
                        "elapsed_seconds": elapsed, "total_seconds": total + load_seconds,
                        "batch_size": len(prompts), "timing_scope": "shared_batch_latency",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()} for t in chunk]
            with fsspec.open(prefix + "-timings.csv", "w", auto_mkdir=True) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
                writer.writeheader()
                writer.writerows(timings)
            write_json(prefix + ".json", {"signature": signature, "config": config,
                       "targets": len(chunk), "candidates": len(records),
                       "invalid": sum(not r["valid"] for r in records)})
            print(f"published {prefix}: {len(records)} candidates", flush=True)


if __name__ == "__main__":
    main()
