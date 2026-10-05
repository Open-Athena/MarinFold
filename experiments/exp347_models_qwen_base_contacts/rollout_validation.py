"""Greedy document-completion diagnostic on the experiment's held-out AFDB pool.

This is a format/learning diagnostic, not the exp89/exp245 rollout-plus-resample
benchmark. Outputs, truncation flags, and per-input timings are retained so sparse
or malformed generations cannot be silently dropped from the reported averages.
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import torch
import torch.distributed as dist
from common import FORMATS, MAX_LENGTH, convert_document
from train import DocumentStream, copy_tree_from_remote
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


def parse_contacts(
    text: str, document_format: str, length: int, start_index: int
) -> tuple[set[tuple[int, int]], int]:
    """Read pairs, counting malformed statements, invalid endpoints, and repeats."""
    text = text.removesuffix("<|endoftext|>").strip()
    invalid = 0
    pairs = []
    if document_format == "contacts_v1":
        pattern = r"<contact>\s*<p(\d+)>\s*<p(\d+)>"
        matches = re.findall(pattern, text)
        remainder = re.sub(pattern, "", text).removesuffix("<end>").strip()
        invalid += remainder.count("<contact>") or int(bool(remainder))
        for a, b in matches:
            if not (0 <= int(a) < 2000 and 0 <= int(b) < 2000):
                invalid += 1
                continue
            pairs.append(
                ((int(a) - start_index) % 2000 + 1, (int(b) - start_index) % 2000 + 1)
            )
    else:
        for line in text.splitlines():
            if not line.strip() or line.strip() == "END":
                continue
            match = re.fullmatch(r"[ \t]*(\d+)[ \t]+(\d+)[ \t]*", line)
            if match is None:
                invalid += 1
            else:
                pairs.append((int(match[1]), int(match[2])))
    valid: set[tuple[int, int]] = set()
    for a, b in pairs:
        pair = (min(a, b), max(a, b))
        if (
            not (1 <= a <= length and 1 <= b <= length)
            or abs(a - b) < 6
            or pair in valid
        ):
            invalid += 1
        else:
            valid.add(pair)
    return valid, invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Saved HF model/tokenizer directory URI"
    )
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if args.documents % world:
        raise ValueError("Document count must be divisible by world size")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    began = time.perf_counter()
    local = Path("/tmp/exp347-rollout-model")
    if rank == 0:
        if local.exists():
            shutil.rmtree(local)
        copy_tree_from_remote(args.checkpoint, local)
    dist.barrier()
    model = (
        Qwen3_5ForCausalLM.from_pretrained(
            local, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(local)
    load_seconds = time.perf_counter() - began
    stream = DocumentStream(args.data + "/validation", args.format, rank, world)
    rows = [stream.next() for _ in range(args.documents // world)]
    props = torch.cuda.get_device_properties(rank)
    metrics, timings, outputs = [], [], []
    with torch.inference_mode():
        for row in rows:
            total_start = time.perf_counter()
            raw = tokenizer.decode(row["contacts_v1"][:-1])
            document = convert_document(raw)
            prompt = (
                document.raw_prefix
                if args.format == "contacts_v1"
                else document.prompted_prefix
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            start_match = re.search(r"<n-term>\s*<p(\d+)>", raw)
            if start_match is None:
                raise ValueError("Reference has no N-terminus")
            n_term = int(start_match[1])
            ids = torch.tensor([prompt_ids], device="cuda")
            limit = min(args.max_new_tokens, MAX_LENGTH - len(prompt_ids))
            if limit < 1:
                raise ValueError("No remaining generation context")
            stop = "<end>" if args.format == "contacts_v1" else "\nEND"
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=limit,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=[stop],
                tokenizer=tokenizer,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - inference_start
            tail = generated[0, len(prompt_ids) :].tolist()
            text = tokenizer.decode(tail, skip_special_tokens=False)
            predictions, invalid = parse_contacts(
                text, args.format, len(row["sequence"]), n_term
            )
            truth = {tuple(sorted(pair)) for pair in row["contacts"]}
            hits = len(predictions & truth)
            precision = hits / len(predictions) if predictions else 0.0
            recall = hits / len(truth)
            ended = stop in text or tokenizer.eos_token_id in tail
            metrics.append(
                {
                    "stem": row["entry_id"],
                    "n_residues": len(row["sequence"]),
                    "true_contacts": len(truth),
                    "predicted_contacts": len(predictions),
                    "correct_contacts": hits,
                    "invalid_or_duplicate_pairs": invalid,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall)
                    if hits
                    else 0.0,
                    "hit_token_cap": len(tail) == limit and not ended,
                    "format_end_emitted": stop in text,
                    "generated_tokens": len(tail),
                }
            )
            outputs.append(
                {"stem": row["entry_id"], "prompt": prompt, "completion": text}
            )
            timings.append(
                {
                    "stem": row["entry_id"],
                    "n_residues": len(row["sequence"]),
                    "n_pairs": len(truth),
                    "mode": f"{args.format}-greedy",
                    "elapsed_seconds": elapsed,
                    "model_load_seconds": load_seconds,
                    "total_seconds": time.perf_counter() - total_start,
                    "model_nickname": args.checkpoint,
                    "runner_tag": "iris",
                    "gpu_name": props.name,
                    "gpu_total_memory_gb": props.total_memory / 1e9,
                    "gpu_compute_capability": f"{props.major}.{props.minor}",
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "torch_version": str(torch.__version__),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
            )
    for name, records in [("metrics", metrics), ("timings", timings)]:
        with fsspec.open(f"{args.out}/{name}-rank-{rank}.csv", "w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(records)
    with fsspec.open(f"{args.out}/outputs-rank-{rank}.json", "w") as handle:
        json.dump(outputs, handle)
    print(
        json.dumps(
            {
                "rank": rank,
                "documents": len(metrics),
                "mean_f1": float(np.mean([m["f1"] for m in metrics])),
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
