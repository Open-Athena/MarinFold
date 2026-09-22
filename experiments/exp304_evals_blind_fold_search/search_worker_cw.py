#!/usr/bin/env python
"""Generate sequence-only rollouts and self-conditioned branches on one H100.

The worker reads `search_targets.parquet`, which contains exactly pair_id,
sequence, and L. It never opens a reference structure or Fold1/Fold2 contact
set. One output parquet per pair preserves every complete generated contact map.
"""

import argparse
import hashlib
import os
import platform
import random
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence

from search_policy import ContactMap, branch_bundles, random_bundles

BEGIN = "<begin_statements>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NUM_POS = 2000
MIN_SEP = 6
CONTEXT = 8192
DEFAULT_MODEL = (
    "hf://buckets/open-athena/MarinFold/checkpoints/"
    "contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344"
)


def stable_seed(value: str) -> int:
    """Return a reproducible 31-bit seed independent of Python hash state."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def stage_model(source: str, destination: Path) -> Path:
    """Stage a public HF export on a pod before vLLM loads it."""
    if "://" not in source:
        return Path(source)
    destination.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(source)
    for info in fs.ls(root, detail=True):
        if info["type"] == "file":
            fs.get_file(info["name"], str(destination / os.path.basename(info["name"])))
    return destination


def parse_rollout(text: str, seq_index: dict[int, int]) -> ContactMap:
    """Decode contacts into sequence coordinates without reference filtering."""
    contacts: set[tuple[int, int]] = set()
    for token_a, token_b in CONTACT_RE.findall(text):
        a, b = seq_index.get(int(token_a)), seq_index.get(int(token_b))
        if a is None or b is None or a == b or abs(a - b) < MIN_SEP:
            continue
        contacts.add((min(a, b), max(a, b)))
    return frozenset(contacts)


def make_prompt(pair_id: str, sequence: str, arm: str, rollout: int,
                given: ContactMap) -> tuple[str, dict[int, int]]:
    """Build one fresh document realization and append any self-generated seeds."""
    residues = residues_from_sequence(sequence)
    doc = build_document(f"{pair_id}:{arm}:r{rollout}", residues, [], config=GenerationConfig())
    prefix = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
    seq_of_pos = {(doc.n_term_index + idx) % NUM_POS: idx for idx in range(doc.seq_len)}
    pos_of_seq = {idx: pos for pos, idx in seq_of_pos.items()}
    rng = random.Random(stable_seed(f"{pair_id}:{arm}:{rollout}:seed-order"))
    ordered = sorted(given)
    rng.shuffle(ordered)
    for i, j in ordered:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        prefix += f" <contact> <p{pos_of_seq[a]}> <p{pos_of_seq[b]}>"
    return prefix, seq_of_pos


def generate_arm(llm: LLM, tokenizer: AutoTokenizer, pair_id: str, sequence: str,
                 arm: str, bundles: list[ContactMap], n_rollouts: int,
                 temperature: float, top_p: float, prefix_offset: int,
                 gpu_meta: dict) -> tuple[list[dict], dict]:
    """Generate an arm and keep complete maps and compute metadata."""
    if not bundles:
        return [], {}
    L = len(sequence)
    prompts, mappings, assignments, params = [], [], [], []
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    if end_id is None or end_id < 0:
        raise ValueError("model tokenizer has no <end> token")
    for idx in range(n_rollouts):
        branch = idx % len(bundles)
        given = bundles[branch]
        prompt, mapping = make_prompt(pair_id, sequence, arm, prefix_offset + idx, given)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        max_new = min(CONTEXT - prompt_tokens, 6 * L + 128)
        if max_new < 1:
            raise ValueError(f"no context left for {pair_id} arm={arm} branch={branch}")
        prompts.append(prompt)
        mappings.append(mapping)
        assignments.append((branch, given, max_new))
        params.append(SamplingParams(
            temperature=temperature, top_p=top_p, top_k=-1, max_tokens=max_new,
            stop_token_ids=[end_id], skip_special_tokens=False,
        ))
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    rows = []
    for idx, (output, seq_index, assignment) in enumerate(zip(outputs, mappings, assignments)):
        branch, given, max_new = assignment
        result = output.outputs[0]
        contacts = parse_rollout(result.text, seq_index)
        rows.append({
            "pair_id": pair_id, "arm": arm, "candidate_id": f"{arm}:{idx}",
            "branch": branch, "rollout": idx, "L": L,
            "given": [list(pair) for pair in sorted(given)],
            "contacts": [list(pair) for pair in sorted(contacts)],
            "n_pred": len(contacts), "finished": result.finish_reason == "stop",
            "n_tokens": len(result.token_ids), "max_new": max_new,
        })
    timing = {
        "pair_id": pair_id, "stem": pair_id, "n_residues": L,
        "n_pairs": L * (L - 1) // 2, "mode": arm, "elapsed_seconds": elapsed,
        "n_rollouts": n_rollouts, "generated_tokens": sum(row["n_tokens"] for row in rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), **gpu_meta,
    }
    return rows, timing


def write_parquet(uri: str, rows: list[dict]) -> None:
    """Write rows to co-located CoreWeave object storage through fsspec."""
    with fsspec.open(uri, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-root", type=int, default=100)
    parser.add_argument("--n-arm", type=int, default=100)
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--arms", default="iid,temp,random,branch5,branch10,branch20")
    args = parser.parse_args()
    shard, n_shards = (int(part) for part in args.shard.split("/"))
    with fsspec.open(args.targets, "rb") as handle:
        targets = pq.read_table(handle).to_pylist()
    if {"pair_id", "sequence", "L"} != set(targets[0]):
        raise ValueError("worker target file must have exactly pair_id, sequence, L")
    targets.sort(key=lambda row: row["L"])
    mine = [row for idx, row in enumerate(targets) if idx % n_shards == shard]
    if args.limit is not None:
        mine = mine[: args.limit]
    print(f"[exp304] shard {shard}/{n_shards}: {len(mine)} targets, arms={args.arms}", flush=True)

    load_start = time.perf_counter()
    model_dir = stage_model(args.model, Path("/tmp/exp304-model"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=CONTEXT,
              gpu_memory_utilization=0.85, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=128, seed=0)
    model_load = time.perf_counter() - load_start
    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_meta = {
        "model_nickname": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "runner_tag": "iris-cw-rno2a", "gpu_name": gpu_properties.name,
        "gpu_total_memory_gb": gpu_properties.total_memory / 1e9,
        "gpu_compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch_version": str(torch.__version__),
    }
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    for position, record in enumerate(mine, 1):
        pair_id, sequence = record["pair_id"], record["sequence"]
        output_uri = f"{args.out.rstrip('/')}/shard-{shard:03d}-of-{n_shards:03d}-{pair_id}.parquet"
        timing_uri = f"{args.out.rstrip('/')}/timing-{shard:03d}-of-{n_shards:03d}-{pair_id}.parquet"
        fs, path = fsspec.core.url_to_fs(output_uri)
        if fs.exists(path) and fs.exists(fsspec.core.url_to_fs(timing_uri)[1]):
            print(f"[exp304] skip completed {pair_id}", flush=True)
            continue
        started = time.perf_counter()
        rows, root_timing = generate_arm(llm, tokenizer, pair_id, sequence, "root",
                                         [frozenset()], args.n_root, 1.0, 0.95, 0, gpu_meta)
        timings = [root_timing]
        root_maps = [frozenset(tuple(pair) for pair in row["contacts"]) for row in rows]
        if "iid" in arms:
            new, timing = generate_arm(llm, tokenizer, pair_id, sequence, "iid",
                                       [frozenset()], args.n_arm, 1.0, 0.95, 0, gpu_meta)
            rows.extend(new)
            timings.append(timing)
        if "temp" in arms:
            new, timing = generate_arm(llm, tokenizer, pair_id, sequence, "temp",
                                       [frozenset()], args.n_arm, 1.3, 1.0, 0, gpu_meta)
            rows.extend(new)
            timings.append(timing)
        if "random" in arms:
            bundles = random_bundles(root_maps, args.branches, 10, stable_seed(pair_id))
            new, timing = generate_arm(llm, tokenizer, pair_id, sequence, "random",
                                       bundles, args.n_arm, 1.0, 0.95, 0, gpu_meta)
            rows.extend(new)
            if timing:
                timings.append(timing)
        for k in (5, 10, 20):
            arm = f"branch{k}"
            if arm not in arms:
                continue
            bundles = branch_bundles(root_maps, args.branches, k)
            new, timing = generate_arm(llm, tokenizer, pair_id, sequence, arm,
                                       bundles, args.n_arm, 1.0, 0.95, 0, gpu_meta)
            rows.extend(new)
            if timing:
                timings.append(timing)
        for timing in timings:
            timing["model_load_seconds"] = model_load / (len(mine) * len(timings))
            timing["total_seconds"] = timing["elapsed_seconds"] + timing["model_load_seconds"]
        write_parquet(output_uri, rows)
        write_parquet(timing_uri, timings)
        print(f"[exp304] {position}/{len(mine)} {pair_id} L={len(sequence)} "
              f"maps={len(rows)} time={time.perf_counter()-started:.1f}s "
              f"finished={sum(row['finished'] for row in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
