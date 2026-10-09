"""Generate paired, resumable contact rollouts on one independent H100.

Uses exp82's prompt resampling and decoding knobs, with per-protein stable
sampling seeds shared by both checkpoints. Extra samples and true-contact
prefixes are diagnostics; the headline always uses the first 100 unconditioned
rollouts. Raw samples are durable, and token-capped samples remain observable.
"""

import argparse
import gzip
import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import torch
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)
from marinfold.inference._tokenizer import model_source_path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from protocol import BEGIN, live_contacts, oracle_pairs, stable_seed
from scoring import analyze_record, summarize_samples


def write_json(uri: str, value: object) -> None:
    """Publish a single complete JSON object through the configured store."""
    with fsspec.open(uri, "wt") as handle:
        json.dump(value, handle)


def stage_model(checkpoint: dict) -> tuple[Path, float]:
    """Verify the pinned S3 export and stage it inside its compute region."""
    started = time.monotonic()
    filesystem, prefix = fsspec.core.url_to_fs(checkpoint["coreweave_uri"])
    destination = Path("/tmp/exp357-model")
    destination.mkdir(exist_ok=True)
    for entry in checkpoint["checkpoint_files"]:
        source = prefix + "/" + entry["name"]
        info = filesystem.info(source)
        etag = str(info.get("ETag", info.get("etag", ""))).strip('"')
        if info["size"] != entry["size"] or etag != entry["digest"]:
            raise ValueError(f"Checkpoint identity mismatch: {source}")
        filesystem.get_file(source, str(destination / entry["name"]))
    return Path(model_source_path(destination)), time.monotonic() - started


def generate(
    model: LLM,
    tokenizer: AutoTokenizer,
    record: dict,
    *,
    oracle: bool,
    full_context: bool = False,
) -> tuple[list[dict], dict]:
    """Time one protein and decode all samples to canonical live contacts."""
    count = 100 if oracle or not record["diagnostic"] else 1000
    residues = residues_from_sequence(record["input_seq"])
    supplied = oracle_pairs(record) if oracle else []
    prompts, params, nterms, caps, lengths = [], [], [], [], []
    for index in range(count):
        document = build_document(
            f"{record['stem']}:r{index}", residues, [], config=GenerationConfig()
        )
        prompt = document.document.split(BEGIN)[0] + BEGIN
        nterm = document.n_term_index
        for i, j in supplied:
            prompt += f" <contact> <p{(nterm + i) % 2000}> <p{(nterm + j) % 2000}>"
        length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        cap = (
            8192 - length if full_context else min(8192 - length, 6 * record["L"] + 128)
        )
        if cap <= 0:
            raise ValueError(f"No output context available: {record['stem']}")
        prompts.append(prompt)
        nterms.append(nterm)
        caps.append(cap)
        lengths.append(length)
        params.append(
            SamplingParams(
                temperature=1.0,
                top_p=0.95,
                top_k=-1,
                max_tokens=cap,
                stop_token_ids=[tokenizer.convert_tokens_to_ids("<end>")],
                skip_special_tokens=False,
                seed=stable_seed(record["dataset"], record["stem"], index, "sampling"),
            )
        )
    start = time.monotonic()
    outputs = model.generate(prompts, params, use_tqdm=False)
    elapsed = time.monotonic() - start
    result = []
    for index, output in enumerate(outputs):
        generated = output.outputs[0]
        result.append(
            dict(
                rollout=index,
                contacts=live_contacts(generated.text, nterms[index], record["L"]),
                finish_reason=generated.finish_reason,
                generated_tokens=len(generated.token_ids),
                max_tokens=caps[index],
                prompt_tokens=lengths[index],
                sampling_seed=params[index].seed,
            )
        )
    return result, dict(
        elapsed_seconds=elapsed,
        n_rollouts=count,
        stopped_rollouts=sum(r["finish_reason"] == "stop" for r in result),
        unfinished_rollouts=sum(r["finish_reason"] != "stop" for r in result),
        generated_tokens=sum(r["generated_tokens"] for r in result),
        supplied_contacts=len(supplied),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--include", nargs="+")
    args = parser.parse_args()
    checkpoint = json.loads(Path(args.checkpoint).read_text())
    with fsspec.open(args.targets, "rt") as handle:
        records = json.load(handle)
    shard, count = map(int, args.shard.split("/"))
    records.sort(key=lambda r: (r["L"], r["dataset"], r["stem"]))
    records = records[shard::count]
    if args.include:
        selected = set(args.include)
        records = [r for r in records if f"{r['dataset']}__{r['stem']}" in selected]
        if len(records) != len(selected):
            raise ValueError("Requested diagnostic keys were not all found")
    if args.smoke:
        records = [next(r for r in records if r["diagnostic"] and 100 <= r["L"] <= 200)]
    filesystem, prefix = fsspec.core.url_to_fs(args.out)
    pending = [
        r
        for r in records
        if not filesystem.exists(
            f"{prefix}/{checkpoint['label']}/complete/{r['dataset']}__{r['stem']}.json"
        )
    ]
    print(
        json.dumps(
            {
                "event": "assigned",
                "checkpoint": checkpoint["label"],
                "shard": args.shard,
                "assigned": len(records),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        return
    model_path, stage_seconds = stage_model(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    config = json.loads((model_path / "config.json").read_text())
    if (
        config["rope_theta"] != 500_000
        or config["rope_scaling"]["rope_type"] != "llama3"
    ):
        raise ValueError("Checkpoint RoPE translation failed")
    if len(tokenizer) != config["vocab_size"] or len(tokenizer) != 2845:
        raise ValueError("Tokenizer vocabulary mismatch")
    start = time.monotonic()
    model = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=512,
        seed=357,
    )
    load_seconds = time.monotonic() - start
    gpu = torch.cuda.get_device_properties(0)
    hardware = dict(
        gpu_name=gpu.name,
        gpu_total_memory_gb=gpu.total_memory / 2**30,
        gpu_compute_capability=f"{gpu.major}.{gpu.minor}",
        hostname=socket.gethostname(),
        platform=platform.platform(),
        torch_version=torch.__version__,
    )
    for record in pending:
        started = time.monotonic()
        key = f"{record['dataset']}__{record['stem']}"
        samples, elapsed = generate(
            model, tokenizer, record, oracle=False, full_context=args.full_context
        )
        oracle, oracle_elapsed = ([], None)
        if record["diagnostic"]:
            oracle, oracle_elapsed = generate(
                model, tokenizer, record, oracle=True, full_context=args.full_context
            )
        metrics = analyze_record(record, samples, oracle)
        sample_metrics = summarize_samples(record, samples[:100])
        uri = f"{args.out}/{checkpoint['label']}/raw/{key}.json.gz"
        with fsspec.open(uri, "wb") as handle:
            handle.write(
                gzip.compress(
                    json.dumps(dict(unconditioned=samples, oracle=oracle)).encode()
                )
            )
        total = time.monotonic() - started
        timings = []
        for mode, measured in (
            ("unconditioned", elapsed),
            ("oracle_half", oracle_elapsed),
        ):
            if measured is None:
                continue
            timings.append(
                dict(
                    dataset=record["dataset"],
                    stem=record["stem"],
                    n_residues=record["L"],
                    n_pairs=max(record["L"] - 6, 0) * max(record["L"] - 5, 0) // 2,
                    mode=mode,
                    **measured,
                    model_load_seconds=load_seconds,
                    model_stage_seconds=stage_seconds,
                    total_seconds=total + load_seconds + stage_seconds,
                    model_nickname=checkpoint["run_name"],
                    checkpoint_step=checkpoint["step"],
                    runner_tag="iris-coreweave",
                    **hardware,
                    timestamp_utc=datetime.now(UTC).isoformat(),
                    shard=args.shard,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=-1,
                    full_context_budget=args.full_context,
                )
            )
        result = dict(
            dataset=record["dataset"],
            stem=record["stem"],
            L=record["L"],
            diagnostic=record["diagnostic"],
            model=checkpoint["label"],
            metrics=metrics,
            single_sample=sample_metrics,
            timings=timings,
            raw_uri=uri,
        )
        write_json(f"{args.out}/{checkpoint['label']}/complete/{key}.json", result)
        print(
            json.dumps(
                {
                    "event": "protein_complete",
                    "key": key,
                    "seconds": total,
                    "unfinished": elapsed["unfinished_rollouts"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
