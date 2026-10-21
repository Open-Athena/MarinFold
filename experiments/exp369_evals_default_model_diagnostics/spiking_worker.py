"""Generate resumable default-model oracle-contact interventions on one GPU."""

import argparse
import gzip
import hashlib
import io
import json
import platform
import random
import re
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np

from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence
from marinfold.document_structures.contacts_v1.read import live_contacts
from protocol import stable_seed

STATEMENT = re.compile(r"<(contact|retract)>\s+<p(\d+)>\s+<p(\d+)>")


def make_prompt(target: dict, supplied: list[list[int]], repeat: int, rollout: int) -> tuple[str, dict[int, int]]:
    """Share sequence realization across doses while resampling every rollout."""
    key = f"exp369:{target['stem']}:{repeat}:{rollout}"
    document = build_document(key, residues_from_sequence(target["sequence"]), [], config=GenerationConfig())
    if document is None:
        raise ValueError(f"Document rejected for {target['stem']}")
    prefix = document.document.split("<begin_statements>")[0] + "<begin_statements>"
    positions = [(document.n_term_index + index) % 2000 for index in range(target["L"])]
    inverse = {position: index for index, position in enumerate(positions)}
    if len(inverse) != target["L"]:
        raise ValueError("Position ring collision")
    rng = random.Random(key + ":context")
    pairs = [list(pair) for pair in supplied]
    rng.shuffle(pairs)
    statements = []
    for i, j in pairs:
        if rng.random() < .5:
            i, j = j, i
        statements.append(f"<contact> <p{positions[i]}> <p{positions[j]}>")
    prompt = " ".join([prefix, *statements])
    restored = {tuple(sorted((inverse[i], inverse[j]))) for i, j in live_contacts(prompt)}
    if restored != {tuple(pair) for pair in supplied}:
        raise ValueError("Seed contacts failed the position-mapping round trip")
    return prompt, inverse


def parsed_contacts(text: str, inverse: dict[int, int]) -> tuple[list[list[int]], dict]:
    """Parse final live contacts and preserve malformed/out-of-frame diagnostics."""
    statements = STATEMENT.findall(text)
    diagnostics = {"malformed": text.count("<contact>") + text.count("<retract>") - len(statements),
                   "out_of_range": 0, "too_close": 0}
    pairs = set()
    for i, j in live_contacts(text):
        if i not in inverse or j not in inverse:
            diagnostics["out_of_range"] += 1
            continue
        pair = tuple(sorted((inverse[i], inverse[j])))
        if pair[1] - pair[0] < 6:
            diagnostics["too_close"] += 1
            continue
        pairs.add(pair)
    return [list(pair) for pair in sorted(pairs)], diagnostics


def write_bytes(uri: str, payload: bytes) -> None:
    """Persist output using the execution host's filesystem configuration."""
    fs, path = fsspec.core.url_to_fs(uri)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "wb") as stream:
        stream.write(payload)


def main() -> None:
    """Run complete 100-rollout conditions with provenance and completion gates."""
    # GPU imports are optional for CPU-only prompt/metric validation.
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--stem")
    parser.add_argument("--counts", help="Restrict the smoke run to these integer doses")
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    plan_sha = hashlib.sha256(raw).hexdigest()
    expected = {"eval-val": 97, "afdb-training-disagreements": 6}
    if plan["eval_set"] not in expected or len(plan["targets"]) != expected[plan["eval_set"]]:
        raise ValueError("Unexpected frozen cohort or target count")
    shard, n_shards = map(int, args.shard.split("/"))
    if not 0 <= shard < n_shards:
        raise ValueError("Invalid shard")
    selected = [target for index, target in enumerate(sorted(plan["targets"], key=lambda row: (row["L"], row["stem"])))
                if index % n_shards == shard and (args.stem is None or target["stem"] == args.stem)]
    allowed = set(map(int, args.counts.split(","))) if args.counts else None
    conditions = []
    for target in selected:
        for repeat in range(plan["repeats"]):
            counts = sorted({dose["k"] for dose in target["doses_by_repeat"][str(repeat)]})
            for count in counts:
                if allowed is not None and count not in allowed:
                    continue
                for policy in (["random"] if count == 0 else ["random", "sse_balanced"]):
                    conditions.append((target, repeat, count, policy))
    if not conditions:
        raise ValueError("No conditions selected")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=.90, max_num_seqs=128, enable_prefix_caching=True,
              generation_config="vllm", seed=369)
    load_seconds = time.perf_counter() - started
    device = torch.cuda.get_device_properties(0)
    metadata = {"model_nickname": plan["model"], "step": plan["step"], "runner_tag": "iris" if args.out.startswith("s3:") else "local",
                "gpu_name": device.name, "gpu_total_memory_gb": device.total_memory / 2**30,
                "gpu_compute_capability": f"{device.major}.{device.minor}", "hostname": socket.gethostname(),
                "platform": platform.platform(), "torch_version": torch.__version__, "vllm_version": vllm.__version__,
                "transformers_version": transformers.__version__, "plan_sha256": plan_sha,
                "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_load_seconds": load_seconds, "model_load_accounting": "once_per_worker_not_additive_across_conditions"}
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    fs, _ = fsspec.core.url_to_fs(args.out)
    for target, repeat, count, policy in conditions:
        total_started = time.perf_counter()
        mode = f"{policy}-r{repeat}-k{count:04d}"
        base = f"{args.out}/units/{target['stem']}/{mode}"
        marker = base + ".complete.json"
        if fs.exists(marker):
            with fs.open(marker) as stream:
                previous = json.load(stream)
            if previous["plan_sha256"] != plan_sha or previous["worker_sha256"] != metadata["worker_sha256"]:
                raise ValueError("Resume provenance mismatch")
            for extension, digest in previous["files_sha256"].items():
                with fs.open(base + extension, "rb") as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise ValueError("Resume artifact checksum mismatch")
            continue
        supplied = target["orders"][f"{policy}:{repeat}"][:count]
        prompts, maps, sampling = [], [], []
        for rollout in range(plan["n_rollouts"]):
            prompt, inverse = make_prompt(target, supplied, repeat, rollout)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            budget = min(6 * target["L"] + 128, 8192 - prompt_tokens)
            if budget < 1:
                raise ValueError("No generation context remaining")
            prompts.append(prompt)
            maps.append(inverse)
            sampling.append(SamplingParams(temperature=plan["temperature"], top_p=plan["top_p"], top_k=-1,
                                           max_tokens=budget, stop_token_ids=[end_id], skip_special_tokens=False,
                                           seed=stable_seed(f"exp369:{target['stem']}:{repeat}:{rollout}:sample")))
        inference_started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - inference_started
        if len(outputs) != plan["n_rollouts"]:
            raise ValueError("Missing completions")
        rows = []
        votes = np.zeros((target["L"], target["L"]), dtype=np.uint16)
        for index, (output, inverse, prompt, params) in enumerate(zip(outputs, maps, prompts, sampling, strict=True)):
            completion = output.outputs[0]
            pairs, diagnostics = parsed_contacts(prompt + " " + completion.text, inverse)
            for i, j in pairs:
                votes[i, j] += 1
                votes[j, i] += 1
            rows.append({"rollout": index, "contacts": pairs, "text": completion.text,
                         "finished": completion.finish_reason == "stop", "finish_reason": completion.finish_reason,
                         "n_tokens": len(completion.token_ids), "max_new_tokens": params.max_tokens,
                         "prompt_tokens": len(output.prompt_token_ids), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         **diagnostics})
        blob = gzip.compress((json.dumps({"stem": target["stem"], "mode": mode, "supplied": supplied, "rows": rows}) + "\n").encode())
        buffer = io.BytesIO()
        np.savez_compressed(buffer, score=votes)
        payloads = {".json.gz": blob, ".npz": buffer.getvalue()}
        timing = {**metadata, "stem": target["stem"], "n_residues": target["L"], "n_pairs": len(target["truth"]),
                  "mode": mode, "policy": policy, "repeat": repeat, "n_supplied": count,
                  "supplied_fraction": count / len(target["truth"]), "elapsed_seconds": elapsed,
                  "n_rollouts": len(rows), "timestamp_utc": datetime.now(UTC).isoformat(),
                  "n_unfinished": sum(not row["finished"] for row in rows),
                  "n_malformed": sum(row["malformed"] for row in rows),
                  "n_out_of_range": sum(row["out_of_range"] for row in rows)}
        for extension, payload in payloads.items():
            write_bytes(base + extension, payload)
        timing["total_seconds"] = time.perf_counter() - total_started
        write_bytes(base + ".timing.json", (json.dumps(timing) + "\n").encode())
        if timing["n_unfinished"] or timing["n_malformed"] or timing["n_out_of_range"]:
            raise ValueError(f"Completion validation failed; raw artifacts preserved: {base}")
        write_bytes(marker, (json.dumps(timing | {"files_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()}}) + "\n").encode())
        print(json.dumps({"stem": target["stem"], "mode": mode, "elapsed_seconds": elapsed, "complete": True}), flush=True)


if __name__ == "__main__":
    main()
