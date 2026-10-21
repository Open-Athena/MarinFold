"""Save complete individual maps under the established rollout sampling recipe.

The launcher supplies the unchanged exp277 reference worker for storage and
hardware helpers. GPU imports are intentionally confined to this GPU entrypoint.
Every protein commits raw maps, timing and a completion marker; retries resume
only after validating the marker's run identity.
"""

import argparse
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)
from marinfold.inference._tokenizer import model_source_path
from reference_worker import (
    BEGIN,
    CONTACT_RE,
    NUM_POSITIONS,
    candidate_pair_count,
    gpu_metadata,
    stage_model,
    write_json,
)
from transformers import AutoTokenizer
from validation import validate_raw
from vllm import LLM, SamplingParams

RAW_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("rollout", pa.int32()), ("pool", pa.string()), ("sampling_seed", pa.int64()),
    ("finish_reason", pa.string()), ("generated_tokens", pa.int32()),
    ("contacts", pa.list_(pa.list_(pa.int16(), 2))),
    ("n_contacts", pa.int32()), ("cumulative_logprob", pa.float64()),
    ("mean_logprob", pa.float64()), ("prompt_tokens", pa.int32()), ("max_tokens", pa.int32()),
])


def main() -> None:
    """Run one shard of 200-rollout per-protein samples and save complete maps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-first", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    plan = json.loads((here / "plan.json").read_text())
    worker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reference_sha = hashlib.sha256((here / "reference_worker.py").read_bytes()).hexdigest()
    assert reference_sha == plan["reference_worker_sha256"]
    assert hashlib.sha256((here / "targets.json").read_bytes()).hexdigest() == plan["targets_sha256"]
    records = json.loads((here / "targets.json").read_text())
    records = [r for r in records if r["stem"] == "8arl_A"] if args.smoke else [r for r in records if r["ordinal"] % args.num_shards == args.shard]
    if args.smoke_first:
        assert args.num_shards == 1
        records.sort(key=lambda r: (r["stem"] != "8arl_A", r["ordinal"]))
    output_uri = plan["output"] + ("/smoke" if args.smoke else "/production")
    fs, root = fsspec.core.url_to_fs(output_uri)
    pending = []
    for record in records:
        marker = f"{root}/complete/{record['stem']}.json"
        if fs.exists(marker):
            old = json.loads(fs.cat_file(marker))
            assert old["worker_sha256"] == worker_sha and old["n_rollouts"] == plan["n_rollouts"]
            assert fs.exists(f"{root}/rollouts/{record['stem']}.parquet")
            assert fs.exists(f"{root}/timings/{record['stem']}.json")
        else:
            pending.append(record)
    if not pending:
        print("All assigned proteins already complete", flush=True)
        return
    model_dir, stage_seconds, _ = stage_model(plan["source"]["uri"], Path("/tmp/exp345-model"), plan)
    for entry in plan["files"]:
        path = model_dir / entry["name"]
        assert path.stat().st_size == entry["size"]
        if entry["digest_kind"] == "sha256":
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["digest"]
    effective = Path(model_source_path(model_dir))
    config = json.loads((effective / "config.json").read_text())
    assert config["rope_theta"] == 500000 and config["rope_scaling"]["rope_type"] == "llama3"
    tokenizer = AutoTokenizer.from_pretrained(str(effective))
    assert len(tokenizer) == config["vocab_size"] == 2845
    end = tokenizer.convert_tokens_to_ids("<end>")
    assert end is not None and end >= 0
    started = time.monotonic()
    model = LLM(model=str(effective), dtype="bfloat16", max_model_len=8192,
                gpu_memory_utilization=.9, enable_prefix_caching=False,
                generation_config="vllm", max_num_seqs=256, seed=plan["seed"])
    load_seconds = time.monotonic()-started
    hardware = gpu_metadata()
    for record in pending:
        started = time.monotonic()
        residues = residues_from_sequence(record["input_seq"])
        prompts, maps, seeds = [], [], []
        for rollout in range(plan["n_rollouts"]):
            document = build_document(f"{record['stem']}:r{rollout}", residues, [], config=GenerationConfig())
            prompts.append(document.document[:document.document.index(BEGIN)+len(BEGIN)])
            maps.append({(document.n_term_index+i) % NUM_POSITIONS: i for i in range(document.seq_len)})
            seeds.append(plan["seed"]*1_000_003 + record["ordinal"]*1000 + rollout)
        prompt_tokens = len(tokenizer(prompts[0], add_special_tokens=False).input_ids)
        max_tokens = min(8192-prompt_tokens, 6*record["L"]+128)
        parameters = [SamplingParams(temperature=1., top_p=.95, top_k=-1, max_tokens=max_tokens,
                         stop_token_ids=[end], skip_special_tokens=False, seed=seed, logprobs=0) for seed in seeds]
        inference_start = time.monotonic()
        outputs = model.generate(prompts, parameters, use_tqdm=False)
        elapsed = time.monotonic()-inference_start
        rows = []
        for rollout, (output, mapping, seed) in enumerate(zip(outputs, maps, seeds, strict=True)):
            completion = output.outputs[0]
            seen, contacts = set(), []
            for first, second in CONTACT_RE.findall(completion.text):
                i, j = mapping.get(int(first)), mapping.get(int(second))
                if i is None or j is None or abs(i-j) < 6:
                    continue
                pair = (min(i, j), max(i, j))
                if pair not in seen:
                    seen.add(pair)
                    contacts.append(list(pair))
            assert completion.cumulative_logprob is not None
            rows.append({"dataset": record["dataset"], "stem": record["stem"], "L": record["L"],
                         "rollout": rollout, "pool": "A" if rollout < 100 else "B", "sampling_seed": seed,
                         "finish_reason": completion.finish_reason, "generated_tokens": len(completion.token_ids),
                         "contacts": contacts, "n_contacts": len(contacts),
                         "cumulative_logprob": completion.cumulative_logprob,
                         "mean_logprob": completion.cumulative_logprob/max(1, len(completion.token_ids)),
                         "prompt_tokens": prompt_tokens, "max_tokens": max_tokens})
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA), buffer, compression="zstd")
        raw = buffer.getvalue()
        fs.pipe_file(f"{root}/rollouts/{record['stem']}.parquet", raw)
        unfinished = sum(row["finish_reason"] != "stop" for row in rows)
        timing = {"stem": record["stem"], "dataset": record["dataset"], "n_residues": record["L"],
                  "n_pairs": candidate_pair_count(record["L"]), "mode": "whole_map_rollout_resample",
                  "elapsed_seconds": elapsed, "model_load_seconds": load_seconds,
                  "model_stage_seconds": stage_seconds,
                  "total_seconds": time.monotonic()-started + load_seconds + stage_seconds,
                  "model_nickname": plan["checkpoint"]["run_name"] + "-step-479417",
                  "runner_tag": "iris-coreweave", **hardware, "timestamp_utc": datetime.now(UTC).isoformat(),
                  "n_rollouts": len(rows), "unfinished_rollouts": unfinished, "shard": args.shard,
                  "batch_size": 200, "generated_tokens": sum(row["generated_tokens"] for row in rows)}
        write_json(timing, f"{output_uri}/timings/{record['stem']}.json")
        marker = {"stem": record["stem"], "n_rollouts": len(rows), "unfinished_rollouts": unfinished,
                  "worker_sha256": worker_sha, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                  "targets_sha256": plan["targets_sha256"], "checkpoint": plan["checkpoint"]}
        print(f"[exp345] {record['stem']} L={record['L']} rollouts={len(rows)} unfinished={unfinished} seconds={elapsed:.1f}", flush=True)
        if unfinished:
            write_json(marker, f"{output_uri}/failures/{record['stem']}.json")
            raise RuntimeError(f"{record['stem']}: {unfinished} capped rollouts saved for inspection")
        local_raw = Path("/tmp/exp345-latest.parquet")
        local_raw.write_bytes(raw)
        validate_raw(local_raw, record, marker)
        write_json(marker, f"{output_uri}/complete/{record['stem']}.json")


if __name__ == "__main__":
    main()
