"""Run 100 resampled continuations for each frozen contact-seeded prompt.

The cached 248B-token checkpoint and tokenizer are SHA256 verified before loading.
Raw prompts, token-ring mappings, completions, sparse votes and direct timings
are saved per case. Model generation never receives evaluation contact labels.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from score_rollout_worker import gpu_metadata
from seed_completion import parse_pairs, render_seeds


def main() -> None:
    """Load once and complete seven fixed conditions for one protein."""
    # These libraries are only available in the pinned GPU inference image.
    from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence
    from marinfold.inference._tokenizer import model_source_path
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    args = parser.parse_args()
    path = Path("/root/inputs") / f"{args.stem}.json"
    inputs = json.loads(path.read_text())
    input_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    out = Path("/data/seed-completion-v1") / args.stem
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for case in inputs["cases"]:
        result_path = out / f"{case['arm']}-{case['replicate']}.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved["input_sha256"] != input_digest or len(saved["rollouts"]) != 100:
                raise ValueError("Resume inputs differ or result is incomplete")
        else:
            pending.append(case)
    if not pending:
        return
    manifest = json.loads(Path("/root/checkpoint_manifest.json").read_text())
    for name, record in manifest["files"].items():
        with (Path("/data/model") / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"Checkpoint file changed: {name}")
    started = time.monotonic()
    directory = model_source_path(Path("/data/model"))
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    config = json.loads((Path(directory) / "config.json").read_text())
    if config["rope_theta"] != 500_000 or config["rope_scaling"]["rope_type"] != "llama3" or len(tokenizer) != config["vocab_size"]:
        raise ValueError("Effective model config changed")
    end = tokenizer.convert_tokens_to_ids("<end>")
    if end is None or end < 0 or end == tokenizer.unk_token_id:
        raise ValueError("Invalid end token")
    model = LLM(model=str(directory), dtype="bfloat16", max_model_len=8192,
                gpu_memory_utilization=.9, enable_prefix_caching=False,
                generation_config="vllm", max_num_seqs=512, seed=0)
    load_seconds = time.monotonic() - started
    hardware = gpu_metadata()
    residues = residues_from_sequence(inputs["sequence"])
    for case in pending:
        total_started = time.monotonic()
        prompts, maps, parameters = [], [], []
        for rollout in range(100):
            document = build_document(f"{args.stem}:r{rollout}", residues, [], config=GenerationConfig())
            sequence_prefix = document.document.split("<begin_statements>")[0] + "<begin_statements>"
            prefix = render_seeds(case["seed_pairs"], document.n_term_index, rollout)
            position_map = {(document.n_term_index + i) % 2000: i for i in range(document.seq_len)}
            if parse_pairs(prefix, position_map) != {tuple(pair) for pair in case["seed_pairs"]}:
                raise ValueError("Seed prompt round-trip failed")
            prompts.append(sequence_prefix + prefix)
            maps.append(position_map)
            n_tokens = len(tokenizer(prompts[-1], add_special_tokens=False).input_ids)
            parameters.append(SamplingParams(temperature=1, top_p=.95, top_k=-1,
                max_tokens=min(8192 - n_tokens, 6 * inputs["L"] + 128),
                stop_token_ids=[end], skip_special_tokens=False, seed=rollout))
        inference_started = time.monotonic()
        outputs = model.generate(prompts, parameters, use_tqdm=False)
        elapsed = time.monotonic() - inference_started
        votes, archive = Counter(), []
        for index, (output, position_map) in enumerate(zip(outputs, maps, strict=True)):
            completion = output.outputs[0]
            pairs = parse_pairs(completion.text, position_map)
            if completion.finish_reason == "stop":
                votes.update(pairs)
            archive.append(dict(prompt=prompts[index], position_map=position_map, text=completion.text,
                finish_reason=completion.finish_reason, generated_tokens=len(completion.token_ids),
                sampling_seed=index, parsed_unique_contacts=len(pairs)))
        timing = dict(stem=args.stem, n_residues=inputs["L"], n_pairs=len(votes),
            mode=f"{case['arm']}-{case['replicate']}", n_seed=case["n_seed"],
            elapsed_seconds=elapsed, model_load_seconds=load_seconds,
            total_seconds=load_seconds + time.monotonic() - total_started,
            model_nickname="contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
            runner_tag="modal", timestamp_utc=datetime.now(UTC).isoformat(),
            n_rollouts=100, generated_tokens=sum(row["generated_tokens"] for row in archive),
            stopped_rollouts=sum(row["finish_reason"] == "stop" for row in archive), **hardware)
        result = dict(input_sha256=input_digest, case=case, timing=timing, rollouts=archive,
            votes=[[i, j, count] for (i, j), count in sorted(votes.items())])
        destination = out / f"{case['arm']}-{case['replicate']}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        temporary.replace(destination)
        print(f"{args.stem}/{timing['mode']}: {elapsed:.1f}s, {timing['stopped_rollouts']}/100 finished", flush=True)


if __name__ == "__main__":
    main()
