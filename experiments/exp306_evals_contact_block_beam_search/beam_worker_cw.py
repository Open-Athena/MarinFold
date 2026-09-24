#!/usr/bin/env python
"""Generate contact-block beam rollouts on one CoreWeave H100 shard."""

import argparse
import hashlib
import json
import platform
import random
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from beam_policy import pair_candidates, sample_candidate

BEGIN = "<begin_statements>"
NUM_POS = 2000
MIN_SEP = 6
CONTEXT = 8192


@dataclass
class Rollout:
    """Mutable state of one document realization during blockwise decoding."""

    index: int
    prompt: str
    prompt_ids: list[int]
    position_to_index: dict[int, int]
    max_new: int
    rng: random.Random
    generated: list[int] = field(default_factory=list)
    contacts: set[tuple[int, int]] = field(default_factory=set)
    beam_choices: list[dict] = field(default_factory=list)
    finished: bool = False
    done: bool = False
    malformed: int = 0
    rejected_beams: int = 0

    @property
    def tokens(self) -> list[int]:
        """Full token prefix for the next model call."""
        return self.prompt_ids + self.generated


def stable_seed(value: str) -> int:
    """Derive a reproducible 31-bit seed independent of Python hash state."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def make_rollout(stem: str, sequence: str, index: int, tokenizer: Any,
                 residues_from_sequence: Any, build_document: Any,
                 generation_config: Any) -> Rollout:
    """Build a fresh exp82-style document prompt without reference contacts."""
    residues = residues_from_sequence(sequence)
    document = build_document(f"{stem}:r{index}", residues, [], config=generation_config())
    prefix = document.document[:document.document.index(BEGIN) + len(BEGIN)]
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    max_new = min(CONTEXT - len(prompt_ids), 6 * len(sequence) + 128)
    if max_new < 1:
        raise ValueError(f"{stem}: no context left for generation")
    position_to_index = {}
    for residue in range(len(sequence)):
        position = (document.n_term_index + residue) % NUM_POS
        token_id = tokenizer.convert_tokens_to_ids(f"<p{position}>")
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"{stem}: missing position token <p{position}>")
        position_to_index[int(token_id)] = residue
    if len(position_to_index) != len(sequence):
        raise ValueError(f"{stem}: repeated position token in sequence")
    return Rollout(
        index=index, prompt=prefix, prompt_ids=list(prompt_ids),
        position_to_index=position_to_index, max_new=max_new,
        rng=random.Random(stable_seed(f"{stem}:r{index}:beam-choice")),
    )


def decode(llm: Any, states: list[Rollout], stem: str, beam_width: int,
           contact_id: int, end_id: int, max_statements: int | None = None) -> None:
    """Sample statement starts, then search and sample each complete position pair."""
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import BeamSearchParams

    beam_params = BeamSearchParams(beam_width=beam_width, max_tokens=2,
                                   ignore_eos=True, temperature=0.0)
    statement = 0
    while True:
        active = [state for state in states if not state.done]
        if not active:
            return
        if max_statements is not None and statement >= max_statements:
            for state in active:
                state.done = True
            return
        for state in active:
            if len(state.generated) >= state.max_new:
                state.done = True
        active = [state for state in active if not state.done]
        if not active:
            return
        prompts = [TokensPrompt(prompt_token_ids=state.tokens) for state in active]
        params = [SamplingParams(
            temperature=1.0, top_p=0.95, top_k=-1, max_tokens=1,
            seed=stable_seed(f"{stem}:r{state.index}:statement:{statement}:start"),
            skip_special_tokens=False,
        ) for state in active]
        starts = llm.generate(prompts, params, use_tqdm=False)
        contacts = []
        for state, output in zip(active, starts):
            ids = output.outputs[0].token_ids
            if len(ids) != 1:
                state.done = True
                state.malformed += 1
                continue
            token = int(ids[0])
            state.generated.append(token)
            if token == end_id:
                state.finished = True
                state.done = True
            elif token == contact_id:
                if len(state.generated) + 2 > state.max_new:
                    state.done = True
                else:
                    contacts.append(state)
            else:
                state.done = True
                state.malformed += 1
        if contacts:
            pair_prompts = [TokensPrompt(prompt_token_ids=state.tokens) for state in contacts]
            beams = llm.beam_search(pair_prompts, beam_params, use_tqdm=False)
            for state, result in zip(contacts, beams):
                candidates = pair_candidates(result.sequences, len(state.tokens),
                                             state.position_to_index, MIN_SEP)
                state.rejected_beams += len(result.sequences) - len(candidates)
                if not candidates:
                    state.done = True
                    state.malformed += 1
                    continue
                selected = sample_candidate(candidates, state.rng)
                state.generated.extend(selected.token_ids)
                state.contacts.add(selected.pair)
                state.beam_choices.append({
                    "rank": selected.rank,
                    "logprob": selected.logprob,
                    "top_logprob": max(item.logprob for item in candidates),
                    "n_valid": len(candidates),
                })
        statement += 1


def write_parquet(uri: str, rows: list[dict]) -> None:
    """Write a compressed result table through the co-located fsspec S3 path."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    with fsspec.open(uri, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle, compression="zstd")


def main() -> None:
    """Load one checkpoint and score an interleaved shard of sequence-only targets."""
    import fsspec
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", required=True, help="i/n")
    parser.add_argument("--cohort", choices=["eval-val", "foldswitch", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--beam-width", type=int, required=True)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-statements", type=int)
    args = parser.parse_args()
    if args.beam_width < 2:
        raise ValueError("beam width must be at least two")
    shard, n_shards = (int(value) for value in args.shard.split("/"))
    with fsspec.open(args.targets, "rt") as handle:
        targets = pd.read_csv(handle).to_dict("records")
    if args.cohort != "both":
        targets = [row for row in targets if row["cohort"] == args.cohort]
    if args.split != "all":
        targets = [row for row in targets if row["split"] == args.split]
    targets.sort(key=lambda row: (int(row["L"]), row["target_id"]))
    mine = [row for index, row in enumerate(targets) if index % n_shards == shard]
    if args.limit is not None:
        mine = mine[:args.limit]
    if not mine:
        print(f"[exp306] no targets for shard {shard}/{n_shards}", flush=True)
        return
    print(f"[exp306] shard {shard}/{n_shards}: {len(mine)} {args.cohort} targets", flush=True)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    contact_id = tokenizer.convert_tokens_to_ids("<contact>")
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    if any(item is None or item == tokenizer.unk_token_id for item in (contact_id, end_id)):
        raise ValueError("checkpoint tokenizer lacks <contact> or <end>")
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=CONTEXT,
              gpu_memory_utilization=0.85, enable_prefix_caching=True,
              generation_config="vllm", max_num_seqs=256, seed=306)
    model_load = time.perf_counter() - load_start
    gpu = torch.cuda.get_device_properties(0)
    mode = f"beam{args.beam_width}"
    meta = {
        "model_nickname": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "runner_tag": "iris-cw-rno2a", "gpu_name": gpu.name,
        "gpu_total_memory_gb": gpu.total_memory / 1e9,
        "gpu_compute_capability": f"{gpu.major}.{gpu.minor}",
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    for position, target in enumerate(mine, 1):
        stem, sequence, cohort = target["stem"], target["sequence"], target["cohort"]
        base = f"{args.out.rstrip('/')}/{mode}/{cohort}/{stem}"
        raw_uri, timing_uri = f"{base}.parquet", f"{base}.timing.parquet"
        filesystem, raw_path = fsspec.core.url_to_fs(raw_uri)
        if filesystem.exists(raw_path) and filesystem.exists(fsspec.core.url_to_fs(timing_uri)[1]):
            print(f"[exp306] skip {stem}", flush=True)
            continue
        states = [make_rollout(stem, sequence, index, tokenizer, residues_from_sequence,
                               build_document, GenerationConfig)
                  for index in range(args.n_rollouts)]
        start = time.perf_counter()
        decode(llm, states, stem, args.beam_width, int(contact_id), int(end_id),
               args.max_statements)
        elapsed = time.perf_counter() - start
        rows = [{
            "cohort": cohort, "dataset": target["dataset"], "stem": stem,
            "rollout": state.index, "mode": mode, "L": len(sequence),
            "prompt": state.prompt,
            "prompt_sha256": hashlib.sha256(state.prompt.encode()).hexdigest(),
            "contacts": [list(pair) for pair in sorted(state.contacts)],
            "n_contacts": len(state.contacts), "n_statements": len(state.beam_choices),
            "n_tokens": len(state.generated), "max_new": state.max_new,
            "finished": state.finished, "malformed": state.malformed,
            "rejected_beams": state.rejected_beams,
            "beam_choices_json": json.dumps(state.beam_choices, separators=(",", ":")),
        } for state in states]
        write_parquet(raw_uri, rows)
        written = time.perf_counter()
        timing = [{
            "stem": stem, "n_residues": len(sequence),
            "n_pairs": len(sequence) * (len(sequence) - 1) // 2,
            "mode": mode, "elapsed_seconds": elapsed,
            "model_load_seconds": model_load / len(mine),
            "total_seconds": written - start + model_load / len(mine),
            "n_rollouts": args.n_rollouts,
            "n_finished": sum(state.finished for state in states),
            "n_malformed": sum(state.malformed for state in states),
            "n_rejected_beams": sum(state.rejected_beams for state in states),
            "generated_tokens": sum(len(state.generated) for state in states),
            "beam_width": args.beam_width,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), **meta,
        }]
        write_parquet(timing_uri, timing)
        print(f"[exp306] {position}/{len(mine)} {stem} L={len(sequence)} "
              f"finished={timing[0]['n_finished']}/{len(states)} "
              f"time={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
