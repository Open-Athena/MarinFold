"""Select candidate trajectories and emit bounded, weighted SFT parquet shards."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from common import files, identity, read_json, rows, seed_for, stage_tokenizer, write_json, write_rows
from corpus import LossProfile, build_example, select_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", choices=["best", "random"], default="random")
    parser.add_argument("--hypothesis-weight", type=float, default=0.1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--rehearsal-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.rehearsal_fraction <= 1:
        raise ValueError("rehearsal fraction must be between zero and one")
    tokenizer = AutoTokenizer.from_pretrained(stage_tokenizer(args.tokenizer, Path("/tmp/exp281-tokenizers")))
    profile = LossProfile(hypothesis=args.hypothesis_weight)
    output_paths = []
    totals = defaultdict(float)
    seen = set()
    generator = None
    for path in files(args.candidates):
        # Generation publishes this marker after both parquet and timing CSV.
        completion = read_json(path.removesuffix(".parquet") + ".json")
        if generator is None:
            generator = completion["config"]["model"]
        if completion["config"]["model"] != generator:
            raise ValueError("one corpus round must use one frozen generator")
        grouped = defaultdict(list)
        for candidate in rows(path):
            grouped[(candidate["target_id"], candidate["forced"], candidate["budget"])].append(candidate)
        examples = []
        for key, pool in grouped.items():
            if args.selection == "best" and any(c["bootstrap"] for c in pool):
                raise ValueError("bootstrap references are not model predictions; rejection needs synthesis samples")
            if key in seen:
                raise ValueError(f"candidate group split across parts or duplicated: {key}")
            seen.add(key)
            selected = select_candidate(pool, args.selection, seed_for(args.seed, *key))
            plain = args.plain or random.Random(seed_for(args.seed, *key, "rehearsal")).random() < args.rehearsal_fraction
            example = build_example(
                header=selected["header"], history=[] if plain else selected["history"],
                reference=selected["reference"], forced=selected["forced"], profile=profile,
                tokenizer=tokenizer, context=args.context, plain=plain,
            )
            example.update(target_id=selected["target_id"], candidate_id=selected["candidate_id"],
                           generator=selected["generator"], budget=selected["budget"],
                           selection=args.selection, selected_f1=selected["score"]["f1"])
            examples.append(example)
            for name in ("hypothesis_weight", "final_weight", "marker_weight"):
                totals[name] += example[name]
            totals["documents"] += 1
            totals["plain_documents"] += plain
            totals["processed_tokens"] += len(example["input_ids"])
            totals["invalid_candidates"] += sum(not c["valid"] for c in pool)
            totals["candidates"] += len(pool)
        out = f"{args.output}/part-{len(output_paths):05d}.parquet"
        write_rows(out, examples)
        output_paths.append(out)
    config = vars(args)
    write_json(f"{args.output}/manifest.json", {"config": config, "config_hash": identity(config),
               "tokenizer_hash": identity(tokenizer.get_vocab()), "totals": dict(totals), "shards": output_paths})


if __name__ == "__main__":
    main()
