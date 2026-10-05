"""Measure the production source stream and verify replay on real co-located data."""

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fsspec
from transformers import AutoTokenizer

from common import ROOT
from corpus_catalog import DEFAULT_OUT
from corpus_stream import CorpusStream


def check_rank(args: tuple[str, str, int, int, int]) -> dict:
    prefix, tokenizer_path, rank, world, examples = args
    with fsspec.open(prefix + "/manifest.json") as handle:
        manifest = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    stream = CorpusStream(prefix, manifest, tokenizer, rank, world)
    began = time.perf_counter()
    token_counts = dict.fromkeys(["contacts_v1", "prompted"], 0)
    for _ in range(examples):
        row = stream.next()
        for name in token_counts:
            token_counts[name] += len(row[name])
    elapsed = time.perf_counter() - began
    cursor = copy.deepcopy(stream.state)
    expected = [stream.next_raw() for _ in range(16)]
    stream.close()
    resumed = CorpusStream(prefix, manifest, tokenizer, rank, world)
    resumed.state = cursor
    if [resumed.next_raw() for _ in range(16)] != expected:
        raise ValueError(f"Rank {rank} did not reproduce its next sixteen source rows")
    resumed.close()
    result = {
        "rank": rank,
        "examples": examples,
        "elapsed_seconds": elapsed,
        "documents_per_second": examples / elapsed,
        "tokens": token_counts,
        "counts": cursor["counts"],
        "resume_verified": True,
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_OUT)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--output", default=ROOT + "/corpus-validation/v1.json")
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local = Path("/tmp/exp347-corpus-tokenizer")
    local.mkdir(exist_ok=True)
    fs, source = fsspec.core.url_to_fs(ROOT + "/base/4B")
    for name in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
        fs.get_file(source + "/" + name, str(local / name))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                check_rank,
                [
                    (args.data, str(local), rank, args.world, args.examples)
                    for rank in range(args.world)
                ],
            )
        )
    counts = {
        source: {
            name: sum(r["counts"][source][name] for r in results)
            for name in results[0]["counts"][source]
        }
        for source in results[0]["counts"]
    }
    if any(c["accepted"] == 0 for c in counts.values()):
        raise ValueError(f"No admitted examples from one source: {counts}")
    output = {
        "data": args.data,
        "ranks": results,
        "counts": counts,
        "note": "Engineering sample; admission rates are not whole-corpus estimates",
    }
    with fsspec.open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
