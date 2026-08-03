# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score contacts-v1 validation documents with vLLM prompt logprobs.

This worker emits one row per validation document/protein with the numerator and
 denominator needed for document-level bootstrap of token-weighted eval loss.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("source_shard", pa.string()),
        ("row_in_shard", pa.int64()),
        ("token_count", pa.int64()),
        ("loss_sum", pa.float64()),
        ("mean_loss", pa.float64()),
    ]
)


def read_parquet(uri: str, **kwargs: Any) -> pa.Table:
    """Read a parquet file through fsspec so gs:// works in Iris pods."""
    import fsspec

    with fsspec.open(uri, "rb") as handle:
        return pq.read_table(handle, **kwargs)


def write_parquet(table: pa.Table, uri: str) -> None:
    """Write a parquet file through fsspec so gs:// works in Iris pods."""
    import fsspec

    with fsspec.open(uri, "wb") as handle:
        pq.write_table(table, handle, compression="zstd")


def stage_model(src: str, dst: Path) -> Path:
    """Copy a remote HF model directory to local disk for vLLM."""
    if "://" not in src:
        return Path(src)

    import fsspec

    dst.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(src)
    files = [entry for entry in fs.ls(root, detail=True) if entry["type"] == "file"]
    if not files:
        raise FileNotFoundError(f"No files under model path {src}")

    start = time.time()
    for entry in files:
        fs.get_file(entry["name"], str(dst / os.path.basename(entry["name"])))
    size = sum(entry["size"] for entry in files)
    print(
        f"[score-loss] staged model {src} -> {dst} "
        f"({len(files)} files, {size / 2**30:.2f} GiB, {time.time() - start:.0f}s)",
        flush=True,
    )
    return dst


def list_input_shards(input_glob: str) -> list[str]:
    """Expand an fsspec glob into sorted shard URIs."""
    import fsspec

    fs, _ = fsspec.core.url_to_fs(input_glob)
    matches = sorted(fs.glob(input_glob))
    return [fs.unstrip_protocol(match) for match in matches]


def tokenize_document(tokenizer: Any, text: str) -> list[int]:
    """Match Levanter TextLmDatasetFormat tokenization: text + EOS, then BOS."""
    eos = tokenizer.eos_token
    if eos is not None:
        text = text + " " + eos
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        ids.insert(0, bos_id)
    return ids


def prompt_loss_sum(output: Any, prompt_token_ids: list[int]) -> float:
    """Sum negative prompt-token logprobs, excluding the first token."""
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise ValueError("vLLM did not return prompt_logprobs")
    if len(prompt_logprobs) != len(prompt_token_ids):
        raise ValueError(
            f"prompt_logprobs length {len(prompt_logprobs)} != prompt length {len(prompt_token_ids)}"
        )

    total = 0.0
    for token_id, logprob_map in zip(prompt_token_ids[1:], prompt_logprobs[1:], strict=True):
        if logprob_map is None:
            raise ValueError("Missing prompt logprob for a non-initial token")
        item = logprob_map.get(token_id)
        if item is None:
            raise KeyError(f"Actual prompt token id {token_id} missing from vLLM prompt_logprobs")
        total -= float(item.logprob)
    return total


def done_doc_ids(out_dir: str, shard_i: int, num_shards: int) -> tuple[set[int], int]:
    """Return written doc ids and existing part count for this shard."""
    import fsspec

    fs, _ = fsspec.core.url_to_fs(out_dir)
    pattern = f"{out_dir.rstrip('/')}/shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet"
    try:
        parts = sorted(fs.glob(pattern))
    except FileNotFoundError:
        return set(), 0

    seen: set[int] = set()
    for part in parts:
        uri = fs.unstrip_protocol(part)
        try:
            table = read_parquet(uri, columns=["doc_id"])
            seen.update(int(x) for x in table.column("doc_id").to_pylist())
        except Exception as exc:
            print(f"[score-loss] ignoring unreadable output part {uri}: {exc}", flush=True)
    return seen, len(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model directory, local or gs://")
    parser.add_argument("--input-glob", required=True, help="Raw contacts-v1 val parquet glob.")
    parser.add_argument("--out", required=True, help="Output directory for per-document parquet parts.")
    parser.add_argument("--text-column", default="document")
    parser.add_argument("--shard", required=True, help="Shard as i/n over validation documents.")
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-frac", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    args = parser.parse_args()

    shard_i, num_shards = (int(piece) for piece in args.shard.split("/"))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_dir = stage_model(args.model, Path("/tmp/marinfold_exp188_model"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    records: list[dict[str, Any]] = []
    doc_id = 0
    for input_uri in list_input_shards(args.input_glob):
        table = read_parquet(input_uri, columns=[args.text_column])
        texts = table.column(args.text_column).to_pylist()
        shard_name = os.path.basename(input_uri)
        for row_in_shard, text in enumerate(texts):
            if doc_id % num_shards == shard_i:
                records.append(
                    {
                        "doc_id": doc_id,
                        "source_shard": shard_name,
                        "row_in_shard": row_in_shard,
                        "input_ids": tokenize_document(tokenizer, text),
                    }
                )
            doc_id += 1

    done, n_existing_parts = done_doc_ids(args.out, shard_i, num_shards)
    records = [record for record in records if int(record["doc_id"]) not in done]
    if args.limit is not None:
        records = records[: args.limit]

    too_long = [record for record in records if len(record["input_ids"]) > args.max_model_len]
    if too_long:
        examples = ", ".join(str(record["doc_id"]) for record in too_long[:5])
        raise ValueError(
            f"{len(too_long)} documents exceed max_model_len={args.max_model_len}; examples: {examples}"
        )

    print(
        f"[score-loss] shard {shard_i}/{num_shards}: {len(records)} docs to score "
        f"({len(done)} already done), total val docs={doc_id}",
        flush=True,
    )
    if not records:
        return 0

    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_frac,
        enable_prefix_caching=False,
        generation_config="vllm",
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    start = time.time()
    part = n_existing_parts
    for offset in range(0, len(records), args.chunk):
        group = records[offset : offset + args.chunk]
        prompts = [{"prompt_token_ids": record["input_ids"]} for record in group]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)

        rows = {name: [] for name in SCHEMA.names}
        for record, output in zip(group, outputs, strict=True):
            token_count = len(record["input_ids"]) - 1
            loss_sum = prompt_loss_sum(output, record["input_ids"])
            rows["doc_id"].append(int(record["doc_id"]))
            rows["source_shard"].append(str(record["source_shard"]))
            rows["row_in_shard"].append(int(record["row_in_shard"]))
            rows["token_count"].append(token_count)
            rows["loss_sum"].append(loss_sum)
            rows["mean_loss"].append(loss_sum / token_count)

        dest = f"{args.out.rstrip('/')}/shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet"
        write_parquet(pa.table(rows, schema=SCHEMA), dest)
        part += 1
        n_tokens = sum(len(record["input_ids"]) for record in group)
        print(
            f"[score-loss] {offset + len(group)}/{len(records)} docs, "
            f"{n_tokens} prompt tokens -> {dest} (elapsed {(time.time() - start) / 60:.1f}m)",
            flush=True,
        )

    print(f"[score-loss] DONE shard {shard_i}/{num_shards} in {(time.time() - start) / 60:.1f}m", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
