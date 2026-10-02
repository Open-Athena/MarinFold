"""Build paired complete-document token caches from co-located AFDB parquets."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from common import MAX_LENGTH, MODELS, ROOT, SOURCE, convert_document, split_key
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@cache
def tokenizer() -> PreTrainedTokenizerBase:
    """Load the shared pinned tokenizer once per process."""
    repo, revision = MODELS["0.8B"]
    return AutoTokenizer.from_pretrained(repo, revision=revision)


def encode(prefix: str, completion: str) -> tuple[list[int], int]:
    """Tokenize an intact string, marking tokens that overlap the completion."""
    result = tokenizer()(
        prefix + completion, add_special_tokens=False, return_offsets_mapping=True
    )
    first = next(
        i for i, (_, end) in enumerate(result["offset_mapping"]) if end > len(prefix)
    )
    return result["input_ids"] + [tokenizer().eos_token_id], first


def prepare_shard(args: tuple[str, str]) -> dict:
    """Validate and tokenize one source shard; failures abort the build."""
    uri, out = args
    name = uri.rsplit("/", 1)[-1]
    records = {"train": [], "validation": []}
    counts = {
        "source_rows": 0,
        "too_long": 0,
        "empty_contacts": 0,
        "train": 0,
        "validation": 0,
        "contacts_v1_tokens": 0,
        "prompted_tokens": 0,
        "contacts": 0,
    }
    source_hash = hashlib.sha256()
    with fsspec.open(uri, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(
            batch_size=128, columns=["document", "entry_id", "seq_cluster_id"]
        ):
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                source_hash.update(row["document"].encode())
                doc = convert_document(row["document"])
                if not doc.contacts:
                    counts["empty_contacts"] += 1
                    continue
                raw, raw_start = encode(doc.raw_prefix, doc.raw_completion)
                prompted, prompted_start = encode(
                    doc.prompted_prefix, doc.prompted_completion
                )
                if max(len(raw), len(prompted)) > MAX_LENGTH:
                    counts["too_long"] += 1
                    continue
                key = split_key(doc.sequence, row["seq_cluster_id"])
                split = "validation" if int(key[:8], 16) % 100 == 0 else "train"
                records[split].append(
                    {
                        "entry_id": row["entry_id"],
                        "split_key": key,
                        "sequence": doc.sequence,
                        "n_contacts": len(doc.contacts),
                        "contacts": [list(pair) for pair in doc.contacts],
                        "contacts_v1": raw,
                        "contacts_v1_start": raw_start,
                        "prompted": prompted,
                        "prompted_start": prompted_start,
                    }
                )
                counts[split] += 1
                if split == "train":
                    counts["contacts_v1_tokens"] += len(raw)
                    counts["prompted_tokens"] += len(prompted)
                    counts["contacts"] += len(doc.contacts)
    for split, rows in records.items():
        if rows:
            with fsspec.open(f"{out}/{split}/{name}", "wb") as handle:
                pq.write_table(pa.Table.from_pylist(rows), handle, compression="zstd")
    result = {
        "source": uri,
        "source_document_sha256": source_hash.hexdigest(),
        **counts,
    }
    with fsspec.open(f"{out}/counts/{name}.json", "w") as handle:
        json.dump(result, handle)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=f"{ROOT}/data/v1")
    parser.add_argument("--shards", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fs, path = fsspec.core.url_to_fs(SOURCE)
    files = sorted(
        fs.glob(path + "/*.parquet"),
        key=lambda p: hashlib.sha256(p.encode()).hexdigest(),
    )[: args.shards]
    if len(files) != args.shards:
        raise ValueError(f"Expected {args.shards} input shards, found {len(files)}")
    # Populate the small tokenizer cache before workers start; no model weights here.
    tokenizer()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(prepare_shard, [(f"s3://{p}", args.out) for p in files])
        )
    totals = {
        k: sum(r[k] for r in results)
        for k in results[0]
        if k not in ("source", "source_document_sha256")
    }
    manifest = {
        "source": SOURCE,
        "max_length": MAX_LENGTH,
        "tokenizer": MODELS["0.8B"],
        "selection": f"{args.shards} SHA256-sorted shard paths; sequence-cluster-hash 1% validation",
        "totals": totals,
        "shards": results,
    }
    with fsspec.open(args.out + "/manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
