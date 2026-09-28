# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Show that the complex corpus fits the 1.5B model's 2845-token vocabulary.

The corpus publishes a 2848-token tokenizer; the model has 2845 embedding rows.
Training with the model's tokenizer is only sound if the two agree on ids
0-2844 and no complex document reaches the three appended ids. This checks both
on a downloaded shard and writes `data/tokenizer_contract.csv`.

It is evidence, not the enforcement: `prepare.py` validates *every* record of
*every* shard against the same contract while the caches are built, so a
document this sample never saw still cannot reach training.

    uv run python verify_tokenizer_contract.py --shard /path/shard-00000-....parquet
"""

import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import download_bucket_files, hf_hub_download
from tokenizers import Tokenizer

from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_BUCKET,
    COMPLEX_BUCKET_PREFIX,
    TOKENIZER,
)

#: `prepare.py`'s contract, restated as data so the CSV can report each clause.
MODEL_VOCAB_SIZE = 2845
UNK_ID = 2844


def vocabularies(cache: Path) -> tuple[dict[str, int], dict[str, int]]:
    """The model's tokenizer vocabulary and the corpus's published one."""
    model = json.loads(
        Path(hf_hub_download(TOKENIZER, "tokenizer.json")).read_text()
    )["model"]["vocab"]
    published_path = cache / "published-tokenizer.json"
    if not published_path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        download_bucket_files(
            COMPLEX_BUCKET,
            files=[
                (
                    f"{COMPLEX_BUCKET_PREFIX}/tokenizer/tokenizer.json",
                    str(published_path),
                )
            ],
            token=False,
        )
    published = json.loads(published_path.read_text())["model"]["vocab"]
    return model, published


def compare_vocabularies(model: dict[str, int], published: dict[str, int]) -> dict:
    """Confirm the published tokenizer only *appends* to the model's."""
    shared = {token: identifier for token, identifier in published.items()
              if identifier < MODEL_VOCAB_SIZE}
    appended = sorted(
        (identifier, token)
        for token, identifier in published.items()
        if identifier >= MODEL_VOCAB_SIZE
    )
    if shared != model:
        differing = sorted(
            token for token in set(shared) | set(model)
            if shared.get(token) != model.get(token)
        )
        raise ValueError(
            f"The two tokenizers disagree below id {MODEL_VOCAB_SIZE}: {differing[:10]}"
        )
    return {
        "model_vocab_size": len(model),
        "published_vocab_size": len(published),
        "appended_tokens": [token for _, token in appended],
        "appended_ids": [identifier for identifier, _ in appended],
    }


def check_shard(shard: Path, rows: int | None) -> dict:
    """Re-encode documents with the model's tokenizer and audit every one."""
    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER, "tokenizer.json"))
    table = pq.ParquetFile(shard)
    checked = 0
    max_id = 0
    max_length = 0
    num_tokens_mismatches = 0
    boundary_failures = 0
    at_cap = 0
    truncated_documents = 0
    for batch in table.iter_batches(
        batch_size=512, columns=["document", "num_tokens", "truncated"]
    ):
        for document, num_tokens, truncated in zip(
            batch.column("document").to_pylist(),
            batch.column("num_tokens").to_pylist(),
            batch.column("truncated").to_pylist(),
            strict=True,
        ):
            ids = tokenizer.encode(document, add_special_tokens=False).ids
            # The published `num_tokens` counts the document alone; the training
            # pipeline appends `<eos>` on top of it.
            if len(ids) != num_tokens:
                num_tokens_mismatches += 1
            if ids[0] != 2 or 8 not in ids or 9 not in ids or ids[-1] != 10:
                boundary_failures += 1
            max_id = max(max_id, max(ids))
            max_length = max(max_length, len(ids))
            # #294 truncates whole statements, so a truncated document stops
            # within one statement of the cap rather than exactly on it. Only the
            # documents landing exactly on 8192 lose their appended `<eos>` to
            # the packer's clip; the rest are unaffected.
            at_cap += len(ids) == 8192
            truncated_documents += bool(truncated)
            if len(ids) > 8192:
                raise ValueError(
                    f"{shard.name}: a document is {len(ids)} tokens, past the "
                    f"8192 cap the corpus claims to enforce"
                )
            checked += 1
            if rows is not None and checked >= rows:
                break
        if rows is not None and checked >= rows:
            break
    return {
        "shard": shard.name,
        "documents_checked": checked,
        "max_token_id": max_id,
        "max_document_tokens": max_length,
        "documents_flagged_truncated": truncated_documents,
        "documents_at_8192_cap": at_cap,
        "num_tokens_mismatches": num_tokens_mismatches,
        "boundary_failures": boundary_failures,
        "in_model_vocabulary": max_id < UNK_ID,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, default=None, help="limit rows checked (default: all)"
    )
    parser.add_argument("--cache", type=Path, default=Path("/tmp/exp343-tokenizers"))
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer_contract.csv"))
    args = parser.parse_args()
    comparison = compare_vocabularies(*vocabularies(args.cache))
    result = check_shard(args.shard, args.rows)
    if not result["in_model_vocabulary"]:
        raise ValueError(
            f"{result['shard']} reaches token id {result['max_token_id']}, at or past "
            f"`<UNK>` ({UNK_ID}) -- the corpus is not inside the model's vocabulary"
        )
    if result["num_tokens_mismatches"] or result["boundary_failures"]:
        raise ValueError(f"Contract failures: {result}")
    row = {
        **result,
        "model_tokenizer": TOKENIZER,
        "model_vocab_size": comparison["model_vocab_size"],
        "published_vocab_size": comparison["published_vocab_size"],
        "appended_tokens": " ".join(comparison["appended_tokens"]),
        "appended_ids": " ".join(str(i) for i in comparison["appended_ids"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
