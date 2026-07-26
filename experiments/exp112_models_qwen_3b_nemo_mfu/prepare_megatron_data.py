# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the contacts-v1 parquet corpus → Megatron `.bin/.idx` for the exp112
full NeMo training run (issue #112).

Megatron's pretraining dataloader can't read parquet; it needs a mmap
`IndexedDataset` (`.bin`+`.idx`). This one-off runs **on the workstation** (uses
this env's boto3/pyarrow to reach the CoreWeave bucket via the external endpoint
`https://cwobject.com`) and shells out to the **NeMo container** for the actual
`preprocess_data.py` tokenization (which has Megatron + the HF tokenizer). Output
is uploaded to `s3://…/exp112_qwen_3b_nemo_mfu/tokenized_megatron/`, from which the
training pods download it (fast, in-cluster LOTA).

Per split: stream parquet `document` column → one JSONL (download+extract+delete
per shard to bound disk) → `preprocess_data.py --json-keys document --append-eod
--tokenizer-type HuggingFaceTokenizer` → upload the `.bin/.idx`.

Docs already carry a semantic `<end>` token; `--append-eod` appends the
tokenizer's `<eos>` (id 1) as the *packing boundary* that Megatron's
`reset_attention_mask`/`reset_position_ids` reset on (block cross-doc attention).

Usage::

    uv run python prepare_megatron_data.py --splits val train        # val first (tiny, fast sanity)
    uv run python prepare_megatron_data.py --splits val --limit-shards 2   # quick e2e test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore.config import Config

BUCKET = "marin-us-east-02a"
CORPUS_PREFIX = "MarinFold/data/document_structures/contacts_v1"
OUT_PREFIX = "MarinFold/exp112_qwen_3b_nemo_mfu/tokenized_megatron"
TOKENIZER = "timodonnell/contacts-v1-tokenizer"
NEMO_IMAGE = os.environ.get("EXP112_IMAGE", "nvcr.io/nvidia/nemo:25.04.02")
EXTERNAL_ENDPOINT = "https://cwobject.com"  # workstation → bucket (pods use http://cwlota.com)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=EXTERNAL_ENDPOINT,
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 10}),
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
    )


def list_shards(s3, split: str) -> list[str]:
    keys = []
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{CORPUS_PREFIX}/{split}/"):
        keys += [o["Key"] for o in pg.get("Contents", []) if o["Key"].endswith(".parquet")]
    return sorted(keys)


def build_jsonl(s3, split: str, workdir: Path, limit: int | None) -> tuple[Path, int]:
    """Stream shards → one JSONL (`{"document": ...}` per line). Download+delete
    each shard to keep peak disk ~= JSONL + one shard."""
    shards = list_shards(s3, split)
    if limit:
        shards = shards[:limit]
    jsonl = workdir / f"{split}_document.jsonl"
    tmp_shard = workdir / f"_{split}_shard.parquet"
    ndocs = 0
    with jsonl.open("w") as out:
        for i, key in enumerate(shards):
            s3.download_file(BUCKET, key, str(tmp_shard))
            col = pq.read_table(tmp_shard, columns=["document"]).column("document")
            for v in col:
                out.write(json.dumps({"document": v.as_py()}) + "\n")
                ndocs += 1
            tmp_shard.unlink(missing_ok=True)
            if (i + 1) % 100 == 0 or i + 1 == len(shards):
                print(f"  [{split}] {i + 1}/{len(shards)} shards, {ndocs:,} docs", flush=True)
    print(f"  [{split}] JSONL: {jsonl} ({ndocs:,} docs, {jsonl.stat().st_size / 1e9:.2f} GB)")
    return jsonl, ndocs


def run_preprocess(jsonl: Path, split: str, workdir: Path, workers: int) -> Path:
    """Tokenize JSONL → bin/idx via preprocess_data.py inside the NeMo container.
    Returns the produced path PREFIX (such that prefix+'.bin'/'.idx' exist)."""
    out_prefix = f"{split}_document"
    cmd = [
        "docker", "run", "--rm", "--network", "host", "--shm-size=2g",
        "-v", f"{workdir}:/work", "-w", "/work",
        "-e", "HF_HUB_DISABLE_TELEMETRY=1",
        NEMO_IMAGE,
        "python", "/opt/megatron-lm/tools/preprocess_data.py",
        "--input", f"/work/{jsonl.name}",
        "--json-keys", "document",
        "--append-eod",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", TOKENIZER,
        "--workers", str(workers),
        "--output-prefix", f"/work/{out_prefix}",
    ]
    print(f"  [{split}] preprocess_data ({workers} workers)…", flush=True)
    subprocess.run(cmd, check=True)
    bins = glob.glob(str(workdir / f"{out_prefix}*.bin"))
    if len(bins) != 1:
        raise RuntimeError(f"expected exactly one .bin for {split}, got {bins}")
    prefix = bins[0][: -len(".bin")]
    assert Path(prefix + ".idx").exists(), f"missing idx for {prefix}"
    print(f"  [{split}] produced {prefix}.{{bin,idx}} "
          f"({Path(prefix + '.bin').stat().st_size / 1e9:.2f} GB bin)")
    return Path(prefix)


def upload(s3, prefix: Path, split: str) -> str:
    """Upload <prefix>.bin/.idx → S3 as {split}_document.{bin,idx}; return the
    S3 path prefix the training data module points at (no extension)."""
    dst_stem = f"{OUT_PREFIX}/{split}_document"
    for ext in (".bin", ".idx"):
        key = dst_stem + ext
        print(f"  [{split}] upload → s3://{BUCKET}/{key} …", flush=True)
        s3.upload_file(str(prefix) + ext, BUCKET, key)
    return f"s3://{BUCKET}/{dst_stem}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--workdir", default=os.environ.get("EXP112_DATAPREP_DIR", "/home/bizon/exp112_dataprep_work"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=None, help="for a quick e2e test")
    ap.add_argument("--keep-jsonl", action="store_true")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()

    results = {}
    for split in args.splits:
        print(f"=== split: {split} ===", flush=True)
        jsonl, ndocs = build_jsonl(s3, split, workdir, args.limit_shards)
        prefix = run_preprocess(jsonl, split, workdir, args.workers)
        s3_prefix = upload(s3, prefix, split)
        results[split] = {"docs": ndocs, "s3_prefix": s3_prefix}
        if not args.keep_jsonl:
            jsonl.unlink(missing_ok=True)

    print("\n=== DONE ===")
    print(json.dumps(results, indent=2))
    print("\nTraining data paths (bootstrap downloads these, strips s3://…/tokenized_megatron/ → local):")
    for split, r in results.items():
        print(f"  {split}: {r['s3_prefix']}.{{bin,idx}}")


if __name__ == "__main__":
    main()
