"""Freeze full GPU training states with same-region, conditional S3 copies.

The GPU trainer prunes its previous checkpoint after committing a successor.
Each copy therefore pins the source ETag, including every multipart range.
Only snapshot.json commits a usable snapshot; incomplete copies cannot be used
as TPU parents. No model bytes pass through the launching workstation.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import s3fs

ROOT = "s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts"


def source_filesystem() -> s3fs.S3FileSystem:
    """Use the explicit CoreWeave credential without changing global profiles."""
    return s3fs.S3FileSystem(
        key=os.environ["CW_KEY_ID"],
        secret=os.environ["CW_KEY_SECRET"],
        client_kwargs={"endpoint_url": "https://cwobject.com"},
        config_kwargs={
            "s3": {"addressing_style": "virtual"},
            "connect_timeout": 10,
            "read_timeout": 90,
            "retries": {"max_attempts": 2},
        },
    )


def copy_object(fs: s3fs.S3FileSystem, source: str, target: str, info: dict) -> dict:
    """Copy the audited object entirely within RNO, pinning every source read."""
    source_bucket, source_key = source.removeprefix("s3://").split("/", 1)
    target_bucket, target_key = target.removeprefix("s3://").split("/", 1)
    if source_bucket != "marin-rno2a" or target_bucket != source_bucket:
        raise ValueError("Parent freezing must stay within the RNO bucket")
    destination = {"Bucket": target_bucket, "Key": target_key}
    origin = {
        "CopySource": {"Bucket": source_bucket, "Key": source_key},
        "CopySourceIfMatch": info["ETag"],
    }
    size = info["size"]
    if size <= 5_000_000_000:
        fs.call_s3("copy_object", **destination, **origin)
    else:
        upload = fs.call_s3("create_multipart_upload", **destination)
        identity = {**destination, "UploadId": upload["UploadId"]}
        try:
            parts = []
            block = 1024**3
            for number, start in enumerate(range(0, size, block), start=1):
                result = fs.call_s3(
                    "upload_part_copy",
                    **identity,
                    **origin,
                    PartNumber=number,
                    CopySourceRange=f"bytes={start}-{min(start + block, size) - 1}",
                )
                parts.append(
                    {"PartNumber": number, "ETag": result["CopyPartResult"]["ETag"]}
                )
            fs.call_s3(
                "complete_multipart_upload",
                **identity,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            fs.call_s3("abort_multipart_upload", **identity)
            raise
    fs.invalidate_cache(target)
    copied = fs.info(target)
    if copied["size"] != size:
        raise ValueError(f"Snapshot size differs from its parent: {target}")
    return {
        "uri": target,
        "size": size,
        "etag": copied["ETag"],
        "source_etag": info["ETag"],
    }


def freeze(fs: s3fs.S3FileSystem, document_format: str) -> dict:
    """Commit one independently named snapshot of the current complete parent."""
    run = f"exp347-qwen35-4b-{document_format}-corpus235m-100bt"
    fork = f"exp347-qwen35-4b-{document_format}-corpus235m-tpu-east5"
    parent = json.loads(fs.cat_file(f"{ROOT}/checkpoints/{run}/resume.json"))
    source = parent["uri"]
    if not source.startswith(f"{ROOT}/checkpoints/{run}/step-"):
        raise ValueError("Parent pointer leaves the source run")
    if parent != json.loads(fs.cat_file(source + "/complete.json")):
        raise ValueError("Parent completion and resume markers differ")
    target = f"{ROOT}/checkpoints/{fork}/source/step-{parent['step']}"
    if fs.exists(target + "/snapshot.json"):
        raise ValueError(
            "A committed snapshot already exists; use its existing manifest"
        )
    objects = fs.find(source, detail=True)
    relative = {
        name.removeprefix(source.removeprefix("s3://") + "/"): info
        for name, info in objects.items()
    }
    required = {
        "complete.json",
        "model/config.json",
        "model/tokenizer.json",
        "model/tokenizer_config.json",
        "model/model.safetensors.index.json",
    }
    required.update(f"resume/rank-{rank}.pt" for rank in range(8))
    if not required.issubset(relative) or parent["world"] != 8:
        raise ValueError("Parent lacks complete eight-rank recovery state")
    index = json.loads(fs.cat_file(source + "/model/model.safetensors.index.json"))
    if not {"model/" + name for name in index["weight_map"].values()}.issubset(
        relative
    ):
        raise ValueError("Parent is missing indexed model weights")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            name: pool.submit(
                copy_object, fs, source + "/" + name, target + "/" + name, info
            )
            for name, info in relative.items()
        }
        copied = {name: future.result() for name, future in futures.items()}
    snapshot = {
        "created_utc": datetime.now(UTC).isoformat(),
        "format": document_format,
        "parent_run": run,
        "fork_run": fork,
        "parent": parent,
        "uri": target,
        "objects": copied,
        "bytes": sum(item["size"] for item in copied.values()),
        "copy_method": "Same-region S3 CopyObject/UploadPartCopy, source ETag preconditions",
    }
    fs.pipe_file(
        target + "/snapshot.json", (json.dumps(snapshot, indent=2) + "\n").encode()
    )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["contacts_v1", "prompted"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to replace an existing parent selection")
    snapshot = freeze(source_filesystem(), args.format)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(
        json.dumps(
            {key: snapshot[key] for key in ["format", "uri", "parent", "bytes"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
