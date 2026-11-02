"""Stage the approved, pinned CoreWeave inputs into us-east5 GCS.

Run on a CPU worker in us-east5 after human approval of tpu_transfer_plan.json.
The default mode only verifies source inventories. --copy streams each object
with bounded memory, verifies its MD5 against GCS, and publishes a completion
marker only after all three inventories match. No storage-transfer service or
local corpus download is used.
"""

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import fsspec

PLAN_PATH = Path(__file__).with_name("data") / "tpu_transfer_plan.json"
BLOCK_BYTES = 8 * 1024 * 1024


def inventory(fs: fsspec.AbstractFileSystem, uri: str) -> list[dict]:
    """Return a reproducible relative-path, size, and source-ETag inventory."""
    root = fs._strip_protocol(uri).rstrip("/") + "/"
    return [
        {
            "path": path.removeprefix(root),
            "size": item["size"],
            "etag": item.get("ETag"),
        }
        for path, item in sorted(fs.find(uri, detail=True).items())
        if item["type"] == "file"
    ]


def inventory_hash(objects: list[dict]) -> str:
    """Hash every relative key, size, and immutable source ETag."""
    return hashlib.sha256(
        json.dumps(objects, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def check_source(objects: list[dict], artifact: dict) -> None:
    """Refuse a changed source before transferring any bytes."""
    actual = (
        len(objects),
        sum(obj["size"] for obj in objects),
        inventory_hash(objects),
    )
    expected = (artifact["objects"], artifact["bytes"], artifact["inventory_sha256"])
    if actual != expected:
        raise ValueError(
            f"Source inventory changed for {artifact['name']}: {actual} != {expected}"
        )


def copy_object(
    source_fs: fsspec.AbstractFileSystem,
    destination_fs: fsspec.AbstractFileSystem,
    artifact: dict,
    obj: dict,
) -> int:
    """Copy one immutable object, or reuse a previously checksum-verified copy."""
    source = f"{artifact['source']}/{obj['path']}"
    destination = f"{artifact['destination']}/{obj['path']}"
    if destination_fs.exists(destination):
        info = destination_fs.info(destination)
        metadata = info.get("metadata", {})
        if (
            info["size"] == obj["size"]
            and metadata.get("source_etag") == obj["etag"]
            and metadata.get("source_uri") == source
            and metadata.get("verified_md5") == info.get("md5Hash")
            and info.get("md5Hash")
        ):
            return 0
    digest = hashlib.md5()
    size = 0
    with (
        source_fs.open(
            source, "rb", block_size=BLOCK_BYTES, cache_type="none"
        ) as reader,
        destination_fs.open(
            destination,
            "wb",
            block_size=BLOCK_BYTES,
            metadata={"source_etag": obj["etag"], "source_uri": source},
            consistency="md5",
        ) as writer,
    ):
        while chunk := reader.read(BLOCK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            writer.write(chunk)
    source_fs.invalidate_cache(source)
    destination_fs.invalidate_cache(destination)
    info = destination_fs.info(destination)
    md5 = base64.b64encode(digest.digest()).decode()
    if (
        size != obj["size"]
        or info["size"] != size
        or info["md5Hash"] != md5
        or source_fs.info(source)["ETag"] != obj["etag"]
    ):
        raise ValueError(f"Copy failed content verification: {source} -> {destination}")
    destination_fs.setxattrs(destination, verified_md5=md5)
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy", action="store_true", help="Execute the human-approved transfer"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    source_fs = fsspec.filesystem("s3", use_listings_cache=False)
    destination_fs = fsspec.filesystem("gcs", cache_timeout=0)
    inventories = [inventory(source_fs, a["source"]) for a in plan["artifacts"]]
    for artifact, objects in zip(plan["artifacts"], inventories, strict=True):
        check_source(objects, artifact)
        print(
            f"VERIFIED source {artifact['name']}: {len(objects)} objects, {artifact['bytes']} bytes",
            flush=True,
        )
    if not args.copy:
        print(
            f"Read-only plan: {plan['total_transfer_bytes']} bytes; no transfer executed."
        )
        return
    for artifact, objects in zip(plan["artifacts"], inventories, strict=True):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(copy_object, source_fs, destination_fs, artifact, obj)
                for obj in objects
            ]
            copied = sum(future.result() for future in futures)
        actual = {
            obj["path"]: obj["size"]
            for obj in inventory(destination_fs, artifact["destination"])
        }
        expected = {obj["path"]: obj["size"] for obj in objects}
        if actual != expected:
            raise ValueError(
                f"Incomplete destination inventory: {artifact['destination']}"
            )
        print(
            f"VERIFIED destination {artifact['name']}: copied {copied} bytes",
            flush=True,
        )
    prefix = plan["artifacts"][0]["destination"].split("/input-mirrors/", 1)[0]
    with destination_fs.open(f"{prefix}/input-mirrors/verified.json", "wt") as handle:
        json.dump(
            {
                "all_verified": True,
                "source_step": plan["source_step"],
                "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "verified_at_utc": datetime.now(UTC).isoformat(),
                "artifacts": plan["artifacts"],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
