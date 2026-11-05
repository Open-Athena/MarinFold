"""Plan and perform the approved, resumable CoreWeave-to-east5 input mirror.

The worker must run in the destination region. It reserves each complete source
read against a durable transfer budget before opening the object, streams without
local staging, and verifies SHA256 by rereading the destination in-region. A
receipt is published only after verification. Restart skips verified receipts;
incomplete attempts consume their reservation, bounding repeated cross-region I/O.
One Iris root owns each transfer directory; never launch overlapping writers.
"""

import argparse
import hashlib
import json
import os
import threading
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import s3fs

from freeze_tpu_parents import source_filesystem

DESTINATION = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp347_qwen_base_contacts"
)
SOURCE_DATA = "s3://marin-us-east-02a/MarinFold/exp347_qwen_base_contacts/data"
CEILING = 590_000_000_000
CHUNK = 16 * 1024**2


class SingleAttemptS3(s3fs.S3FileSystem):
    """Disable s3fs range-read retries so the transfer ledger bounds payload I/O."""

    retries = 1


def encoded(value: Any) -> bytes:
    """Return deterministic JSON for manifest identity and atomic object writes."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def make_plan(preflight: dict, snapshots: list[dict], fs: s3fs.S3FileSystem) -> dict:
    """Inventory immutable source objects under the explicitly approved route."""
    approval = preflight["approval"]
    if approval["maximum_cross_region_bytes"] != CEILING:
        raise ValueError("Expected the approved 590 GB transfer ceiling")
    if preflight["destination_root"] != DESTINATION:
        raise ValueError("Transfer approval does not cover another destination")
    catalog_bytes = Path("data/corpus235m/catalog.json").read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != preflight["source_catalog_sha256"]:
        raise ValueError("Local catalog differs from the audited source")
    catalog = json.loads(catalog_bytes)
    objects = []

    def append(source: str, target: str, size: int, etag: str, kind: str) -> None:
        objects.append(
            {
                "source": source,
                "target": target,
                "size": size,
                "etag": etag,
                "kind": kind,
            }
        )

    for name, component in catalog.items():
        for shard in component["shards"]:
            if not shard["uri"].startswith(component["source"] + "/"):
                raise ValueError("Catalog shard leaves its declared source")
            relative = shard["uri"].removeprefix(component["source"] + "/")
            append(
                shard["uri"],
                f"{DESTINATION}/data/corpus235m-v1/{name}/{relative}",
                shard["size"],
                shard["etag"],
                "corpus",
            )
    for name in ["catalog.json", "manifest.json"]:
        source = SOURCE_DATA + "/corpus235m-v1/" + name
        info = fs.info(source)
        append(
            source,
            f"{DESTINATION}/data/corpus235m-v1/{name}",
            info["size"],
            info["ETag"],
            "metadata",
        )
    validation = SOURCE_DATA + "/v1/validation"
    for path, info in sorted(fs.find(validation, detail=True).items()):
        relative = path.removeprefix(validation.removeprefix("s3://") + "/")
        append(
            "s3://" + path,
            f"{DESTINATION}/data/validation/{relative}",
            info["size"],
            info["ETag"],
            "validation",
        )
    if {s["format"] for s in snapshots} != {"contacts_v1", "prompted"} or len(
        snapshots
    ) != 2:
        raise ValueError("Both independently frozen parents are required")
    for snapshot in snapshots:
        marker = json.loads(fs.cat_file(snapshot["uri"] + "/snapshot.json"))
        if marker != snapshot:
            raise ValueError("Frozen parent differs from its committed snapshot")
        target = f"{DESTINATION}/checkpoints/{snapshot['fork_run']}/source/step-{snapshot['parent']['step']}"
        for name, item in snapshot["objects"].items():
            append(
                item["uri"],
                target + "/" + name,
                item["size"],
                item["etag"],
                "checkpoint",
            )
        info = fs.info(snapshot["uri"] + "/snapshot.json")
        append(
            snapshot["uri"] + "/snapshot.json",
            target + "/snapshot.json",
            info["size"],
            info["ETag"],
            "metadata",
        )
    if len({x["target"] for x in objects}) != len(objects):
        raise ValueError("Transfer destinations collide")
    total = sum(x["size"] for x in objects)
    if total > CEILING:
        raise ValueError("Plan exceeds the approved transfer ceiling")
    priority = {"checkpoint": 0, "metadata": 1, "validation": 2, "corpus": 3}
    objects.sort(key=lambda item: (priority[item["kind"]], item["target"]))
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "approval": approval,
        "destination": DESTINATION,
        "catalog_sha256": preflight["source_catalog_sha256"],
        "objects": objects,
        "parents": snapshots,
        "source_bytes": total,
        "ceiling_bytes": CEILING,
    }


class Transfer:
    """One transfer writer with durable budget reservations and verified receipts."""

    def __init__(
        self, source_fs: Any, destination_fs: Any, plan: dict, ledger: str
    ) -> None:
        self.source_fs = source_fs
        self.destination_fs = destination_fs
        self.plan = plan
        self.ledger = ledger.rstrip("/")
        self.lock = threading.Lock()
        self.identity = hashlib.sha256(encoded(plan)).hexdigest()
        budget_path = self.ledger + "/budget.json"
        if destination_fs.exists(budget_path):
            self.budget = json.loads(destination_fs.cat_file(budget_path))
            if self.budget["plan_sha256"] != self.identity:
                raise ValueError("Existing transfer ledger belongs to another plan")
        else:
            self.budget = {
                "plan_sha256": self.identity,
                "reserved_source_bytes": 0,
                "ceiling_bytes": plan["ceiling_bytes"],
            }
            destination_fs.pipe_file(budget_path, encoded(self.budget))

    def reserve(self, size: int) -> None:
        """Durably charge a whole object before any payload is read across regions."""
        with self.lock:
            total = self.budget["reserved_source_bytes"] + size
            if total > self.plan["ceiling_bytes"]:
                raise ValueError(
                    "Transfer budget exhausted; do not start another source read"
                )
            updated = {**self.budget, "reserved_source_bytes": total}
            self.destination_fs.pipe_file(
                self.ledger + "/budget.json", encoded(updated)
            )
            self.budget = updated

    def copy_one(self, item: dict) -> dict:
        """Resume a verified object or copy once and verify the regional destination."""
        identity = hashlib.sha256(item["source"].encode()).hexdigest()
        receipt_path = f"{self.ledger}/objects/{identity}.json"
        dest = self.destination_fs
        if dest.exists(receipt_path):
            receipt = json.loads(dest.cat_file(receipt_path))
            if receipt["source"] != item or receipt["plan_sha256"] != self.identity:
                raise ValueError("Receipt belongs to a different transfer object")
            current = dest.info(item["target"])
            if any(current.get(k) != v for k, v in receipt["metadata"].items()):
                raise ValueError(f"Verified regional object changed: {item['target']}")
            return receipt
        info = self.source_fs.info(item["source"])
        if info["size"] != item["size"] or info.get("ETag") != item["etag"]:
            raise ValueError(f"Source object changed since inventory: {item['source']}")
        self.reserve(item["size"])
        digest = hashlib.sha256()
        count = 0
        with (
            self.source_fs.open(
                item["source"], "rb", block_size=CHUNK, cache_type="none"
            ) as src,
            dest.open(item["target"], "wb", block_size=CHUNK) as out,
        ):
            while data := src.read(min(CHUNK, item["size"] - count)):
                digest.update(data)
                count += len(data)
                out.write(data)
        if count != item["size"]:
            raise ValueError(f"Incomplete source read: {item['source']}")
        self.source_fs.invalidate_cache(item["source"])
        if self.source_fs.info(item["source"])["ETag"] != item["etag"]:
            raise ValueError(f"Source changed during transfer: {item['source']}")
        local_digest = hashlib.sha256()
        with dest.open(
            item["target"], "rb", block_size=CHUNK, cache_type="none"
        ) as handle:
            while data := handle.read(CHUNK):
                local_digest.update(data)
        if digest.digest() != local_digest.digest():
            raise ValueError(f"Regional copy checksum mismatch: {item['target']}")
        dest.invalidate_cache(item["target"])
        info = dest.info(item["target"])
        if info["size"] != item["size"] or not info.get("generation"):
            raise ValueError("Destination lacks the expected size/generation")
        receipt = {
            "plan_sha256": self.identity,
            "source": item,
            "uri": item["target"],
            "source_etag": item["etag"],
            "metadata": {"size": info["size"], "generation": info["generation"]},
            "sha256": digest.hexdigest(),
        }
        dest.pipe_file(receipt_path, encoded(receipt))
        return receipt

    def run(self, workers: int) -> None:
        """Copy a bounded number of objects concurrently, then commit the mirror."""
        receipts = {}
        items = iter(self.plan["objects"])
        pool = ThreadPoolExecutor(max_workers=workers)
        pending = set()
        try:
            for _ in range(workers):
                if (item := next(items, None)) is not None:
                    pending.add(pool.submit(self.copy_one, item))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    receipt = future.result()
                    receipts[receipt["source"]["source"]] = receipt
                    if len(receipts) % 25 == 0 or len(receipts) == len(
                        self.plan["objects"]
                    ):
                        print(
                            json.dumps(
                                {
                                    "completed_objects": len(receipts),
                                    "total_objects": len(self.plan["objects"]),
                                    **self.budget,
                                }
                            ),
                            flush=True,
                        )
                    if (item := next(items, None)) is not None:
                        pending.add(pool.submit(self.copy_one, item))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        mirror = {
            "root": self.plan["destination"] + "/data/corpus235m-v1",
            "catalog_sha256": self.plan["catalog_sha256"],
            "objects": {
                uri: value
                for uri, value in receipts.items()
                if value["source"]["kind"] == "corpus"
            },
        }
        self.destination_fs.pipe_file(mirror["root"] + "/mirror.json", encoded(mirror))
        self.destination_fs.pipe_file(
            self.ledger + "/complete.json",
            encoded(
                {
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "plan_sha256": self.identity,
                    "objects": len(receipts),
                    "verified_bytes": sum(
                        x["source"]["size"] for x in receipts.values()
                    ),
                    **self.budget,
                }
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument(
        "--preflight", type=Path, default=Path("data/tpu_forks/preflight.json")
    )
    plan_parser.add_argument("--snapshots", nargs=2, type=Path, required=True)
    plan_parser.add_argument("--out", type=Path, required=True)
    copy_parser = sub.add_parser("copy")
    copy_parser.add_argument("--plan", type=Path, required=True)
    copy_parser.add_argument("--ledger", required=True)
    copy_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "plan":
        if args.out.exists():
            raise ValueError("A transfer plan is immutable; reuse it for recovery")
        plan = make_plan(
            json.loads(args.preflight.read_text()),
            [json.loads(p.read_text()) for p in args.snapshots],
            source_filesystem(),
        )
        args.out.write_bytes(encoded(plan))
        print(
            json.dumps(
                {
                    "objects": len(plan["objects"]),
                    "source_bytes": plan["source_bytes"],
                    "ceiling_bytes": plan["ceiling_bytes"],
                }
            )
        )
        return
    plan = json.loads(args.plan.read_text())
    if plan["destination"] != DESTINATION or plan["ceiling_bytes"] != CEILING:
        raise ValueError("Plan leaves the approved route or budget")
    if not args.ledger.startswith(DESTINATION + "/transfers/"):
        raise ValueError("Transfer ledger must be in the destination region")
    if not 1 <= args.workers <= 8:
        raise ValueError("The bounded staging worker supports one to eight streams")
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        zone = response.read().decode().split("/")[-1]
    if zone not in {"us-east5-a", "us-east5-b", "us-east5-c"}:
        raise ValueError(f"Transfer worker is outside the approved region: {zone}")
    source = SingleAttemptS3(
        key=os.environ["CW_KEY_ID"],
        secret=os.environ["CW_KEY_SECRET"],
        client_kwargs={"endpoint_url": "https://cwobject.com"},
        config_kwargs={
            "s3": {"addressing_style": "virtual"},
            "connect_timeout": 10,
            "read_timeout": 90,
            "retries": {"total_max_attempts": 1},
        },
    )
    destination, _ = fsspec.core.url_to_fs(DESTINATION)
    print(
        json.dumps(
            {
                "zone": zone,
                "source_bytes": plan["source_bytes"],
                "ceiling_bytes": CEILING,
            }
        ),
        flush=True,
    )
    Transfer(source, destination, plan, args.ledger).run(args.workers)


if __name__ == "__main__":
    main()
