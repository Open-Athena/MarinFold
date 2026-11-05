"""Exercise restart, corruption and budget boundaries without cloud transfers."""

import hashlib
import io
import json
import uuid

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from stage_tpu_inputs import Transfer


class VersionedMemory(MemoryFileSystem):
    """Expose object identities with the same contract as S3/GCS metadata."""

    def info(self, path: str, **kwargs) -> dict:
        result = super().info(path, **kwargs)
        if result["type"] == "file":
            digest = hashlib.sha256(self.cat_file(path)).hexdigest()
            result.update(ETag=digest, generation=digest)
        return result


class CorruptingMemory(VersionedMemory):
    """Simulate corruption in the destination readback, after a successful PUT."""

    def open(self, path: str, mode: str = "rb", **kwargs):
        if mode == "rb" and path.endswith("/object"):
            return io.BytesIO(b"corrupted")
        return super().open(path, mode, **kwargs)


def setup_transfer(
    ceiling: int = 100,
) -> tuple[VersionedMemory, VersionedMemory, dict, str]:
    root = "/" + uuid.uuid4().hex
    source = VersionedMemory(skip_instance_cache=True)
    target = VersionedMemory(skip_instance_cache=True)
    source.pipe_file(root + "/source", b"protein contacts")
    info = source.info(root + "/source")
    plan = {
        "destination": root + "/destination",
        "catalog_sha256": "f" * 64,
        "ceiling_bytes": ceiling,
        "objects": [
            {
                "source": root + "/source",
                "target": root + "/destination/object",
                "size": info["size"],
                "etag": info["ETag"],
                "kind": "corpus",
            }
        ],
    }
    return source, target, plan, root + "/ledger"


def test_resume_uses_verified_copy_after_source_removal() -> None:
    source, target, plan, ledger = setup_transfer()
    first = Transfer(source, target, plan, ledger)
    first.run(2)
    source.rm(plan["objects"][0]["source"])
    resumed = Transfer(source, target, plan, ledger)
    resumed.run(2)
    assert resumed.budget["reserved_source_bytes"] == len(b"protein contacts")
    mirror = json.loads(
        target.cat_file(plan["destination"] + "/data/corpus235m-v1/mirror.json")
    )
    assert (
        mirror["objects"][plan["objects"][0]["source"]]["sha256"]
        == hashlib.sha256(b"protein contacts").hexdigest()
    )


def test_budget_reservation_precedes_payload_copy() -> None:
    source, target, plan, ledger = setup_transfer(ceiling=3)
    transfer = Transfer(source, target, plan, ledger)
    with pytest.raises(ValueError, match="budget exhausted"):
        transfer.run(1)
    assert not target.exists(plan["objects"][0]["target"])
    assert (
        json.loads(target.cat_file(ledger + "/budget.json"))["reserved_source_bytes"]
        == 0
    )


def test_failed_verification_consumes_budget_and_cannot_publish_mirror() -> None:
    source, _, plan, ledger = setup_transfer(ceiling=len(b"protein contacts"))
    target = CorruptingMemory(skip_instance_cache=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        Transfer(source, target, plan, ledger).run(1)
    assert not target.exists(ledger + "/complete.json")
    with pytest.raises(ValueError, match="budget exhausted"):
        Transfer(source, VersionedMemory(skip_instance_cache=True), plan, ledger).run(1)


def test_modified_regional_copy_is_not_silently_retransferred() -> None:
    source, target, plan, ledger = setup_transfer()
    Transfer(source, target, plan, ledger).run(1)
    target.pipe_file(plan["objects"][0]["target"], b"corrupt")
    with pytest.raises(ValueError, match="regional object changed"):
        Transfer(source, target, plan, ledger).run(1)


def test_changed_source_is_rejected_before_reserving_bytes() -> None:
    source, target, plan, ledger = setup_transfer()
    source.pipe_file(plan["objects"][0]["source"], b"different source")
    transfer = Transfer(source, target, plan, ledger)
    with pytest.raises(ValueError, match="Source object changed"):
        transfer.run(1)
    assert transfer.budget["reserved_source_bytes"] == 0


def test_plan_identity_prevents_reusing_a_budget_for_another_transfer() -> None:
    source, target, plan, ledger = setup_transfer()
    Transfer(source, target, plan, ledger)
    with pytest.raises(ValueError, match="another plan"):
        Transfer(source, target, {**plan, "ceiling_bytes": 1000}, ledger)
