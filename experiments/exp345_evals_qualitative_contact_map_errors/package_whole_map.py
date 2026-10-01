"""Package validated rollout parquets in a deterministic public download."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from fetch_whole_map import validate_raw

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Validate each committed unit before producing the public raw archive."""
    cache = HERE / ".cache/whole_maps/production"
    records = {r["stem"]: r for r in json.loads((HERE / "data/whole_map_targets.json").read_text())}
    manifest = json.loads((HERE / "data/whole_map_raw_manifest.json").read_text())
    assert len(manifest["markers"]) == len(records) == 97
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for marker in sorted(manifest["markers"], key=lambda m: m["stem"]):
            path = cache / f"{marker['stem']}.parquet"
            validate_raw(path, records[marker["stem"]], marker)
            data = path.read_bytes()
            info = tarfile.TarInfo(path.name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    raw = gzip.compress(buffer.getvalue(), mtime=0)
    (HERE / "whole_map_raw.tar.gz").write_bytes(raw)
    (HERE / "data/whole_map_archive.json").write_text(json.dumps({"filename": "whole_map_raw.tar.gz", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "proteins": 97, "rollouts": 19400}, indent=2) + "\n")
    print(f"Packaged {len(raw):,} bytes")


if __name__ == "__main__":
    main()
