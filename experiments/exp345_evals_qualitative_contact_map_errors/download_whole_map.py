"""Download the complete rollout archive anonymously and verify every map."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

from fetch_whole_map import validate_raw

HERE = Path(__file__).resolve().parent
PUBLIC = "buckets/open-athena/MarinFold/data/exp345-qualitative-contact-map-errors/v2"


def main() -> None:
    """Fetch only the public data archive; no model or private credentials needed."""
    fs = HfFileSystem(token=False)
    manifest = json.loads((HERE / "data/whole_map_raw_manifest.json").read_text())
    cache = HERE / ".cache/whole_maps/production"
    cache.mkdir(parents=True, exist_ok=True)
    records = {r["stem"]: r for r in json.loads((HERE / "data/whole_map_targets.json").read_text())}
    missing = [m for m in manifest["markers"] if not (cache / f"{m['stem']}.parquet").exists()]
    if missing:
        archive = fs.cat_file(PUBLIC + "/whole_map_raw.tar.gz")
        expected = json.loads((HERE / "data/whole_map_archive.json").read_text())
        assert hashlib.sha256(archive).hexdigest() == expected["sha256"]
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            names = {f"{stem}.parquet" for stem in records}
            assert set(tar.getnames()) == names
            for member in tar.getmembers():
                assert member.isfile() and member.name in names
                source = tar.extractfile(member)
                assert source is not None
                (cache / member.name).write_bytes(source.read())
    for marker in manifest["markers"]:
        validate_raw(cache / f"{marker['stem']}.parquet", records[marker["stem"]], marker)
    print(f"Validated {len(manifest['markers'])} proteins / {manifest['n_maps']} complete rollouts")


if __name__ == "__main__":
    main()
