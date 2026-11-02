"""Publish frozen rescore inputs and small results, then verify anonymous reads."""

import json
import subprocess
from pathlib import Path

from huggingface_hub import HfFileSystem
from prepare_inputs import CACHE, DATA, PUBLIC, digest

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Upload the reproducibility bundle through hf and verify every byte."""
    files = [(CACHE / "inputs.json.gz", "inputs.json.gz")]
    files.extend(
        (p, str(p.relative_to(HERE)))
        for root in (DATA, HERE / "plots")
        for p in sorted(root.iterdir())
        if p.is_file() and p.name != "publication_manifest.json"
    )
    fs = HfFileSystem(token=False)
    manifest = []
    for path, relative in files:
        destination = PUBLIC + "/" + relative
        subprocess.run(["hf", "buckets", "cp", str(path), destination], check=True)
        expected = digest(path.read_bytes())
        fs.invalidate_cache()
        actual = digest(fs.read_bytes(destination.removeprefix("hf://")))
        if actual != expected:
            raise ValueError(f"Published bytes differ: {destination}")
        manifest.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": expected}
        )
    (DATA / "publication_manifest.json").write_text(
        json.dumps({"prefix": PUBLIC, "files": manifest}, indent=2) + "\n"
    )
    print(f"Verified {len(manifest)} public artifacts anonymously.")


if __name__ == "__main__":
    main()
