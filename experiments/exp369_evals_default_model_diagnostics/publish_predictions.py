"""Publish completed proteins incrementally, or restore their public predictions."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

from publish_to_hf import DESTINATION, build_archive

HERE = Path(__file__).resolve().parent
INDEX = HERE / "data/public_predictions.json"


def main() -> None:
    """Validate complete proteins before publishing; restore needs no authentication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    parser.add_argument("--run", type=Path, default=Path("/data/exp369/full-v2"))
    parser.add_argument("--archives", type=Path, default=Path("/data/exp369/public-predictions"))
    parser.add_argument("--limit", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--upload", action="store_true")
    modes.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.restore:
        index = json.loads(INDEX.read_text())
        fs = HfFileSystem(token=False)
        entries = list(index["proteins"].items())
        for stem, entry in entries[:args.limit]:
            with fs.open(entry["uri"].removeprefix("hf://"), "rb") as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"Public prediction checksum mismatch: {stem}")
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                members = [member for member in archive.getmembers() if member.name.startswith("units/")]
                archive.extractall(args.run, members=members, filter="data")
            print(f"Restored {stem}", flush=True)
        return
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {"plan_sha256": digest, "expected_proteins": len(plan["targets"]), "proteins": {}}
    if index["plan_sha256"] != digest:
        raise ValueError("Public prediction index belongs to a different frozen plan")
    published = 0
    for target in plan["targets"]:
        modes = [(repeat, count, policy) for repeat, grid in target["doses_by_repeat"].items()
                 for count in sorted({dose["k"] for dose in grid})
                 for policy in (["random"] if count == 0 else ["random", "sse_balanced"])]
        markers = [args.run / "units" / target["stem"] / f"{policy}-r{repeat}-k{count:04d}.complete.json"
                   for repeat, count, policy in modes]
        if not all(marker.exists() for marker in markers):
            continue
        marker_digest = hashlib.sha256(b"".join(marker.read_bytes() for marker in markers)).hexdigest()
        previous = index["proteins"].get(target["stem"])
        if previous:
            if previous["markers_sha256"] != marker_digest:
                raise ValueError(f"Already-published condition changed: {target['stem']}")
            continue
        files = {}
        for marker in markers:
            metadata = json.loads(marker.read_text())
            if metadata["plan_sha256"] != digest or metadata["n_rollouts"] != 100:
                raise ValueError(f"Incomplete or mismatched condition: {marker}")
            for extension in (".complete.json", ".timing.json", ".npz", ".json.gz"):
                path = Path(str(marker).replace(".complete.json", extension))
                if not path.exists():
                    break
                if extension in metadata["files_sha256"]:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["files_sha256"][extension]:
                        raise ValueError(f"Local prediction checksum mismatch: {path}")
                files[str(path.relative_to(args.run))] = path
        if len(files) != 4 * len(markers):
            continue
        archive = args.archives / f"{target['stem']}.tar.gz"
        result = build_archive(files, archive)
        uri = f"{DESTINATION}/predictions-v2/{target['stem']}-{result['sha256']}.tar.gz"
        if args.upload:
            subprocess.run(["hf", "buckets", "cp", str(archive), uri], check=True, stdout=subprocess.DEVNULL)
            index["proteins"][target["stem"]] = {"uri": uri, **result, "conditions": len(markers), "markers_sha256": marker_digest}
            INDEX.write_text(json.dumps(index, indent=2) + "\n")
        published += 1
        print(f"{'Published' if args.upload else 'Prepared'} {target['stem']}: {result['size_bytes']} bytes ({len(index['proteins'])}/{len(plan['targets'])} public)", flush=True)
        if args.limit is not None and published >= args.limit:
            break


if __name__ == "__main__":
    main()
