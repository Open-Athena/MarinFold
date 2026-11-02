"""Fetch closed public artifacts; never download a checkpoint or an open sample file."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Download complete per-protein records with anonymous public access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--two-epoch", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    assert not (args.extend and args.two_epoch)
    plan_name = (
        "two_epoch_plan.json"
        if args.two_epoch
        else ("extension_plan.json" if args.extend else "plan.json")
    )
    plan = json.loads((HERE / "data" / plan_name).read_text())
    phase = "smoke" if args.smoke else "raw"
    root = plan["artifact_root"].removeprefix("hf://") + "/" + phase
    fs = HfFileSystem(token=False)
    suffix = "_two_epoch" if args.two_epoch else ("_1000" if args.extend else "")
    destination = HERE / "scratch" / (phase + suffix)
    destination.mkdir(exist_ok=True, parents=True)
    markers = fs.glob(root + "/*/*.complete.json")
    tasks = []
    summary = []
    for marker in markers:
        relative = Path(marker.removeprefix(root + "/"))
        path = destination / relative
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_text(fs.read_text(marker))
        data = json.loads(path.read_text())
        raw_uri = data["raw_uri"].removeprefix("hf://")
        assert raw_uri.startswith(root + "/")
        raw_path = destination / raw_uri.removeprefix(root + "/")
        if not raw_path.exists() and not args.metadata_only:
            tasks.append((raw_uri, raw_path))
        summary.append(
            {
                "model": relative.parts[0],
                **data["timing"],
                "count": data["count"],
                "unfinished": data["unfinished"],
                "extended": data["extended"],
            }
        )

    def fetch_one(task: tuple[str, Path]) -> str:
        remote, local = task
        partial = local.with_suffix(".partial")
        HfFileSystem(token=False).get_file(remote, str(partial))
        partial.rename(local)
        return str(local)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch_one, tasks))
    for worker in fs.glob(root + "/*/worker-*.json"):
        path = destination / worker.removeprefix(root + "/")
        path.parent.mkdir(exist_ok=True)
        path.write_text(fs.read_text(worker))
    if args.two_epoch:
        launches = json.loads((HERE / "data/launches_two_epoch.json").read_text())
        for remote, sha in sorted(
            {(r["source_uri"], r["source_sha256"]) for r in launches}
        ):
            assert remote == plan["artifact_root"] + f"/source/{sha}.zip"
            path = HERE / "scratch" / (sha + ".zip")
            if not path.exists():
                fetch_one((remote.removeprefix("hf://"), path))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    print(
        json.dumps(
            {
                "phase": phase,
                "complete_conditions": len(markers),
                "samples": sum(r["count"] for r in summary),
                "unfinished": sum(r["unfinished"] for r in summary),
                "extended": sum(r["extended"] for r in summary),
                "new_downloads": len(tasks),
            },
            indent=2,
        )
    )
    (destination / "status.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
