"""Fetch closed public artifacts; never download a checkpoint or an open sample file."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Download complete per-protein records with anonymous public access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE / "data/plan.json").read_text())
    phase = "smoke" if args.smoke else "raw"
    root = plan["artifact_root"].removeprefix("hf://") + "/" + phase
    fs = HfFileSystem(token=False)
    destination = HERE / "scratch" / phase
    destination.mkdir(exist_ok=True, parents=True)
    markers = fs.glob(root + "/*/*.complete.json")
    tasks = []
    summary = []
    for marker in markers:
        data = json.loads(fs.read_text(marker))
        relative = Path(marker.removeprefix(root + "/"))
        path = destination / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
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
