"""Require valid long-target generation before expanding CoreWeave fan-out."""

import gzip
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfFileSystem

from select_diversity import cached, parse_contacts, sample_seed

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Validate both 900-sample smoke outputs and freeze the launch gate."""
    plan = json.loads((HERE / "data/diversity_plan.json").read_text())
    target = max(
        json.loads((HERE / "inputs/diversity/targets.json").read_text()),
        key=lambda t: (t["L"], t["stem"]),
    )
    fs = HfFileSystem(token=False)
    records = {}
    for model in plan["models"]:
        key = f"{target['stem']}-t1.0-randomized"
        uri = f"{plan['artifact_root']}/raw/{model}/{key}.json.gz"
        path = HERE / "scratch/diversity/raw" / model / f"{key}.json.gz"
        data = cached(fs, uri, path)
        raw = json.loads(gzip.decompress(data))
        marker_uri = uri.replace(".json.gz", ".complete.json")
        marker = json.loads(cached(fs, marker_uri, path.with_suffix(".complete.json")))
        assert marker["count"] == 900 and marker["unfinished"] == 0
        assert raw["model"] == model and raw["target"] == target
        assert raw["temperature"] == 1 and raw["mode"] == "randomized"
        assert [s["rollout"] for s in raw["samples"]] == list(range(100, 1000))
        for sample in raw["samples"]:
            k = sample["rollout"]
            assert sample["finish_reason"] == "stop"
            assert sample["seed"] == sample_seed(
                target["stem"], "randomized", k // 100, k % 100
            )
            parsed = parse_contacts(sample["text"], sample["offset"], target["L"])
            assert [list(p) for p in parsed["contacts"]] == sample["contacts"]
        records[model] = {**marker, "raw_sha256": hashlib.sha256(data).hexdigest()}
    (HERE / "data/diversity_smoke.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print("Both 900-sample generation gates passed")


if __name__ == "__main__":
    main()
