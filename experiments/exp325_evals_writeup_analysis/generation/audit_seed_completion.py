"""Reconstruct every prompt, vote and final contact set, then check raw coordinates."""

import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from seed_completion import complete_pairs, parse_pairs, state_digest

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Fail on changed seed pairs, vote counts, completion selection or output maps."""
    raw = ROOT / "scratch/seed_completion"
    maps_table = pd.read_csv(ROOT / "data/seed_completion_maps.csv").set_index(["stem", "arm", "map_seed"])
    helico_rows = pd.read_csv(ROOT / "data/helico_seed_completion_samples.csv", float_precision="round_trip")
    counts = Counter()
    with tarfile.open(ROOT / "scratch/seed_completion_coordinates.tar.gz") as archive:
        for path in sorted((raw / "inputs").glob("*.json")):
            inputs = json.loads(path.read_text())
            stem = inputs["stem"]
            mapping = {int(k): int(v) for k, v in inputs["mapping"].items()}
            with np.load(ROOT / "scratch/helico/seed_completion/maps" / f"{stem}.npz") as saved:
                for case in inputs["cases"]:
                    key = f"{case['arm']}-{case['replicate']}"
                    record = json.loads((raw / "results" / stem / f"{key}.json").read_text())
                    votes = Counter()
                    for rollout in record["rollouts"]:
                        position_map = {int(k): int(v) for k, v in rollout["position_map"].items()}
                        if parse_pairs(rollout["prompt"], position_map) != {tuple(pair) for pair in case["seed_pairs"]}:
                            raise ValueError(f"{stem}/{key}: prompt seed mismatch")
                        if rollout["finish_reason"] == "stop":
                            votes.update(parse_pairs(rollout["text"], position_map))
                        counts["rollouts"] += 1
                    recomputed = [[i, j, count] for (i, j), count in sorted(votes.items())]
                    if recomputed != record["votes"]:
                        raise ValueError("Raw text votes differ from saved votes")
                    additions = complete_pairs(case["seed_pairs"], recomputed, mapping, case["final_contacts"])
                    state = saved[key]
                    expected = np.zeros_like(state)
                    for i, j in case["seed_pairs"] + [[i, j] for i, j, _ in additions]:
                        expected[mapping[i], mapping[j]] = expected[mapping[j], mapping[i]] = 2
                    if not np.array_equal(state, expected):
                        raise ValueError("Completed conditioning differs from ranked pairs")
                    if state_digest(state) != maps_table.loc[(stem, case["arm"], case["replicate"]), "state_sha256"]:
                        raise ValueError("Prepared map digest mismatch")
                    member = f"seed_completion/{stem}/{key}"
                    with np.load(io.BytesIO(archive.extractfile(member + ".npz").read())) as prediction:
                        if not np.array_equal(prediction["contact_state"], state) or prediction["coords"].shape[0] != 3 or not np.isfinite(prediction["coords"]).all():
                            raise ValueError("Durable prediction state/coordinates mismatch")
                    scores = json.load(archive.extractfile(member + ".json"))["samples"]
                    frame = helico_rows[(helico_rows.stem == stem) & (helico_rows.arm == case["arm"]) &
                                        (helico_rows.map_seed == case["replicate"])].sort_values("sample_idx")
                    for metric in ("gdt_ts", "tm_score", "lddt", "ranking_score", "ptm"):
                        if not np.array_equal(frame[metric], [row[metric] for row in scores]):
                            raise ValueError("CSV metrics differ from durable raw scores")
                    counts["maps"] += 1
                    counts["structures"] += 3
    if dict(counts) != dict(rollouts=3500, maps=35, structures=105):
        raise ValueError(f"Unexpected audit counts: {counts}")
    (ROOT / "data/seed_completion_audit.json").write_text(json.dumps(dict(status="passed", **counts), indent=2) + "\n")
    print(dict(counts))


if __name__ == "__main__":
    main()
