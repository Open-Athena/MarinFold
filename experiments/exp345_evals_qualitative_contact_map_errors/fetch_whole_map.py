"""Download and validate complete individual rollout artifacts from CoreWeave."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import s3fs

HERE = Path(__file__).resolve().parent


def validate_raw(path: Path, record: dict, marker: dict) -> None:
    """Check completeness, sample identity, bounds and marker content integrity."""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == marker["raw_sha256"]
    frame = pq.read_table(path).to_pandas()
    assert len(frame) == 200 and sorted(frame.rollout) == list(range(200))
    assert set(frame.stem) == {record["stem"]} and set(frame.L) == {record["L"]}
    assert frame.sampling_seed.nunique() == 200 and frame.finish_reason.eq("stop").all()
    assert frame.groupby("pool").size().to_dict() == {"A": 100, "B": 100}
    assert (frame.generated_tokens <= frame.max_tokens).all()
    assert frame.cumulative_logprob.notna().all() and frame.mean_logprob.notna().all()
    for row in frame.itertuples():
        pairs = [tuple(map(int, pair)) for pair in row.contacts]
        assert len(pairs) == len(set(pairs)) == row.n_contacts
        assert all(0 <= i < j < record["L"] and j-i >= 6 for i, j in pairs)


def main() -> None:
    """Fetch the committed units, fail if expected units are absent, save timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE / "data/whole_map_plan.json").read_text())
    records = json.loads((HERE / "data/whole_map_targets.json").read_text())
    if args.smoke:
        records = [r for r in records if r["stem"] == "8arl_A"]
    phase = "smoke" if args.smoke else "production"
    root = plan["output"].removeprefix("s3://") + "/" + phase
    fs = s3fs.S3FileSystem(profile="cw", endpoint_url="https://cwobject.com",
                         config_kwargs={"s3": {"addressing_style": "virtual"}})
    markers = {Path(p).stem: p for p in fs.glob(root + "/complete/*.json")}
    expected = {r["stem"] for r in records}
    if not args.partial and set(markers) != expected:
        raise ValueError(f"Expected {len(expected)} complete proteins, got {len(markers)}; missing={sorted(expected-set(markers))}")
    cache = HERE / ".cache/whole_maps" / phase
    cache.mkdir(parents=True, exist_ok=True)

    def get_one(record: dict) -> tuple[dict, dict]:
        stem = record["stem"]
        marker = json.loads(fs.cat_file(markers[stem]))
        assert marker["checkpoint"] == plan["checkpoint"] and marker["unfinished_rollouts"] == 0
        assert marker["targets_sha256"] == plan["targets_sha256"]
        path = cache / f"{stem}.parquet"
        if not path.exists():
            path.write_bytes(fs.cat_file(root + f"/rollouts/{stem}.parquet"))
        validate_raw(path, record, marker)
        timing = json.loads(fs.cat_file(root + f"/timings/{stem}.json"))
        return marker, timing

    with ThreadPoolExecutor(max_workers=8) as pool:
        items = list(pool.map(get_one, [r for r in records if r["stem"] in markers]))
    print(f"Validated {len(items)}/{len(expected)} proteins, {len(items)*200} complete maps")
    if not args.partial:
        pd.DataFrame([timing for _, timing in items]).to_csv(HERE / f"data/whole_map_{phase}_timings.csv", index=False)
        (HERE / f"data/whole_map_{'smoke_validation' if args.smoke else 'raw_manifest'}.json").write_text(
            json.dumps({"source": "s3://" + root, "n_proteins": len(items), "n_maps": len(items)*200,
                        "markers": [marker for marker, _ in items]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
