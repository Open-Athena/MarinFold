#!/usr/bin/env python
"""Select fixed candidate shortlists without importing any reference data."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from search_policy import canonical, diverse_indices

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARMS = ("iid", "temp", "random", "branch5", "branch10", "branch20")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=HERE / "_cache" / "raw")
    parser.add_argument("--shortlist-size", type=int, default=16)
    parser.add_argument("--novelty", type=float, default=1.0)
    args = parser.parse_args()
    files = sorted(args.raw.glob("shard-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no candidate parquet under {args.raw}")
    expected = set(pd.read_parquet(DATA / "search_targets.parquet").pair_id)
    if len(files) != len(expected):
        raise ValueError(f"found {len(files)} candidate files for {len(expected)} targets")
    chosen = []
    observed = set()
    for file in files:
        raw = pd.read_parquet(file)
        pair_id = raw.pair_id.iloc[0]
        if pair_id in observed:
            raise ValueError(f"duplicate candidate file for {pair_id}")
        observed.add(pair_id)
        for arm in ARMS:
            pool = raw[raw.arm.isin(["root", arm]) & raw.finished].reset_index(drop=True)
            if pool.empty or arm not in set(pool.arm):
                continue
            # The seed contacts are part of the generated document. The worker
            # records only continuation contacts in `contacts`, so reconstruct
            # each complete candidate before making a blind selection.
            maps = [canonical(row.contacts) | canonical(row.given)
                    for row in pool.itertuples()]
            indices = diverse_indices(maps, args.shortlist_size, args.novelty)
            for rank, idx in enumerate(indices, 1):
                row = pool.iloc[idx]
                chosen.append({"pair_id": pair_id, "method": arm,
                               "rank": rank, "candidate_id": row.candidate_id,
                               "source_arm": row.arm, "n_pred": len(maps[idx]),
                               "finished": bool(row.finished)})
    shortlist = pd.DataFrame(chosen).sort_values(["pair_id", "method", "rank"])
    if observed != expected:
        raise ValueError(f"candidate cohort mismatch: missing={expected-observed}, extra={observed-expected}")
    if shortlist.empty:
        raise ValueError("no candidates selected")
    DATA.mkdir(exist_ok=True)
    path = DATA / "sealed_shortlists.csv"
    shortlist.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (DATA / "sealed_shortlists.sha256").write_text(f"{digest}  sealed_shortlists.csv\n")
    print(f"sealed {len(shortlist)} candidates across {shortlist.pair_id.nunique()} proteins")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
