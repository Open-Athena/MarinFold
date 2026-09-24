#!/usr/bin/env python
"""Preserve observed per-pair wall time alongside pure inference timings."""

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(
    r"\[exp304\] \d+/\d+ ([^ ]+) L=(\d+) maps=(\d+) time=([\d.]+)s finished=(\d+)/(\d+)"
)


def main() -> None:
    rows = []
    for path in sorted((HERE / "_cache" / "joblogs").glob("*.log")):
        run = "iid500" if "iid500" in path.name else "blind-search-v1"
        for match in PATTERN.finditer(path.read_text()):
            pair_id, length, maps, wall, finished, attempted = match.groups()
            rows.append({"run": run, "pair_id": pair_id, "L": int(length),
                         "maps": int(maps), "pair_wall_seconds": float(wall),
                         "finished": int(finished), "attempted": int(attempted),
                         "pod": path.stem})
    table = pd.DataFrame(rows)
    if len(table) != 134 or table.duplicated(["run", "pair_id"]).any():
        raise ValueError("expected exactly one completed log entry per run and protein")
    table["inference_seconds"] = float("nan")
    table["amortized_model_load_seconds"] = float("nan")
    for run, path in (("iid500", HERE / "data" / "timings_iid500.csv"),
                      ("blind-search-v1", HERE / "data" / "timings.csv")):
        source = pd.read_csv(path).groupby("pair_id").agg(
            inference_seconds=("elapsed_seconds", "sum"),
            amortized_model_load_seconds=("model_load_seconds", "sum"),
        )
        selected = table[table.run == run].set_index("pair_id").join(
            source, validate="one_to_one", rsuffix="_source"
        )
        if selected.inference_seconds_source.isna().any():
            raise ValueError(f"missing pure inference timings for {run}")
        table.loc[table.run == run, "inference_seconds"] = (
            selected.inference_seconds_source.to_numpy()
        )
        table.loc[table.run == run, "amortized_model_load_seconds"] = (
            selected.amortized_model_load_seconds_source.to_numpy()
        )
    table["other_pair_seconds"] = table.pair_wall_seconds - table.inference_seconds
    table.sort_values(["run", "pair_id"]).to_csv(HERE / "data" / "pair_wall_times.csv", index=False)
    print(table.groupby("run")[["pair_wall_seconds", "inference_seconds", "other_pair_seconds"]].sum())


if __name__ == "__main__":
    main()
