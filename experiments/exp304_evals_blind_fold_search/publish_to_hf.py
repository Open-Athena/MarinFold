#!/usr/bin/env python
"""Publish exp304 tables and per-rollout artifacts to the public HF bucket."""

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/contacts-v1-blind-fold-search-exp304"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for source, destination in (
        (HERE / "data", f"{DEST}/tables"),
        (HERE / "plots", f"{DEST}/plots"),
        (HERE / "_cache" / "raw", f"{DEST}/raw/blind-search-v1"),
        (HERE / "_cache" / "iid500", f"{DEST}/raw/iid500"),
        (HERE / "_cache" / "iid1000_tail_primary", f"{DEST}/raw/iid1000-tail-primary"),
        (HERE / "_cache" / "helico" / "data", f"{DEST}/helico/inputs"),
        (HERE / "_cache" / "helico" / "results", f"{DEST}/helico/results"),
        (HERE / "_cache" / "helico_iid" / "data", f"{DEST}/helico-iid/inputs"),
        (HERE / "_cache" / "helico_iid" / "results", f"{DEST}/helico-iid/results"),
        (HERE / "_cache" / "helico_iid" / "deck", f"{DEST}/helico-iid/deck"),
        (HERE / "_cache" / "joblogs", f"{DEST}/joblogs"),
    ):
        if not source.exists():
            raise FileNotFoundError(source)
        command = ["hf", "buckets", "sync", str(source), destination]
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
