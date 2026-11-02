"""Measure interface residues for the phase 5 candidates, one pass over the shards.

The candidates are 3.6% of the corpus but they are scattered across all 441 tar
shards, so the pass reads everything and computes only for the members it wants.
Reading is the dominant cost (37 min on 8 workers in-region), which is why the
candidate list is bounded per pair rather than covering every complex.

Structure of the run mirrors curate_extract_chains.py: a spawn-based pool over
shards, each worker rebuilding its own candidate set and its own pyarrow S3
filesystem, because a filesystem created before a fork is not safe in the child.
Workers return their rows rather than writing files, since the whole output is
about 32,000 rows.
"""

import argparse
import csv
import gzip
import io
import json
import multiprocessing
import os
import time
from pathlib import Path

from curate_extract_chains import iter_tar, list_shards, read_shard
from curate_interface_residues import interface_residues, parse_atoms

import pyarrow.fs

FIELDS = ["model_entity_id", "pair_label", "rank", "n_atoms",
          "interface_a", "interface_b", "min_side", "passes", "seconds"]

_STATE: dict = {}


def load_candidates(path: Path) -> dict[str, tuple[str, int]]:
    """Map each candidate model id to its pair label and its rank within the pair."""
    with gzip.open(path, "rt") as handle:
        return {row["model_entity_id"]: (row["pair_label"], int(row["rank"]))
                for row in csv.DictReader(handle)}


def init_worker(candidates: str, source: str, radius: float, minimum: int) -> None:
    _STATE["candidates"] = load_candidates(Path(candidates))
    _STATE["wanted"] = set(_STATE["candidates"])
    _STATE["radius"] = radius
    _STATE["minimum"] = minimum
    _STATE["fs"] = (pyarrow.fs.S3FileSystem(
        region=pyarrow.fs.resolve_s3_region(source[len("s3://"):].split("/")[0]))
        if source.startswith("s3://") else None)


def measure_shard(name: str) -> tuple[str, list[dict]]:
    """Interface counts for every candidate that lives in one shard."""
    candidates = _STATE["candidates"]
    radius = _STATE["radius"]
    rows: list[dict] = []
    for model, text in iter_tar(io.BytesIO(read_shard(name, _STATE["fs"])), name,
                                wanted=_STATE["wanted"]):
        pair_label, rank = candidates[model]
        mark = time.time()
        chains = parse_atoms(text)
        if len(chains) != 2:
            raise ValueError(f"{name}: {model} has {len(chains)} chains, expected 2")
        (first, first_res), (second, second_res) = (chains[key] for key in sorted(chains))
        left, right = interface_residues(first, second, first_res, second_res, radius)
        rows.append({
            "model_entity_id": model,
            "pair_label": pair_label,
            "rank": rank,
            "n_atoms": len(first) + len(second),
            "interface_a": left,
            "interface_b": right,
            "min_side": min(left, right),
            "passes": int(min(left, right) >= _STATE["minimum"]),
            "seconds": round(time.time() - mark, 4),
        })
    return name, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="a directory of .tar shards or an s3:// prefix holding them")
    parser.add_argument("--candidates", type=Path,
                        default=Path("data/curation/phase5_candidates.csv.gz"))
    parser.add_argument("--shard-stride", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--contact-angstroms", type=float, default=5.0)
    parser.add_argument("--min-interface", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-absent", type=int, default=0,
                        help="tolerated candidates never seen in any shard")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    shards = list_shards(args.source, args.shard_stride)[:args.max_shards]
    print(f"{len(candidates):,} candidates across {len(shards):,} shards, "
          f"{args.workers} workers, {args.contact_angstroms:g} A, "
          f"min {args.min_interface} residues both sides", flush=True)

    started = time.time()
    measured = 0
    context = multiprocessing.get_context("spawn")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle, context.Pool(
            args.workers, initializer=init_worker,
            initargs=(str(args.candidates), args.source,
                      args.contact_angstroms, args.min_interface)) as pool:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        seen: set[str] = set()
        passed = 0
        for done, (name, rows) in enumerate(
                pool.imap_unordered(measure_shard, shards), start=1):
            for row in rows:
                writer.writerow(row)
                seen.add(row["model_entity_id"])
                passed += row["passes"]
            measured += len(rows)
            handle.flush()
            print(f"[{time.time() - started:9.0f}s] {done}/{len(shards)} "
                  f"{Path(name).name} +{len(rows)} "
                  f"({measured:,} measured, {passed:,} pass)", flush=True)

    absent = sorted(set(candidates) - seen)
    summary = {
        "source": args.source,
        "n_shards": len(shards),
        "n_workers": args.workers,
        "contact_angstroms": args.contact_angstroms,
        "min_interface": args.min_interface,
        "n_candidates": len(candidates),
        "n_measured": measured,
        "n_passing": passed,
        "n_absent": len(absent),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    summary_path = args.out.parent / "phase5_interface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if len(absent) > args.max_absent:
        absent_path = args.out.parent / "phase5_interface_absent.csv"
        absent_path.write_text("model_entity_id\n" + "\n".join(absent) + "\n")
        raise SystemExit(f"{len(absent):,} candidates never appeared in a shard, "
                         f"listed in {absent_path}")


if __name__ == "__main__":
    main()
