# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture per-candidate Foldseek-search timing → ``data/timings.csv``.

Per the repo's "capture timings for every predictor run" rule
(``AGENTS.md``), the query tool is a predictor-like run and we record
per-input wall-time + worker metadata at evaluation time rather than
reconstructing it from logs.

Foldseek ``easy-search`` is normally *batched* (one process searches all
candidates at once), so this script records two modes:

- ``per_candidate`` — one ``easy-search`` per candidate cif, giving a true
  per-input ``elapsed_seconds`` (the AGENTS per-input convention and the
  time-vs-length curve). Each call pays Foldseek's process + query-DB
  setup, so these times are an upper bound on the amortized batched cost.
- ``batched`` — a single ``easy-search`` over the whole candidate dir
  (``stem == "__ALL__"``), the realistic throughput.

Columns keep the repo's shared timing-schema fields even when some are
blank here (no GPU, no torch, no separately measurable model-load phase),
then add Foldseek-specific context like ``n_db_reps``, ``alignment_type``,
and ``n_hits``. ``n_db_reps`` is essential context: search cost scales with
the target DB, and the prototype DB (~229 reps) is tiny next to the full
~1.3M-rep training DB, so absolute times here are dominated by per-call
startup, not DB scaling.

Example::

    uv run python collect_timings.py \
        --candidate-dir candidates/foldbench/data/protenix-foldbench-monomers/gt \
        --db db_full/db/targetDB --db-tag afdb-24M-full-reps-1331330 --no-per-candidate
"""

import argparse
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from foldseek_env import foldseek_version
from query_similarity import (
    DEFAULT_MAX_SEQS,
    _strip_struct_ext,
    count_residues,
    easy_search,
    iter_candidate_files,
    parse_m8,
)

_FIELDS = [
    "stem",
    "n_residues",
    "n_pairs",
    "mode",
    "elapsed_seconds",
    "model_load_seconds",
    "total_seconds",
    "model_nickname",
    "gpu_name",
    "gpu_total_memory_gb",
    "gpu_compute_capability",
    "runner_tag",
    "hostname",
    "platform",
    "torch_version",
    "timestamp_utc",
    "n_db_reps",
    "alignment_type",
    "max_seqs",
    "n_hits",
    "cpu_model",
    "n_cpus",
    "foldseek_version",
    "db_snapshot_tag",
]


def _cpu_model() -> str:
    """Best-effort CPU model string (Linux /proc/cpuinfo, else platform)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def db_n_reps(db_prefix: Path) -> int:
    """Number of entries in a Foldseek DB (one line per entry in .lookup)."""
    lookup = Path(str(db_prefix) + ".lookup")
    if not lookup.exists():
        return -1
    return sum(1 for _ in lookup.open())


def worker_meta(
    db_prefix: Path,
    db_snapshot_tag: str,
    alignment_type: int,
    max_seqs: int,
) -> dict:
    """Constant per-run metadata stamped onto every timing row."""
    return {
        "n_db_reps": db_n_reps(db_prefix),
        "alignment_type": alignment_type,
        "max_seqs": max_seqs,
        "model_nickname": "foldseek-easy-search-tmalign",
        "gpu_name": "",  # CPU TM-align; set if --gpu search is added later
        "gpu_total_memory_gb": None,
        "gpu_compute_capability": None,
        "runner_tag": "local",
        "cpu_model": _cpu_model(),
        "n_cpus": os.cpu_count() or -1,
        # Anonymous by default so a personal machine name (e.g. a laptop's
        # ``socket.gethostname()``) never lands in a committed CSV. Set
        # ``MARINFOLD_TIMING_HOSTNAME`` when worker traceability is wanted
        # (e.g. a named cloud/iris worker). cpu_model + platform already give
        # the hardware context the timing comparison needs.
        "hostname": os.environ.get("MARINFOLD_TIMING_HOSTNAME", ""),
        "platform": platform.platform(),
        "torch_version": None,
        "foldseek_version": foldseek_version(),
        "db_snapshot_tag": db_snapshot_tag,
    }


def _time_search(
    query: Path,
    db_prefix: Path,
    alignment_type: int,
    max_seqs: int,
) -> tuple[float, int]:
    """Run one easy-search over ``query`` (file or dir); return (seconds, n_hits)."""
    with tempfile.TemporaryDirectory(prefix="fs_timing_") as td:
        tmp = Path(td)
        out_m8 = tmp / "aln.m8"
        t0 = time.perf_counter()
        easy_search(
            query,
            db_prefix,
            out_m8,
            tmp / "fs_tmp",
            alignment_type=alignment_type,
            max_seqs=max_seqs,
        )
        elapsed = time.perf_counter() - t0
        n_hits = len(parse_m8(out_m8))
    return elapsed, n_hits


def collect(
    candidate_dir: Path,
    db_prefix: Path,
    out_csv: Path,
    *,
    db_snapshot_tag: str = "unknown",
    alignment_type: int = 1,
    max_seqs: int = DEFAULT_MAX_SEQS,
    per_candidate: bool = True,
) -> pd.DataFrame:
    """Time the foldseek search per candidate (+ batched) and write the CSV."""
    cifs = iter_candidate_files(candidate_dir)
    if not cifs:
        raise RuntimeError(f"no candidate structures under {candidate_dir}")
    meta = worker_meta(db_prefix, db_snapshot_tag, alignment_type, max_seqs)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    res_by_file: dict[Path, int] = {}
    if per_candidate:
        for i, cif in enumerate(cifs):
            stem = _strip_struct_ext(cif.name)
            n_res = count_residues(cif)
            res_by_file[cif] = n_res
            elapsed, n_hits = _time_search(cif, db_prefix, alignment_type, max_seqs)
            rows.append({
                "stem": stem,
                "n_residues": n_res,
                "n_pairs": None,
                "mode": "per_candidate",
                "elapsed_seconds": round(elapsed, 4),
                "model_load_seconds": None,
                "total_seconds": round(elapsed, 4),
                "n_hits": n_hits,
                "timestamp_utc": stamp,
                **meta,
            })
            print(f"[{i + 1}/{len(cifs)}] {stem}: {n_res} res, {elapsed:.3f}s, {n_hits} hits")

    # Batched: one search over the whole dir = realistic throughput. Reuse the
    # per-candidate residue counts when we have them instead of re-parsing.
    total_res = (
        sum(res_by_file.values())
        if per_candidate
        else sum(count_residues(c) for c in cifs)
    )
    elapsed, n_hits = _time_search(candidate_dir, db_prefix, alignment_type, max_seqs)
    rows.append({
        "stem": "__ALL__",
        "n_residues": total_res,
        "n_pairs": None,
        "mode": "batched",
        "elapsed_seconds": round(elapsed, 4),
        "model_load_seconds": None,
        "total_seconds": round(elapsed, 4),
        "n_hits": n_hits,
        "timestamp_utc": stamp,
        **meta,
    })
    print(f"[batched] {len(cifs)} candidates in {elapsed:.3f}s "
          f"({elapsed / len(cifs) * 1000:.1f} ms/candidate amortized)")

    df = pd.DataFrame(rows)[_FIELDS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows; {meta['n_db_reps']} reps in DB).")
    return df


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True, help="Foldseek target DB prefix")
    ap.add_argument("--out", type=Path, default=here / "data" / "timings.csv")
    ap.add_argument("--db-tag", default="unknown")
    ap.add_argument("--alignment-type", type=int, default=1)
    ap.add_argument("--max-seqs", type=int, default=DEFAULT_MAX_SEQS)
    ap.add_argument("--no-per-candidate", action="store_true", help="Only the batched row")
    args = ap.parse_args()

    collect(
        args.candidate_dir,
        args.db,
        args.out,
        db_snapshot_tag=args.db_tag,
        alignment_type=args.alignment_type,
        max_seqs=args.max_seqs,
        per_candidate=not args.no_per_candidate,
    )


if __name__ == "__main__":
    main()
