"""Pull per-(mode, stem) timings.json from the Modal output Volume
and aggregate into ``data/timings.csv``.

Output schema mirrors exp20's ``data/timings.csv`` columns where they
apply, plus exp12-specific columns:

  stem, n_residues, n_pairs, mode, n_seeds, n_samples_per_seed, n_cycle,
  elapsed_seconds, model_load_seconds, total_seconds,
  model_nickname, gpu_name, gpu_total_memory_gb, gpu_compute_capability,
  runner_tag, hostname, platform, torch_version, timestamp_utc

n_pairs = N*(N-1)/2 (upper-triangle CB-CB pair count). Matches exp20's
column definition for cross-experiment joins.

Streams per-stem from the Modal Volume into a temp local dir, parses
the timings.json, then drops the local copy. Peak local disk stays
in the KB range (timings.json files are ~700 bytes each).
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CSV_FIELDS = [
    # Identity.
    "stem", "n_residues", "n_pairs", "mode",
    # Settings.
    "n_seeds", "n_samples_per_seed", "n_cycle",
    # Timings.
    "elapsed_seconds", "model_load_seconds", "total_seconds",
    # Provenance.
    "model_nickname", "gpu_name", "gpu_total_memory_gb",
    "gpu_compute_capability", "runner_tag",
    "hostname", "platform", "torch_version", "timestamp_utc",
]


def sync_timings_json(mode: str, stem: str, dest: Path) -> Path | None:
    """``modal volume get`` /outputs/{mode}/{stem}/timings.json to local."""
    target_dir = dest / mode / stem
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "modal", "volume", "get",
        "foldbench-protenix-runs", f"{mode}/{stem}/timings.json",
        str(target_dir / "timings.json"),
    ]
    res = subprocess.run(cmd, capture_output=True, env=os.environ.copy())
    if res.returncode != 0:
        return None
    return target_dir / "timings.json"


def main() -> None:
    stems_file = Path("/tmp/timing_stems.txt")
    if not stems_file.exists():
        raise SystemExit(f"missing {stems_file} — did dispatch_timing.py run?")
    stems = [s.strip() for s in stems_file.read_text().splitlines() if s.strip()]

    manifest = list(csv.DictReader(open("inputs/manifest.csv")))
    by_stem = {r["stem"]: r for r in manifest}

    out_csv = Path("data/timings.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="exp12_timings_") as scratch:
        scratch_dir = Path(scratch)
        for stem in stems:
            for mode in ("single_seq", "msa"):
                p = sync_timings_json(mode, stem, scratch_dir)
                if p is None:
                    print(f"WARN: no timings.json for {mode}/{stem}")
                    continue
                t = json.loads(p.read_text())
                n = int(by_stem[stem]["n_residues"])
                row = {
                    "stem": stem,
                    "n_residues": n,
                    "n_pairs": n * (n - 1) // 2,
                    "mode": mode,
                    "n_seeds": t.get("n_seeds"),
                    "n_samples_per_seed": t.get("n_samples_per_seed"),
                    "n_cycle": t.get("n_cycle"),
                    "elapsed_seconds": t.get("elapsed_seconds"),
                    "model_load_seconds": t.get("model_load_seconds"),
                    "total_seconds": t.get("total_seconds"),
                    "model_nickname": t.get("model_nickname"),
                    "gpu_name": t.get("gpu_name"),
                    "gpu_total_memory_gb": t.get("gpu_total_memory_gb"),
                    "gpu_compute_capability": t.get("gpu_compute_capability"),
                    "runner_tag": t.get("runner_tag"),
                    "hostname": t.get("hostname"),
                    "platform": t.get("platform"),
                    "torch_version": t.get("torch_version"),
                    "timestamp_utc": t.get("timestamp_utc"),
                }
                rows.append(row)
                print(
                    f"{mode}/{stem}: n_res={n} "
                    f"elapsed={row['elapsed_seconds']}s "
                    f"load={row['model_load_seconds']}s"
                )

    # Sort for stable output: by mode then n_residues.
    rows.sort(key=lambda r: (r["mode"], r["n_residues"]))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
