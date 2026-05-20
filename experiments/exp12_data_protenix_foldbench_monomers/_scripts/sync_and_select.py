"""Stream Modal outputs to local disk one stem at a time, run select-best
per stem to keep only the top-1 sample's CIF/JSON/distogram, then drop
the raw download to free disk.

Net result: ``best/{mode}/{stem}/`` with structure.cif, confidence.json,
distogram.npz, provenance.json for each (paired) stem. The raw outputs/
dir is only populated transiently.

Reads /tmp/paired_stems.txt (one stem per line) — produced earlier by
``comm -12 done_ss.txt done_msa.txt``.
"""

import subprocess
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_best import select_best  # noqa: E402


def sync_stem(mode: str, stem: str, dest: Path) -> None:
    """``modal volume get`` one stem's outputs to ``dest/{mode}/{stem}/``."""
    target_parent = dest / mode
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / stem
    if target.exists():
        shutil.rmtree(target)
    cmd = [
        sys.executable, "-m", "modal", "volume", "get",
        "foldbench-protenix-runs", f"{mode}/{stem}", str(target_parent) + "/",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main(stems_file: Path, modes: list[str]) -> None:
    stems = [s.strip() for s in stems_file.read_text().splitlines() if s.strip()]
    print(f"syncing {len(stems)} stems × {len(modes)} modes...")
    runs_dir = Path("outputs")
    best_dir = Path("best")
    runs_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)
    n_done = 0
    for stem in stems:
        try:
            for mode in modes:
                sync_stem(mode, stem, runs_dir)
            select_best(
                runs_dir=runs_dir,
                out_dir=best_dir,
                modes=modes,
                stems=[stem],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {stem}: SYNC/SELECT FAILED ({type(e).__name__}: {e})")
            continue
        finally:
            for mode in modes:
                d = runs_dir / mode / stem
                if d.exists():
                    shutil.rmtree(d)
        n_done += 1
        if n_done % 5 == 0:
            print(f"  [{n_done}/{len(stems)}] processed")
    print(f"done: {n_done}/{len(stems)} stems → best/")


if __name__ == "__main__":
    main(
        stems_file=Path("/tmp/paired_stems.txt"),
        modes=["single_seq", "msa"],
    )
