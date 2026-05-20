"""Stream raw Protenix outputs from Modal Volumes, score every sample, free disk.

For each (protein, mode) the Modal output Volume has 5 seeds × 8 diffusion
samples = 40 CIFs + 40 summary_confidence JSONs + 5 distograms. The
local ``best/`` tree only keeps the top-1 per (mode, stem). To compute
metrics on ALL samples we stream the raw dirs down per stem, score,
then delete — keeping peak local disk small.

The 48 originals live on Modal workspace ``timodonnell``, the 52
backfills on ``open-athena``. Profile is selected per stem via
``MODAL_PROFILE``.

Reads /tmp/paired_stems.txt (the 48) and /tmp/missing_stems.txt (the
52) to know which workspace each stem lives on.

CSV is opened in append mode + de-duped on (mode, stem, seed,
sample_idx), so a partial run is safely resumable.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import score_all_samples  # noqa: E402


def sync_stem_outputs(profile: str, mode: str, stem: str, dest: Path) -> bool:
    """``modal volume get`` outputs/{mode}/{stem}/ from the given profile."""
    parent = dest / mode
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / stem
    if target.exists():
        shutil.rmtree(target)
    env = {**os.environ, "MODAL_PROFILE": profile}
    cmd = [
        sys.executable, "-m", "modal", "volume", "get",
        "foldbench-protenix-runs", f"{mode}/{stem}", str(parent) + "/",
    ]
    res = subprocess.run(cmd, capture_output=True, env=env)
    return res.returncode == 0


def stems_to_profile() -> dict[str, str]:
    """Map each stem to the Modal profile (workspace) it lives on."""
    timo = [s.strip() for s in Path("/tmp/paired_stems.txt").read_text().splitlines() if s.strip()]
    oath = [s.strip() for s in Path("/tmp/missing_stems.txt").read_text().splitlines() if s.strip()]
    out = {s: "default" for s in timo}  # default profile == timodonnell workspace
    for s in oath:
        out[s] = "open-athena"
    return out


def main() -> None:
    out_csv = Path("data/scores_all_samples.csv")
    runs_dir = Path("outputs")
    inputs_dir = Path("inputs")
    modes = ["single_seq", "msa"]

    profile_for = stems_to_profile()
    # Process in manifest order so resumes are deterministic.
    import csv as _csv
    manifest = list(_csv.DictReader((inputs_dir / "manifest.csv").open()))
    stems = [r["stem"] for r in manifest]
    print(f"streaming {len(stems)} stems × {len(modes)} modes (8000 rows expected)")

    n_stems_done = 0
    for stem in stems:
        profile = profile_for.get(stem)
        if profile is None:
            print(f"WARN: no profile mapping for {stem}; skipping.")
            continue
        # Sync this stem's outputs/{mode}/{stem}/ for each mode.
        ok = True
        for mode in modes:
            if not sync_stem_outputs(profile, mode, stem, runs_dir):
                print(f"WARN: sync failed for {profile}/{mode}/{stem}; skipping stem.")
                ok = False
                break
        if not ok:
            for mode in modes:
                d = runs_dir / mode / stem
                if d.exists():
                    shutil.rmtree(d)
            continue
        try:
            score_all_samples(
                runs_dir=runs_dir, inputs_dir=inputs_dir, out_csv=out_csv,
                modes=modes, stems_filter=[stem], append=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"WARN: score failed for {stem}: {type(e).__name__}: {e}")
        finally:
            for mode in modes:
                d = runs_dir / mode / stem
                if d.exists():
                    shutil.rmtree(d)
        n_stems_done += 1
        if n_stems_done % 5 == 0:
            print(f"  ----- {n_stems_done}/{len(stems)} stems processed -----")
    print(f"done: {n_stems_done}/{len(stems)} stems processed → {out_csv}")


if __name__ == "__main__":
    main()
