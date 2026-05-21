"""Audit foldbench-protenix-runs Volume: per (mode, stem), count seeds with
all expected outputs (CIF + summary_confidence + distogram)."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal_app  # noqa: E402

import modal  # noqa: E402

OUT_VOL = modal.Volume.from_name("foldbench-protenix-runs")


@modal_app.app.function(volumes={"/outputs": OUT_VOL}, timeout=120, cpu=0.5)
def audit_protein(mode: str, stem: str) -> dict:
    base = Path("/outputs") / mode / stem
    if not base.exists():
        return {"mode": mode, "stem": stem, "complete_seeds": 0, "any_seed": False}
    seeds_complete = 0
    seeds_any = 0
    for seed in (1, 2, 3, 4, 5):
        seed_dir = base / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        seeds_any += 1
        has_cif = any(seed_dir.glob(f"{stem}_sample_*.cif"))
        has_conf = any(seed_dir.glob(f"{stem}_summary_confidence_sample_*.json"))
        has_dist = (seed_dir / f"{stem}_distogram.npz").exists()
        if has_cif and has_conf and has_dist:
            seeds_complete += 1
    return {
        "mode": mode, "stem": stem,
        "complete_seeds": seeds_complete,
        "any_seed": seeds_any > 0,
    }


def main() -> None:
    manifest = list(csv.DictReader(Path("inputs/manifest.csv").open()))
    args = [(m, r["stem"]) for m in ("single_seq", "msa") for r in manifest]
    with modal_app.app.run():
        results = list(audit_protein.starmap(args))
    n_total = len(results)
    n_complete = sum(1 for r in results if r["complete_seeds"] == 5)
    n_partial = sum(1 for r in results if 0 < r["complete_seeds"] < 5)
    n_missing = sum(1 for r in results if r["complete_seeds"] == 0)
    print(f"complete (5/5 seeds): {n_complete} / {n_total}")
    print(f"partial:              {n_partial}")
    print(f"missing:              {n_missing}")
    if n_partial or n_missing:
        for r in results:
            if r["complete_seeds"] != 5:
                print(f"  {r['mode']}/{r['stem']}: {r['complete_seeds']}/5 seeds")


if __name__ == "__main__":
    main()
