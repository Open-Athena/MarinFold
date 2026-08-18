# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6 -- pull the baseline predictions down and score them like #74/#78 did.

Four structure predictors run on the 209 monomers that have never been
predicted: ESMFold and ESMFold2 (#78's Modal apps) and Protenix-v2 in
single-sequence and MSA mode (#12's app, top-1 of 40 samples by ranking score,
#74's selection). Each writes one mmCIF per protein to a Modal volume; this
syncs them into ``{pred_root}/{model}/{stem}/structure.cif`` -- the layout #78's
scorer expects -- and scores them with **#78's `evaluate_dataset` imported, not
reimplemented**, so the rows land in the same schema and metric as every
published baseline number.

**The control.** A sample of proteins that already have published scores is
re-scored through this path, from #78's own stored predictions, and compared to
#213's table. If this path were subtly different -- a different contact
threshold, a different candidate universe, the wrong chain -- that comparison is
where it shows. It runs on ESMFold and ESMFold2, whose predictions for the
published proteins are still on their Modal volumes; Protenix's published
samples live on a volume this run does not touch, and its only exp245-specific
step (top-1 selection) is #74's ``select_best`` imported unchanged.

    uv run --extra gt python score_baselines.py --sync      # modal -> /data/exp245
    uv run --extra gt python score_baselines.py --score
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
PRED_ROOT = U.WORK / "pred"
SCORES = U.WORK / "baseline_scores"
REPORT = DATA / "baseline_scoring.json"

#: ``modal`` outside the experiment venvs: the pinned client here is the one
#: that works for volume downloads on this workstation.
MODAL_BIN = "/home/bizon/anaconda3/bin/modal"

#: Modal volumes the predictions land in, and the local model directory each
#: becomes. Protenix is handled separately: its volume holds 40 samples per
#: protein and mode, and #74's `select_best` picks the one to score.
ESM_VOLUMES = {
    "esmfold": "esmfold-exp78-runs",
    "esmfold2": "esmfold2-exp78-runs",
}
PROTENIX_VOLUME = "exp245-protenix-runs"
PROTENIX_MODES = ("single_seq", "msa")
MODELS = ("esmfold", "esmfold2", "protenix-v2_single_seq", "protenix-v2_msa")

#: #213's published per-protein table, the reference the control compares to.
EXP213_WIDE = (U.EXPERIMENTS / "exp213_evals_train_sequence_overlap_audit"
               / "data" / "per_protein_wide.csv.gz")
EXP213_COLUMNS = {
    "esmfold": "ESMFold",
    "esmfold2": "ESMFold2",
    "protenix-v2_single_seq": "Protenix-v2 single-seq",
    "protenix-v2_msa": "Protenix-v2 + MSA",
}
#: Structure prediction is stochastic (Protenix samples; ESMFold2 diffuses), so
#: the control matches to a tolerance rather than to the digit. ESMFold is
#: deterministic and should reproduce far more tightly than this.
CONTROL_TOLERANCE = 0.02


def sync_volume(volume: str, stems: list[str], destination: Path) -> list[str]:
    """Download ``{stem}/structure.cif`` for each stem from a Modal volume.

    Through the Python SDK rather than ``modal volume get``: the CLI issues a
    ``VolumeListFiles`` call per invocation and a few hundred of those trip
    Modal's rate limiter within seconds, while one ``Volume`` handle streams
    files without listing. Resumable -- an existing non-empty file is skipped.
    """
    import modal

    destination.mkdir(parents=True, exist_ok=True)
    handle = modal.Volume.from_name(volume)
    missing: list[str] = []
    for index, stem in enumerate(stems, 1):
        target = destination / stem / "structure.cif"
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as handle_out:
                for chunk in handle.read_file(f"{stem}/structure.cif"):
                    handle_out.write(chunk)
        except Exception as error:  # noqa: BLE001 - one absent prediction must
            # not abort the sync; every missing stem is named in the report.
            target.unlink(missing_ok=True)
            missing.append(stem)
            print(f"  [sync] {stem}: {type(error).__name__}", flush=True)
        if index % 50 == 0:
            print(f"  [sync] {volume}: {index}/{len(stems)}", flush=True)
    return missing


def sync_protenix(stems: list[str], destination: Path) -> dict:
    """Sync Protenix samples and reduce them to one structure per mode.

    #74 owns the selection rule -- top-1 of the 40 samples by the model's own
    ranking score -- so its ``select_best`` is imported and called rather than
    restated.
    """
    exp74 = U.EXP74_DIR
    if str(exp74) not in sys.path:
        sys.path.insert(0, str(exp74))
    from select_best import select_best

    runs = U.WORK / "protenix_runs"
    runs.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [MODAL_BIN, "volume", "get", "--force", PROTENIX_VOLUME, "**", str(runs)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"modal volume get failed: {result.stderr[-2000:]}")
    best = U.WORK / "protenix_best"
    select_best(runs_dir=runs, out_dir=best, modes=list(PROTENIX_MODES), stems=stems)
    moved = {}
    for mode in PROTENIX_MODES:
        model_root = destination / f"protenix-v2_{mode}"
        model_root.mkdir(parents=True, exist_ok=True)
        count = 0
        for stem in stems:
            source = best / mode / stem / "structure.cif"
            if not source.exists():
                continue
            target = model_root / stem / "structure.cif"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            count += 1
        moved[mode] = count
    return moved


def score(manifest: Path, models: tuple[str, ...], out: Path) -> pd.DataFrame:
    """Score predicted structures with #78's evaluator, imported verbatim."""
    exp78 = U.EXP78_DIR
    if str(exp78) not in sys.path:
        sys.path.insert(0, str(exp78))
    from contact_eval import evaluate_dataset

    out.mkdir(parents=True, exist_ok=True)
    evaluate_dataset(
        manifest_csv=manifest, pred_root=PRED_ROOT, out_dir=out,
        models=models, gt_root=U.WORK / "cif", limit=None,
    )
    # #78 writes `contact_precision.csv` per invocation; the `_all` name in its
    # own data/ dir is the concatenation of its two manifests.
    return pd.read_csv(out / "contact_precision.csv")


def control(scored: pd.DataFrame) -> dict:
    """Compare re-scored published proteins against #213's table."""
    reference = pd.read_csv(EXP213_WIDE)
    reference = reference[reference["cut"].isin(("R", "AUC"))]
    comparisons = []
    for row in scored[(scored.predictor == "structure")
                      & scored["cut"].isin(("R", "AUC"))
                      & scored["range"].isin(("all", "long"))].itertuples():
        column = EXP213_COLUMNS.get(row.model)
        if column is None:
            continue
        match = reference[(reference.stem == row.stem)
                          & (reference["range"] == getattr(row, "range"))
                          & (reference["cut"] == row.cut)]
        if match.empty:
            continue
        expected = float(match.iloc[0][column])
        comparisons.append({
            "stem": row.stem, "model": row.model, "range": getattr(row, "range"),
            "cut": row.cut, "published": expected, "rescored": float(row.precision),
            "absolute_difference": abs(expected - float(row.precision)),
        })
    worst = max((c["absolute_difference"] for c in comparisons), default=0.0)
    return {
        "n_comparisons": len(comparisons),
        "max_absolute_difference": round(worst, 6),
        "tolerance": CONTROL_TOLERANCE,
        "passed": bool(comparisons) and worst <= CONTROL_TOLERANCE,
        "comparisons": sorted(comparisons, key=lambda c: -c["absolute_difference"])[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sync", action="store_true", help="download predictions")
    parser.add_argument("--score", action="store_true", help="score what is local")
    parser.add_argument("--control-manifest", type=Path,
                        default=DATA / "control_manifest.csv",
                        help="published proteins to re-score through this path")
    parser.add_argument("--control-models", default="esmfold,esmfold2",
                        help="models the control compares (default: the two "
                             "whose published predictions are still on Modal)")
    args = parser.parse_args()
    if not (args.sync or args.score):
        parser.error("pass --sync, --score, or both")

    manifest = pd.read_csv(DATA / "predictor_manifest_new.csv")
    stems = manifest.stem.tolist()
    report: dict = {"n_proteins": len(stems)}

    if args.sync:
        if args.control_manifest.exists():
            control_stems = pd.read_csv(args.control_manifest).stem.tolist()
            for model, volume in ESM_VOLUMES.items():
                missing = sync_volume(volume, control_stems, PRED_ROOT / model)
                report[f"{model}_control_missing"] = missing
        for model, volume in ESM_VOLUMES.items():
            missing = sync_volume(volume, stems, PRED_ROOT / model)
            report[f"{model}_missing"] = missing
            print(f"[sync] {model}: {len(stems) - len(missing)}/{len(stems)}", flush=True)
        report["protenix_synced"] = sync_protenix(stems, PRED_ROOT)
        print(f"[sync] protenix: {report['protenix_synced']}", flush=True)

    if args.score:
        scored = score(DATA / "predictor_manifest_new.csv", MODELS, SCORES / "new")
        scored.to_csv(DATA / "baseline_precision_new.csv.gz", index=False)
        report["scored_rows"] = int(len(scored))
        report["scored_units"] = int(scored.groupby("model").stem.nunique().to_dict()
                                     and scored.stem.nunique())
        report["scored_by_model"] = scored.groupby("model").stem.nunique().to_dict()
        if args.control_manifest.exists():
            control_models = tuple(args.control_models.split(","))
            control_scored = score(
                args.control_manifest, control_models, SCORES / "control")
            report["control"] = control(control_scored)
            print(json.dumps(report["control"], indent=2)[:800], flush=True)

    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("control",)}, indent=2)[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
