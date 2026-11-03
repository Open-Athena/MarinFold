"""Score fixed residue/atom correspondences; never realign by structural quality."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray, lddt

HERE = Path(__file__).resolve().parent
METRICS = ("tm_score", "lddt", "gdt_ts")


def ca_pdb(coords: np.ndarray, residue_ids: np.ndarray) -> str:
    """Write CA coordinates with fixed one-based residue correspondence."""
    lines = []
    for i, (xyz, rid) in enumerate(zip(coords, residue_ids, strict=True)):
        if not np.isfinite(xyz).all():
            continue
        x, y, z = xyz
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{int(rid) + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    return "\n".join(lines) + "\nTER\nEND\n"


def tmscore(
    pred: np.ndarray, ref: np.ndarray, residue_ids: np.ndarray, folder: Path
) -> dict:
    """Run reference-normalized TMscore and optimized GDT-TS with fixed numbering."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pred_ca.pdb").write_text(ca_pdb(pred, residue_ids))
    (folder / "ref_ca.pdb").write_text(ca_pdb(ref, residue_ids))
    stdout = subprocess.check_output(
        [
            str(HERE / "scratch/TMscore"),
            "pred_ca.pdb",
            "ref_ca.pdb",
        ],
        cwd=folder,
        text=True,
    )
    (folder / "TMscore.txt").write_text(stdout)
    tm = re.search(r"^TM-score\s*=\s*([\d.]+)", stdout, re.MULTILINE)
    gdt = re.search(r"^GDT-TS-score=\s*([\d.]+)", stdout, re.MULTILINE)
    length = re.search(r"^Structure2:.*Length=\s*(\d+)", stdout, re.MULTILINE)
    if tm is None or gdt is None or length is None or int(length[1]) != len(ref):
        raise ValueError(f"invalid TMscore output: {stdout}")
    return dict(tm_score=float(tm[1]), gdt_ts=float(gdt[1]))


def atom_lddt(
    reference: np.ndarray,
    predicted: np.ndarray,
    residue_ids: np.ndarray,
    names: np.ndarray,
    elements: np.ndarray,
) -> float:
    """Compute all-heavy-atom lDDT, counting missing subject atoms as broken distances."""
    atoms = AtomArray(len(reference))
    atoms.coord = reference
    atoms.res_id = residue_ids + 1
    atoms.chain_id[:] = "A"
    atoms.atom_name = names
    atoms.element = elements
    return float(
        lddt(
            atoms,
            predicted,
            inclusion_radius=15,
            distance_bins=(0.5, 1.0, 2.0, 4.0),
            exclude_same_residue=True,
        )
    )


def score_one(folder: Path) -> dict:
    """Score one saved structure against every resolved native heavy atom."""
    with np.load(folder / "structure.npz", allow_pickle=False) as arrays:
        ca = arrays["ref_atom_names"] == "CA"
        ref, pred = arrays["ref_coords"], arrays["aligned_pred"]
        coverage = arrays["covered"]
        if np.any(~np.isfinite(pred[coverage])) or not np.isfinite(ref).all():
            raise ValueError("nonfinite covered coordinates")
        metrics = tmscore(
            pred[ca], ref[ca], arrays["ref_res_ids"][ca], folder / "scoring"
        )
        metrics.update(
            lddt=atom_lddt(
                ref,
                pred,
                arrays["ref_res_ids"],
                arrays["ref_atom_names"],
                arrays["ref_elements"],
            ),
            lddt_ca=atom_lddt(
                ref[ca],
                pred[ca],
                arrays["ref_res_ids"][ca],
                arrays["ref_atom_names"][ca],
                arrays["ref_elements"][ca],
            ),
            n_reference_atoms=len(ref),
            n_matched_atoms=int(coverage.sum()),
            atom_coverage=float(coverage.mean()),
            n_reference_ca=int(ca.sum()),
            ca_coverage=float(coverage[ca].mean()),
        )
        if any(not 0 <= metrics[m] <= 1 for m in METRICS):
            raise ValueError(f"invalid metrics: {metrics}")
    return metrics


def main(smoke: bool) -> None:
    """Validate and score either the smoke gate or all 200 predictions."""
    build = json.loads((HERE / "data/tmscore_build.json").read_text())
    if (
        hashlib.sha256((HERE / "scratch/TMscore").read_bytes()).hexdigest()
        != build["binary_sha256"]
    ):
        raise ValueError("TMscore binary differs from build manifest")
    tasks = json.loads((HERE / "inputs/tasks.json").read_text())
    plan = json.loads((HERE / "data/plan.json").read_text())
    plan_sha = hashlib.sha256((HERE / "data/plan.json").read_bytes()).hexdigest()
    if smoke:
        tasks = [
            t for t in tasks if t["stem"] == plan["stems"][0] and t["rollout"] == 0
        ]
    rows, timing_rows, missing_atoms = [], [], []
    runtime_versions = set()
    for task in tasks:
        folder = HERE / "scratch/results" / task["task_id"]
        metadata = json.loads((folder / "result.json").read_text())
        runtime_versions.add(json.dumps(metadata["versions"], sort_keys=True))
        if (
            metadata["input_sha256"] != task["input_sha256"]
            or metadata["plan_sha256"] != plan_sha
        ):
            raise ValueError("input identity mismatch")
        for name, expected in metadata["output_sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError("structure digest mismatch")
        scalar = {k: v for k, v in metadata.items() if not isinstance(v, dict)}
        metrics = score_one(folder)
        if (
            metrics["ca_coverage"] != 1
            or metrics["n_reference_ca"] != metadata["n_residues"]
        ):
            raise ValueError("incomplete CA coverage; investigate before reporting")
        if not np.isfinite(metadata["ranking_score"]):
            raise ValueError("nonfinite selector score")
        rows.append({**scalar, **metrics})
        timing_rows.append(scalar)
        if task["model"] == "default" and task["rollout"] == 0:
            with np.load(folder / "structure.npz", allow_pickle=False) as arrays:
                for index in np.flatnonzero(~arrays["covered"]):
                    missing_atoms.append(
                        dict(
                            stem=task["stem"],
                            resolved_residue_index_0based=int(
                                arrays["ref_res_ids"][index]
                            ),
                            atom_name=str(arrays["ref_atom_names"][index]),
                            element=str(arrays["ref_elements"][index]),
                        )
                    )
    if smoke:
        (HERE / "data/smoke_passed.json").write_text(
            json.dumps(
                dict(
                    plan_sha256=plan_sha,
                    n_predictions=len(rows),
                    checks=[
                        "checkpoint digest",
                        "paired source prompts/seeds",
                        "effective contact count",
                        "finite coordinates/confidence",
                        "full CA coverage",
                        "all three metrics finite",
                    ],
                    tasks=[r["task_id"] for r in rows],
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            pd.DataFrame(rows)[["task_id", *METRICS, "atom_coverage"]].to_string(
                index=False
            )
        )
        return
    assert len(rows) == 200 and len({r["task_id"] for r in rows}) == 200
    if len({r["runner_sha256"] for r in rows}) != 1 or len(runtime_versions) != 1:
        raise ValueError("mixed inference source or runtime versions")
    pd.DataFrame(rows).to_csv(HERE / "data/per_rollout_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(HERE / "data/timings.csv", index=False)
    pd.DataFrame(
        missing_atoms,
        columns=["stem", "resolved_residue_index_0based", "atom_name", "element"],
    ).to_csv(HERE / "data/missing_reference_atoms.csv", index=False)
    manifest = dict(
        n_targets=10,
        n_models=2,
        n_predictions=200,
        plan_sha256=plan_sha,
        tmscore_build=build,
        minimum_atom_coverage=min(r["atom_coverage"] for r in rows),
        minimum_ca_coverage=min(r["ca_coverage"] for r in rows),
        helico_source_sha=plan["helico_source_sha"],
        helico_checkpoint_sha256=plan["helico_checkpoint_sha256"],
        runner_sha256=sorted({r["runner_sha256"] for r in rows}),
        scoring_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_versions=metadata["versions"],
    )
    (HERE / "data/run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    seconds = np.array([r["elapsed_seconds"] for r in rows])
    workers = {r["worker_id"]: r["model_load_seconds"] for r in rows}
    execution = dict(
        n_predictions=len(rows),
        n_workers_including_smoke=len(workers),
        inference_seconds=float(seconds.sum()),
        inference_gpu_hours=float(seconds.sum() / 3600),
        inference_gpu_usd_at_0_001097_per_second=float(seconds.sum() * 0.001097),
        median_inference_seconds=float(np.median(seconds)),
        min_inference_seconds=float(seconds.min()),
        max_inference_seconds=float(seconds.max()),
        unique_worker_setup_seconds=sum(workers.values()),
        nominal_h100_usd_per_second=0.001097,
        cost_note="Inference-time component only, not a Modal invoice; excludes startup, failed startup gates, volume commits, idle time, CPU and RAM.",
        all_ranking_scores_equal_ptm=all(r["ranking_score"] == r["ptm"] for r in rows),
        n_has_clash=sum(r["has_clash"] for r in rows),
        earliest_prediction_utc=min(r["timestamp_utc"] for r in rows),
        latest_prediction_utc=max(r["timestamp_utc"] for r in rows),
    )
    (HERE / "data/execution_summary.json").write_text(
        json.dumps(execution, indent=2) + "\n"
    )
    print(
        f"Scored {len(rows)} structures; minimum heavy-atom coverage {manifest['minimum_atom_coverage']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
