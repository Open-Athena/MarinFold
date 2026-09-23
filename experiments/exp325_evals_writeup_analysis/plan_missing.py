"""Freeze target lists and controls for the remaining predictor work; submit nothing.

The confidence comparison must match sequence length, eligible-pair mask,
positive-contact count, diffusion budget and confidence selection. Reusing the
full oracle while giving a random control only L positives is not a fair test.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def randomized_map(state: np.ndarray, seed: int, match_separation: bool, residue_positions: np.ndarray | None = None) -> np.ndarray:
    """Randomize a known three-state contact map without changing its information budget.

    Args:
        state: Symmetric matrix with 0 unknown, 1 absent, 2 present. The caller
            must explicitly translate the predictor's state encoding to this one.
        seed: Deterministic map randomization seed.
        match_separation: Preserve present-pair counts separately at |i-j| 6–11,
            12–23 and >=24. Indices must be the predictor's residue coordinates.

    Returns:
        A new matrix with identical unknown mask and number of present pairs.
        Chance overlap with the oracle is retained, rather than forcing false pairs.
    """
    if state.ndim != 2 or state.shape[0] != state.shape[1] or not np.array_equal(state, state.T):
        raise ValueError("Expected a square symmetric map")
    if not np.isin(state, [0, 1, 2]).all() or np.any(np.diag(state) != 0):
        raise ValueError("Expected 0=unknown, 1=absent, 2=present and unknown diagonal")
    i, j = np.where(np.triu(state != 0, 1))
    positions = np.arange(len(state)) if residue_positions is None else np.asarray(residue_positions)
    if positions.shape != (len(state),):
        raise ValueError("Residue position vector does not match token map")
    separation = np.abs(positions[j] - positions[i])
    if (separation < 6).any():
        raise ValueError("Expected only eligible pairs at separation >=6")
    bins = np.digitize(separation, [12, 24]) if match_separation else np.zeros(len(i), dtype=int)
    rng = np.random.default_rng(seed)
    out = state.copy()
    out[i, j] = out[j, i] = 1
    for group in np.unique(bins):
        candidates = np.flatnonzero(bins == group)
        count = int((state[i[candidates], j[candidates]] == 2).sum())
        chosen = rng.choice(candidates, size=count, replace=False)
        out[i[chosen], j[chosen]] = out[j[chosen], i[chosen]] = 2
    return out


def main() -> None:
    """Write a bounded full-test plan and a balanced 20-target confidence screen."""
    targets = pd.read_csv(DATA / "targets.csv")
    missing = targets[targets.eval_set == "eval-test"].sort_values("stem")
    missing.to_csv(DATA / "missing_eval_test_targets.csv", index=False)
    natural = targets[targets.designed == 0].copy()
    natural["selection_hash"] = natural.stem.map(lambda s: hashlib.sha256(f"325:{s}".encode()).hexdigest())
    # Select using labels and length only, never either accuracy or confidence.
    selected = natural.sort_values("selection_hash").groupby("tier", sort=True).head(5).sort_values(["tier", "stem"])
    selected.to_csv(DATA / "confidence_targets.csv", index=False)
    plan = {
        "status": "fixed design; see test_contact_run.json and helico_*_run.json for execution",
        "checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "raw_training_tokens": 248583762834,
        "contacts": {
            "targets": "missing_eval_test_targets.csv", "n_targets": len(missing),
            "n_rollouts_per_target": 100, "inference_complete": "data/test_contact_run.json", "total_rollouts": 100 * len(missing),
            "recipe": {"temperature": 1.0, "top_p": 0.95, "top_k": -1, "token_budget": "6L+128", "fresh_realization": True},
            "worker": "generation/score_rollout_worker.py (exp277 adaptation)",
            "checkpoint_local_compute": "Modal us-east; CoreWeave clusters had zero workers; stage public checkpoint once to volume",
            "checkpoint_uri": "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344",
            "output_uri": "Modal volume marinfold-exp325:/results/exp277-step266344-test-v1/",
            "workers": "8 independent Modal H100 workers; one-target smoke first",
            "preserve": ["dense votes", "ordered per-rollout pairs", "completion/parse status", "timings.csv", "worker and model metadata"],
            "publication": "Consolidate raw artifacts and tables to public HF data/exp325-writeup-analysis/exp277-step266344/v1/",
        },
        "folding": {
            "targets": "Verified intersection of missing_eval_test_targets.csv and Helico prompt-to-token map",
            "method": "Request top-L predicted contacts; Helico applies its standard post-mapping separation filter; all other pairs unknown; no contact-count sweep",
            "model": "contacts-msafree-01-step-6000.pt",
            "model_sha256": "779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644",
            "source_commit": "b10385d736673c81b10e70d1099962af6f2573c0",
            "compute": "Modal H100, existing Helico checkpoint volume; transfer only small contact/target artifacts",
            "samples": 3, "recycles": 6, "seed": 42,
            "selection": "Highest ranking_score among three diffusion samples; never use GDT-TS or lDDT to select",
            "compare": "same targets and definitions as archived Helico exp14 and exp311; record mapping exclusions",
        },
        "confidence": {
            "targets": "confidence_targets.csv", "n_targets": len(selected),
            "selection": "Five natural proteins per MSA tier, SHA256(325:stem) order; no accuracy-based selection",
            "arms": ["oracle full map", "5 uniform randomized full maps", "5 separation-matched randomized full maps"],
            "maps_per_target": 11, "diffusion_samples_per_map": 3,
            "total_structures": len(selected) * 11 * 3,
            "controls": "Same unknown mask, positive count and negative count; sequence length is identical within target. All maps use three diffusion samples and identical seeds. Match the confidence-selection budget.",
            "metrics": ["Within-target oracle-vs-random confidence win rate (ties half)",
                        "Oracle rank among the 11 maps using confidence alone",
                        "GDT-TS / lDDT to verify that randomized maps really produce inferior structures"],
            "uncertainty": "Bootstrap proteins, not maps or diffusion samples",
            "limitation": "Uniform random maps are a weak negative control; success does not establish ranking of plausible alternative folds.",
            "required_runtime_output": "stem,arm,map_seed,sample_idx,ranking_score,mean_plddt,gdt_ts,lddt,n_present,n_absent,n_unknown,elapsed_seconds,model_load_seconds,total_seconds,worker metadata",
        },
        "before_submit": ["Stage one 5.9 GB verified public checkpoint copy; all H100 workers use the same us-east volume",
                          "Adapt and dry-run the existing launchers with these target lists",
                          "Review live capacity and dry-run cost estimate", "Record the publication eval-test read when scored"],
    }
    (DATA / "missing_analysis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Prepared {len(missing)} held-out targets and {len(selected)} confidence-control targets; no jobs submitted.")


if __name__ == "__main__":
    main()
