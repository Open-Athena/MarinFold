#!/usr/bin/env python
"""Classify structure-level fold coverage from individually folded iid maps."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze import MIN_CONTACTS_FS, MIN_ENRICHMENT, MIN_RECALL
from iid1000_mode_coverage import absence_curve, first_draw

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCORES = HERE / "_cache" / "helico_iid" / "scores.parquet"
N_DRAWS = 1000
PRIMARY_GLOBAL_GDT_FLOOR = 0.35
PRIMARY_CONTROL_FRACTION = 0.90
PRIMARY_REGION_MARGIN = 0.10


def hits(frame: pd.DataFrame, global_gdt: float,
         region_margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Return mutually exclusive Fold1/Fold2 structural-hit masks."""
    margin = frame.gdt_region_global_fit_fold1 - frame.gdt_region_global_fit_fold2
    valid = frame.marinfold_finished.astype(bool)
    hit1 = valid & (frame.gdt_common_fold1 >= global_gdt) & (margin >= region_margin)
    hit2 = valid & (frame.gdt_common_fold2 >= global_gdt) & (-margin >= region_margin)
    return hit1.to_numpy(dtype=bool), hit2.to_numpy(dtype=bool)


def calibrated_hits_at(
    maps: pd.DataFrame,
    controls: pd.DataFrame,
    global_floor: float,
    control_fraction: float,
    region_margin: float,
) -> tuple[np.ndarray, np.ndarray, bool, bool, float, float]:
    """Apply a true-contact-control-calibrated structural screen."""
    indexed = controls.set_index("kind")
    margin = maps.gdt_region_global_fit_fold1 - maps.gdt_region_global_fit_fold2
    valid = maps.marinfold_finished.astype(bool).to_numpy()
    fold_hits: dict[int, np.ndarray] = {}
    control_passes: dict[int, bool] = {}
    thresholds: dict[int, float] = {}
    for fold in (1, 2):
        control = indexed.loc[f"true_fold{fold}"]
        sign = 1 if fold == 1 else -1
        control_margin = sign * (control.gdt_region_global_fit_fold1
                                 - control.gdt_region_global_fit_fold2)
        control_gdt = float(control[f"gdt_common_fold{fold}"])
        control_ok = bool(control_gdt >= global_floor and control_margin >= region_margin)
        threshold = max(global_floor, control_fraction * control_gdt)
        candidate_margin = sign * margin.to_numpy()
        candidate_gdt = maps[f"gdt_common_fold{fold}"].to_numpy()
        fold_hits[fold] = (valid & control_ok & (candidate_gdt >= threshold)
                           & (candidate_margin >= region_margin))
        control_passes[fold] = control_ok
        thresholds[fold] = threshold
    return (fold_hits[1], fold_hits[2], control_passes[1], control_passes[2],
            thresholds[1], thresholds[2])


def calibrated_hits(maps: pd.DataFrame, controls: pd.DataFrame
                    ) -> tuple[np.ndarray, np.ndarray, bool, bool, float, float]:
    """Use the declared post-hoc control-calibrated screen."""
    return calibrated_hits_at(
        maps, controls, PRIMARY_GLOBAL_GDT_FLOOR, PRIMARY_CONTROL_FRACTION,
        PRIMARY_REGION_MARGIN,
    )


def summarize_threshold(frame: pd.DataFrame, global_gdt: float,
                        region_margin: float) -> dict:
    """Count 1,000-draw coverage and true-contact control validity."""
    protein_rows, controls_ok = [], []
    for pair_id, group in frame.groupby("pair_id"):
        iid = group[group.kind == "iid"].sort_values("rollout")
        hit1, hit2 = hits(iid, global_gdt, region_margin)
        controls = group[group.kind.isin(["true_fold1", "true_fold2"])]
        c1, c2 = hits(controls, global_gdt, region_margin)
        kinds = controls.kind.to_numpy()
        control_ok = bool(c1[kinds == "true_fold1"].item()
                          and c2[kinds == "true_fold2"].item())
        controls_ok.append(control_ok)
        protein_rows.append((hit1.any(), hit2.any()))
    states = np.array([int(a) + int(b) for a, b in protein_rows])
    return {
        "global_gdt_min": global_gdt, "region_gdt_margin_min": region_margin,
        "neither": int((states == 0).sum()), "one": int((states == 1).sum()),
        "both": int((states == 2).sum()), "both_controls_pass": int(sum(controls_ok)),
    }


def summarize_calibrated(frame: pd.DataFrame, global_floor: float,
                         control_fraction: float, region_margin: float) -> dict:
    """Count 1,000-draw coverage for one control-calibrated screen."""
    states, controls_ok = [], []
    for _, group in frame.groupby("pair_id"):
        maps = group[group.kind == "iid"].sort_values("rollout")
        controls = group[group.kind.isin(["true_fold1", "true_fold2"])]
        hit1, hit2, control1, control2, _, _ = calibrated_hits_at(
            maps, controls, global_floor, control_fraction, region_margin
        )
        states.append(int(hit1.any()) + int(hit2.any()))
        controls_ok.append(control1 and control2)
    values = np.asarray(states)
    return {
        "global_gdt_floor": global_floor,
        "control_fraction": control_fraction,
        "region_gdt_margin_min": region_margin,
        "neither": int((values == 0).sum()),
        "one": int((values == 1).sum()),
        "both": int((values == 2).sum()),
        "both_controls_pass": int(sum(controls_ok)),
    }


def main() -> None:
    """Create primary coverage curves, sensitivity table, and deck selections."""
    frame = pd.read_parquet(SCORES)
    exact = pd.read_csv(DATA / "helico_iid_proteins.csv").set_index("pair_id").strict_exact
    iid = frame[frame.kind == "iid"]
    if len(iid) != 29_000 or iid.groupby("pair_id").size().ne(1000).any():
        raise ValueError("expected 1,000 individually folded iid maps for each of 29 proteins")
    curve_counts = {name: np.zeros(N_DRAWS + 1, dtype=int)
                    for name in ("neither", "one", "both", "fold1_seen", "fold2_seen")}
    assessable_counts = {name: np.zeros(N_DRAWS + 1, dtype=int)
                         for name in ("neither", "one", "both")}
    exact_counts = {name: np.zeros(N_DRAWS + 1, dtype=int)
                    for name in ("neither", "one", "both")}
    expected = {name: np.zeros(N_DRAWS + 1) for name in ("neither", "one", "both")}
    per_protein, selections = [], []
    bridge = {name: 0 for name in (
        "contact_fold1", "contact_fold2", "structural_fold1", "structural_fold2",
        "both_fold1", "both_fold2", "structural_only_fold1", "structural_only_fold2",
        "no_contact_fold1", "no_contact_fold2",
    )}
    for pair_id, group in frame.groupby("pair_id"):
        maps = group[group.kind == "iid"].sort_values("rollout")
        controls = group[group.kind.isin(["true_fold1", "true_fold2"])]
        hit1, hit2, control1_ok, control2_ok, threshold1, threshold2 = calibrated_hits(
            maps, controls
        )
        no_contact = group[group.kind == "no_contacts"]
        no_contact1, no_contact2, _, _, _, _ = calibrated_hits(no_contact, controls)
        bridge["no_contact_fold1"] += int(no_contact1.item())
        bridge["no_contact_fold2"] += int(no_contact2.item())
        contact_valid = maps.marinfold_finished & (maps.n_pred_fs >= MIN_CONTACTS_FS)
        contact1 = (contact_valid & (maps.contact_recall_fold1_fs >= MIN_RECALL)
                    & (maps.contact_phi_fs >= MIN_ENRICHMENT)).to_numpy(dtype=bool)
        contact2 = (contact_valid & (maps.contact_recall_fold2_fs >= MIN_RECALL)
                    & (-maps.contact_phi_fs >= MIN_ENRICHMENT)).to_numpy(dtype=bool)
        for fold, contact_hit, structural_hit in ((1, contact1, hit1), (2, contact2, hit2)):
            bridge[f"contact_fold{fold}"] += int(contact_hit.sum())
            bridge[f"structural_fold{fold}"] += int(structural_hit.sum())
            bridge[f"both_fold{fold}"] += int((contact_hit & structural_hit).sum())
            bridge[f"structural_only_fold{fold}"] += int((~contact_hit & structural_hit).sum())
        seen1 = np.r_[False, np.maximum.accumulate(hit1)]
        seen2 = np.r_[False, np.maximum.accumulate(hit2)]
        curve_counts["neither"] += (~seen1 & ~seen2).astype(int)
        curve_counts["one"] += (seen1 ^ seen2).astype(int)
        curve_counts["both"] += (seen1 & seen2).astype(int)
        curve_counts["fold1_seen"] += seen1.astype(int)
        curve_counts["fold2_seen"] += seen2.astype(int)
        n1, n2 = int(hit1.sum()), int(hit2.sum())
        no1, no2 = absence_curve(n1, N_DRAWS), absence_curve(n2, N_DRAWS)
        neither = absence_curve(n1 + n2, N_DRAWS)
        expected["neither"] += neither
        expected["both"] += 1 - no1 - no2 + neither
        expected["one"] += no1 + no2 - 2 * neither
        if control1_ok and control2_ok:
            assessable_counts["neither"] += (~seen1 & ~seen2).astype(int)
            assessable_counts["one"] += (seen1 ^ seen2).astype(int)
            assessable_counts["both"] += (seen1 & seen2).astype(int)
        if bool(exact.loc[pair_id]):
            exact_counts["neither"] += (~seen1 & ~seen2).astype(int)
            exact_counts["one"] += (seen1 ^ seen2).astype(int)
            exact_counts["both"] += (seen1 & seen2).astype(int)
        first1, first2 = first_draw(hit1), first_draw(hit2)
        per_protein.append({
            "pair_id": pair_id, "L": int(maps.L.iloc[0]),
            "strict_exact": bool(exact.loc[pair_id]),
            "n_fold1_structural_hits": n1, "n_fold2_structural_hits": n2,
            "n_fold1_contact_hits": int(contact1.sum()),
            "n_fold2_contact_hits": int(contact2.sum()),
            "n_fold1_contact_and_structural_hits": int((contact1 & hit1).sum()),
            "n_fold2_contact_and_structural_hits": int((contact2 & hit2).sum()),
            "first_fold1_structural_hit": first1, "first_fold2_structural_hit": first2,
            "first_dual_structural_hit": max(first1, first2) if first1 and first2 else None,
            "state_100": int(seen1[100]) + int(seen2[100]),
            "state_200": int(seen1[200]) + int(seen2[200]),
            "state_500": int(seen1[500]) + int(seen2[500]),
            "state_1000": int(seen1[-1]) + int(seen2[-1]),
            "true_fold1_control_pass": control1_ok,
            "true_fold2_control_pass": control2_ok,
            "no_contact_fold1_hit": bool(no_contact1.item()),
            "no_contact_fold2_hit": bool(no_contact2.item()),
            "fold1_gdt_threshold": threshold1, "fold2_gdt_threshold": threshold2,
        })
        margin = maps.gdt_region_global_fit_fold1 - maps.gdt_region_global_fit_fold2
        for fold in (1, 2):
            signed = margin if fold == 1 else -margin
            threshold = threshold1 if fold == 1 else threshold2
            rank = maps.assign(_hit=hit1 if fold == 1 else hit2,
                               _margin=signed,
                               _margin_pass=signed >= PRIMARY_REGION_MARGIN,
                               _relative=maps[f"gdt_common_fold{fold}"] / threshold)
            chosen = rank.sort_values(
                ["_hit", "_margin_pass", "_relative", "_margin"], ascending=False
            ).iloc[0]
            selections.append({
                "pair_id": pair_id, "fold": fold, "target_id": chosen.target_id,
                "rollout": int(chosen.rollout), "passes_primary": bool(chosen["_hit"]),
                "true_contact_control_pass": control1_ok if fold == 1 else control2_ok,
                "gdt_threshold": threshold,
                "gdt_fraction_of_threshold": float(chosen["_relative"]),
                "region_gdt_advantage": float(chosen["_margin"]),
                "gdt_common_target": float(chosen[f"gdt_common_fold{fold}"]),
                "gdt_common_other": float(chosen[f"gdt_common_fold{3 - fold}"]),
                "mean_plddt": float(chosen.mean_plddt),
                "gdt_region_target": float(chosen[f"gdt_region_global_fit_fold{fold}"]),
                "gdt_region_other": float(chosen[f"gdt_region_global_fit_fold{3 - fold}"]),
                "rmsd_common_target": float(chosen[f"rmsd_common_fold{fold}"]),
                "contact_recall_fold1_fs": float(chosen.contact_recall_fold1_fs),
                "contact_recall_fold2_fs": float(chosen.contact_recall_fold2_fs),
            })
    curve = pd.DataFrame({"budget": np.arange(N_DRAWS + 1), **curve_counts,
                          **{f"assessable_{name}": values
                             for name, values in assessable_counts.items()},
                          **{f"exact_{name}": values for name, values in exact_counts.items()},
                          **{f"expected_{name}": values for name, values in expected.items()}})
    if not (curve.neither + curve.one + curve.both).eq(29).all():
        raise ValueError("structural categories do not partition 29 proteins")
    if not (curve.exact_neither + curve.exact_one + curve.exact_both).eq(17).all():
        raise ValueError("exact-sequence structural categories do not partition 17 proteins")
    curve.to_csv(DATA / "helico_iid_structural_curve.csv", index=False)
    pd.DataFrame(per_protein).sort_values("pair_id").to_csv(
        DATA / "helico_iid_structural_per_protein.csv", index=False
    )
    pd.DataFrame(selections).sort_values(["pair_id", "fold"]).to_csv(
        DATA / "helico_iid_selected_rollouts.csv", index=False
    )
    sensitivity = [summarize_threshold(frame, absolute, margin) for absolute, margin in
                   itertools.product((0.35, 0.4, 0.5, 0.6), (0.05, 0.10, 0.15))]
    pd.DataFrame(sensitivity).to_csv(DATA / "helico_iid_threshold_sensitivity.csv", index=False)
    calibrated_sensitivity = [
        summarize_calibrated(frame, floor, fraction, PRIMARY_REGION_MARGIN)
        for floor, fraction in itertools.product((0.35, 0.4, 0.5), (0.7, 0.8, 0.9, 1.0))
    ]
    pd.DataFrame(calibrated_sensitivity).to_csv(
        DATA / "helico_iid_control_fraction_sensitivity.csv", index=False
    )
    structural_preference = (iid.gdt_region_global_fit_fold1
                             - iid.gdt_region_global_fit_fold2)
    finite = iid.contact_phi_fs.notna() & structural_preference.notna()
    correlation = spearmanr(iid.loc[finite, "contact_phi_fs"], structural_preference[finite])
    bridge_rows = [{"metric": name, "value": value} for name, value in bridge.items()]
    bridge_rows.extend([
        {"metric": "spearman_contact_vs_structural_preference", "value": correlation.statistic},
        {"metric": "spearman_p_value", "value": correlation.pvalue},
    ])
    pd.DataFrame(bridge_rows).to_csv(DATA / "helico_iid_bridge_summary.csv", index=False)
    screen_protocol = {
        "name": "post_hoc_control_calibrated_structural_screen",
        "status": "exploratory; chosen after inspecting initial controls and the known 3j7w/3j7v positive",
        "global_metric": "Kabsch-aligned GDT-TS on common C-alpha positions",
        "global_gdt_floor": PRIMARY_GLOBAL_GDT_FLOOR,
        "minimum_fraction_of_corresponding_true_contact_control": PRIMARY_CONTROL_FRACTION,
        "region_metric": "Kabsch-aligned GDT-TS on switching-region C-alpha positions after global fit",
        "minimum_signed_region_advantage": PRIMARY_REGION_MARGIN,
        "candidate_requires_finished_marinfold_completion": True,
        "control_gate": "the corresponding true-contact control must itself pass the global floor and region advantage",
        "fixed_threshold_sensitivity_global_gdt": [0.35, 0.4, 0.5, 0.6],
        "fixed_threshold_sensitivity_region_margin": [0.05, 0.10, 0.15],
        "control_fraction_sensitivity": [0.7, 0.8, 0.9, 1.0],
    }
    (DATA / "helico_iid_structural_screen.json").write_text(
        json.dumps(screen_protocol, indent=2) + "\n"
    )
    print(curve[curve.budget.isin([100, 200, 500, 750, 1000])]
          [["budget", "neither", "one", "both"]].to_string(index=False))
    print(pd.DataFrame(sensitivity).to_string(index=False))


if __name__ == "__main__":
    main()
