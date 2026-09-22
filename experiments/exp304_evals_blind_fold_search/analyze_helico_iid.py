#!/usr/bin/env python
"""Classify structure-level fold coverage from individually folded iid maps."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from iid1000_mode_coverage import absence_curve, first_draw

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCORES = HERE / "_cache" / "helico_iid" / "scores.parquet"
N_DRAWS = 1000
PRIMARY_GLOBAL_GDT = 0.50
PRIMARY_REGION_MARGIN = 0.10


def hits(frame: pd.DataFrame, global_gdt: float,
         region_margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Return mutually exclusive Fold1/Fold2 structural-hit masks."""
    margin = frame.gdt_region_global_fit_fold1 - frame.gdt_region_global_fit_fold2
    hit1 = (frame.gdt_common_fold1 >= global_gdt) & (margin >= region_margin)
    hit2 = (frame.gdt_common_fold2 >= global_gdt) & (-margin >= region_margin)
    return hit1.to_numpy(dtype=bool), hit2.to_numpy(dtype=bool)


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


def main() -> None:
    """Create primary coverage curves, sensitivity table, and deck selections."""
    frame = pd.read_parquet(SCORES)
    iid = frame[frame.kind == "iid"]
    if len(iid) != 29_000 or iid.groupby("pair_id").size().ne(1000).any():
        raise ValueError("expected 1,000 individually folded iid maps for each of 29 proteins")
    curve_counts = {name: np.zeros(N_DRAWS + 1, dtype=int)
                    for name in ("neither", "one", "both", "fold1_seen", "fold2_seen")}
    expected = {name: np.zeros(N_DRAWS + 1) for name in ("neither", "one", "both")}
    per_protein, selections = [], []
    for pair_id, group in frame.groupby("pair_id"):
        maps = group[group.kind == "iid"].sort_values("rollout")
        hit1, hit2 = hits(maps, PRIMARY_GLOBAL_GDT, PRIMARY_REGION_MARGIN)
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
        controls = group[group.kind.isin(["true_fold1", "true_fold2"])]
        control1, control2 = hits(controls, PRIMARY_GLOBAL_GDT, PRIMARY_REGION_MARGIN)
        kinds = controls.kind.to_numpy()
        control1_ok = bool(control1[kinds == "true_fold1"].item())
        control2_ok = bool(control2[kinds == "true_fold2"].item())
        first1, first2 = first_draw(hit1), first_draw(hit2)
        per_protein.append({
            "pair_id": pair_id, "L": int(maps.L.iloc[0]),
            "n_fold1_structural_hits": n1, "n_fold2_structural_hits": n2,
            "first_fold1_structural_hit": first1, "first_fold2_structural_hit": first2,
            "first_dual_structural_hit": max(first1, first2) if first1 and first2 else None,
            "state_100": int(seen1[100]) + int(seen2[100]),
            "state_200": int(seen1[200]) + int(seen2[200]),
            "state_500": int(seen1[500]) + int(seen2[500]),
            "state_1000": int(seen1[-1]) + int(seen2[-1]),
            "true_fold1_control_pass": control1_ok,
            "true_fold2_control_pass": control2_ok,
        })
        margin = maps.gdt_region_global_fit_fold1 - maps.gdt_region_global_fit_fold2
        for fold in (1, 2):
            signed = margin if fold == 1 else -margin
            rank = maps.assign(_hit=hit1 if fold == 1 else hit2,
                               _margin=signed,
                               _global=maps[f"gdt_common_fold{fold}"])
            chosen = rank.sort_values(["_hit", "_margin", "_global"], ascending=False).iloc[0]
            selections.append({
                "pair_id": pair_id, "fold": fold, "target_id": chosen.target_id,
                "rollout": int(chosen.rollout), "passes_primary": bool(chosen["_hit"]),
                "gdt_common_target": float(chosen[f"gdt_common_fold{fold}"]),
                "gdt_common_other": float(chosen[f"gdt_common_fold{3 - fold}"]),
                "gdt_region_target": float(chosen[f"gdt_region_global_fit_fold{fold}"]),
                "gdt_region_other": float(chosen[f"gdt_region_global_fit_fold{3 - fold}"]),
                "rmsd_common_target": float(chosen[f"rmsd_common_fold{fold}"]),
                "tm_common_target": float(chosen[f"tm_common_fold{fold}"]),
                "contact_recall_fold1_fs": float(chosen.contact_recall_fold1_fs),
                "contact_recall_fold2_fs": float(chosen.contact_recall_fold2_fs),
            })
    curve = pd.DataFrame({"budget": np.arange(N_DRAWS + 1), **curve_counts,
                          **{f"expected_{name}": values for name, values in expected.items()}})
    if not (curve.neither + curve.one + curve.both).eq(29).all():
        raise ValueError("structural categories do not partition 29 proteins")
    curve.to_csv(DATA / "helico_iid_structural_curve.csv", index=False)
    pd.DataFrame(per_protein).sort_values("pair_id").to_csv(
        DATA / "helico_iid_structural_per_protein.csv", index=False
    )
    pd.DataFrame(selections).sort_values(["pair_id", "fold"]).to_csv(
        DATA / "helico_iid_selected_rollouts.csv", index=False
    )
    sensitivity = [summarize_threshold(frame, absolute, margin) for absolute, margin in
                   itertools.product((0.4, 0.5, 0.6), (0.05, 0.10, 0.15))]
    pd.DataFrame(sensitivity).to_csv(DATA / "helico_iid_threshold_sensitivity.csv", index=False)
    print(curve[curve.budget.isin([100, 200, 500, 750, 1000])]
          [["budget", "neither", "one", "both"]].to_string(index=False))
    print(pd.DataFrame(sensitivity).to_string(index=False))


if __name__ == "__main__":
    main()
