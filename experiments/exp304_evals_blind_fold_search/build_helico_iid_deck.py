#!/usr/bin/env python
"""Build a detailed two-page report for every fold-switching protein."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from analyze_helico_iid import (
    PRIMARY_CONTROL_FRACTION,
    PRIMARY_GLOBAL_GDT_FLOOR,
    PRIMARY_REGION_MARGIN,
)
from build_helico_targets import SOURCE, prepare_structure

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE / "_cache" / "helico_iid"
PREDICTIONS = ROOT / "results" / "predictions" / "iid1000"
DECK = ROOT / "deck"
OUTPUT = HERE / "plots" / "helico_iid_per_protein_deck.pdf"
BLUE, ORANGE, GRAY = "#1676b8", "#df7f1d", "#737373"


def contact_array(values: list) -> np.ndarray:
    """Normalize Arrow's object array of two-element arrays for plotting."""
    if not len(values):
        return np.empty((0, 2), dtype=int)
    return np.vstack(values).astype(int)


def mark_switching_region(axis, bounds: tuple[int, int]) -> None:
    """Mark switching-region boundaries on both contact-map axes."""
    for boundary in bounds:
        axis.axvline(boundary, color="#555555", linestyle=":", linewidth=0.8)
        axis.axhline(boundary, color="#555555", linestyle=":", linewidth=0.8)


def contact_panel(axis, contacts: list, reference: list, title: str,
                  color: str, bounds: tuple[int, int]) -> None:
    """Plot one MarinFold contact map over its experimental reference contacts."""
    if len(reference):
        ref = contact_array(reference)
        axis.scatter(ref[:, 0], ref[:, 1], s=3, color="#d1d4d8", alpha=0.7,
                     rasterized=True, label="reference")
    if len(contacts):
        pred = contact_array(contacts)
        axis.scatter(pred[:, 0], pred[:, 1], s=2.5, color=color, alpha=0.7,
                     rasterized=True, label="MarinFold")
    axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
    axis.set_xlabel("Residue index")
    axis.set_ylabel("Residue index")
    axis.set_aspect("equal", adjustable="box")
    mark_switching_region(axis, bounds)
    axis.spines[["top", "right"]].set_visible(False)


def reference_contact_panel(axis, fold1: list, fold2: list,
                            bounds: tuple[int, int]) -> None:
    """Plot the two experimental contact maps in opposing triangles."""
    first, second = contact_array(fold1), contact_array(fold2)
    axis.scatter(first[:, 0], first[:, 1], s=3, color=BLUE, alpha=0.65,
                 rasterized=True, label="Fold1")
    axis.scatter(second[:, 1], second[:, 0], s=3, color=ORANGE, alpha=0.65,
                 rasterized=True, label="Fold2 (mirrored)")
    axis.set_title("Experimental contact maps", loc="left", fontsize=10, fontweight="bold")
    axis.set_xlabel("Residue index")
    axis.set_ylabel("Residue index")
    axis.set_aspect("equal", adjustable="box")
    mark_switching_region(axis, bounds)
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)


def render_structure(reference: Path, output: Path, prediction: Path | None,
                     pred_color: str, second_reference: Path | None = None) -> None:
    """Ray-trace a reference overlay or an individual Helico prediction."""
    if output.exists():
        return
    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-cq"])
    cmd.reinitialize()
    cmd.load(str(reference), "reference")
    cmd.hide("everything")
    cmd.show("cartoon", "reference")
    cmd.color("gray70", "reference")
    if second_reference is not None:
        cmd.load(str(second_reference), "second")
        cmd.show("cartoon", "second")
        cmd.color("orange", "second")
        cmd.align("second and name CA", "reference and name CA", cycles=0)
        cmd.color("marine", "reference")
        cmd.set("cartoon_transparency", 0.12, "second")
    if prediction is not None:
        cmd.load(str(prediction), "prediction")
        cmd.show("cartoon", "prediction")
        cmd.color(pred_color, "prediction")
        cmd.align("prediction and name CA", "reference and name CA", cycles=0)
        cmd.set("cartoon_transparency", 0.35, "reference")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.orient()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(output), width=900, height=650, ray=1)


def add_image(axis, path: Path, title: str) -> None:
    axis.imshow(mpimg.imread(path))
    axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
    axis.set_axis_off()


def save_page(pdf: PdfPages, fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    """Render structure assets, per-protein pages, and the combined deck."""
    selections = pd.read_csv(DATA / "helico_iid_selected_rollouts.csv")
    summaries = pd.read_csv(DATA / "helico_iid_structural_per_protein.csv").set_index("pair_id")
    scores = pd.read_parquet(ROOT / "scores.parquet")
    separation = pd.read_csv(DATA / "helico_iid_reference_separation.csv").set_index("pair_id")
    inputs = pd.read_parquet(ROOT / "data" / "inputs.parquet").set_index("target_id")
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    curve = pd.read_csv(DATA / "helico_iid_structural_curve.csv").set_index("budget")
    reference_dir, render_dir = DECK / "references", DECK / "renders"
    page_dir = DECK / "pages"
    reference_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT) as pdf:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.06, 0.88, "Individual-rollout structural validation", fontsize=26,
                 fontweight="bold")
        fig.text(0.06, 0.76,
                 "29 fold-switching proteins × 1,000 iid MarinFold contact maps\n"
                 "Each map was folded independently with Helico; no consensus contacts were used.",
                 fontsize=17, va="top")
        fig.text(0.06, 0.52,
                 f"Exploratory post-hoc structural hit: common-position Kabsch GDT-TS ≥ "
                 f"max({PRIMARY_GLOBAL_GDT_FLOOR:.2f}, "
                 f"{PRIMARY_CONTROL_FRACTION:.0%} of the corresponding true-contact control) and "
                 f"switching-region GDT-TS advantage ≥ {PRIMARY_REGION_MARGIN:.2f}.\n"
                 "Every page reports raw Kabsch GDT-TS and standard CA RMSD. All PDBs and score "
                 "tables are retained so this rule and the presentation can be changed.",
                 fontsize=13, va="top")
        fig.text(0.06, 0.20,
                 "Each protein has two pages: contact-map evidence and structural evidence. "
                 "Candidate selection is an oracle analysis against the references.",
                 fontsize=12, color="#444444")
        save_page(pdf, fig, page_dir / "000_intro.png")

        final = curve.loc[1000]
        n_assessable = int(final.assessable_neither + final.assessable_one
                           + final.assessable_both)
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.suptitle("Aggregate result and interpretation", x=0.05, ha="left",
                     fontsize=23, fontweight="bold")
        image_axis = fig.add_axes((0.04, 0.12, 0.66, 0.76))
        image_axis.imshow(mpimg.imread(HERE / "plots" / "helico_iid_structural_coverage.png"))
        image_axis.set_axis_off()
        text_axis = fig.add_axes((0.72, 0.14, 0.25, 0.70))
        text_axis.set_axis_off()
        text_axis.text(
            0, 1,
            "At 1,000 rollouts\n"
            f"  Both folds: {int(final.both)}/29\n"
            f"  One fold: {int(final.one)}/29\n"
            f"  Neither fold: {int(final.neither)}/29\n\n"
            f"Exact-sequence subset\n"
            f"  Both: {int(final.exact_both)}/17\n"
            f"  One: {int(final.exact_one)}/17\n"
            f"  Neither: {int(final.exact_neither)}/17\n\n"
            "Validator gate\n"
            f"  Both true-contact controls pass: {n_assessable}/29\n"
            f"  Dual among those: {int(final.assessable_both)}/{n_assessable}\n\n"
            "A missing structural mode is evidence about\n"
            "the search only when Helico's true-contact\n"
            "control for that mode succeeds. Candidate\n"
            "selection on later pages is reference-aware\n"
            "and serves as an oracle analysis.",
            va="top", fontsize=11, linespacing=1.45,
        )
        save_page(pdf, fig, page_dir / "001_aggregate.png")

        for protein_index, pair_id in enumerate(sorted(summaries.index), 1):
            target = truth.loc[pair_id]
            selected = selections[selections.pair_id == pair_id].set_index("fold")
            chosen1, chosen2 = selected.loc[1], selected.loc[2]
            status1 = "primary hit" if chosen1.passes_primary else "best miss"
            status2 = "primary hit" if chosen2.passes_primary else "best miss"
            pair_scores = scores[scores.pair_id == pair_id].set_index("target_id")
            summary = summaries.loc[pair_id]
            reference_scores = separation.loc[pair_id]
            ref_paths = {}
            fold1_positions = {}
            for fold in (1, 2):
                path = reference_dir / f"{pair_id}__fold{fold}.cif.gz"
                cif, positions, _, _ = prepare_structure(
                    str(target[f"fold{fold}"]), str(target.sequence)
                )
                if not path.exists():
                    path.write_bytes(cif)
                if fold == 1:
                    fold1_positions = positions
                ref_paths[fold] = path
            images = {
                "references": render_dir / f"{pair_id}__references.png",
                "pred1": render_dir / f"{chosen1.target_id}__vs_fold1.png",
                "pred2": render_dir / f"{chosen2.target_id}__vs_fold2.png",
                "control1": render_dir / f"{pair_id}__control_fold1.png",
                "control2": render_dir / f"{pair_id}__control_fold2.png",
            }
            render_structure(ref_paths[1], images["references"], None, "marine", ref_paths[2])
            render_structure(ref_paths[1], images["pred1"],
                             PREDICTIONS / f"{chosen1.target_id}.pdb.gz", "marine")
            render_structure(ref_paths[2], images["pred2"],
                             PREDICTIONS / f"{chosen2.target_id}.pdb.gz", "orange")
            render_structure(ref_paths[1], images["control1"],
                             PREDICTIONS / f"{pair_id}__true_fold1.pdb.gz", "marine")
            render_structure(ref_paths[2], images["control2"],
                             PREDICTIONS / f"{pair_id}__true_fold2.pdb.gz", "orange")

            input1 = inputs.loc[chosen1.target_id]
            input2 = inputs.loc[chosen2.target_id]
            true1 = inputs.loc[f"{pair_id}__true_fold1"].contacts
            true2 = inputs.loc[f"{pair_id}__true_fold2"].contacts
            region_ranks = [rank for position, rank in fold1_positions.items()
                            if int(target.fs_lo) <= position < int(target.fs_hi)]
            bounds = (min(region_ranks), max(region_ranks) + 1)
            fig = plt.figure(figsize=(13.33, 7.5))
            grid = fig.add_gridspec(2, 3, left=0.06, right=0.97, bottom=0.09, top=0.82,
                                    hspace=0.36, wspace=0.27)
            fig.suptitle(f"{protein_index:02d}. {pair_id} — contact maps and structural scores",
                         x=0.06, ha="left", fontsize=19, fontweight="bold")
            fig.text(0.06, 0.865,
                     f"L={int(summary.L)} · Fold1={target.fold1} · Fold2={target.fold2} · "
                     f"1,000 rollouts · {'exact' if summary.strict_exact else 'near-match'} sequence · "
                     f"state={int(summary.state_1000)} modes · "
                     f"reference TM-score={reference_scores.tm_common:.3f}",
                     fontsize=10, color="#444444")
            reference_contact_panel(fig.add_subplot(grid[0, 0]), true1, true2, bounds)
            contact_panel(fig.add_subplot(grid[0, 1]), input1.contacts, true1,
                          f"Fold1 {status1} · rollout {int(chosen1.rollout) + 1}", BLUE,
                          bounds)
            contact_panel(fig.add_subplot(grid[0, 2]), input2.contacts, true2,
                          f"Fold2 {status2} · rollout {int(chosen2.rollout) + 1}", ORANGE,
                          bounds)
            axis = fig.add_subplot(grid[1, 0])
            iid = pair_scores[pair_scores.kind == "iid"]
            points = axis.scatter(
                iid.tm_common_fold1, iid.tm_common_fold2,
                c=iid.rollout + 1, cmap="viridis", s=9, alpha=0.5, rasterized=True,
            )
            axis.scatter([pair_scores.loc[chosen1.target_id].tm_common_fold1],
                         [pair_scores.loc[chosen1.target_id].tm_common_fold2],
                         color=BLUE, s=70, marker="*", label="GDT-screen Fold1 candidate")
            axis.scatter([pair_scores.loc[chosen2.target_id].tm_common_fold1],
                         [pair_scores.loc[chosen2.target_id].tm_common_fold2],
                         color=ORANGE, s=70, marker="*", label="GDT-screen Fold2 candidate")
            axis.plot([0, 1], [0, 1], color="#999999", linewidth=1)
            axis.set(xlim=(0, 1), ylim=(0, 1),
                     xlabel="Whole-structure TM-score vs Fold1",
                     ylabel="Whole-structure TM-score vs Fold2")
            axis.set_box_aspect(1)
            axis.set_title("All 1,000 predictions · whole-structure TM-score", loc="left",
                           fontsize=9, fontweight="bold")
            axis.legend(frameon=False, fontsize=8)
            fig.colorbar(points, ax=axis, fraction=0.046, pad=0.04, label="Rollout")
            axis.spines[["top", "right"]].set_visible(False)
            text_axis = fig.add_subplot(grid[1, 1:])
            text_axis.set_axis_off()
            text_axis.text(0, 1,
                "Post-hoc structural screen\n"
                f"  Fold1 hits: {int(summary.n_fold1_structural_hits)}\n"
                f"  Fold2 hits: {int(summary.n_fold2_structural_hits)}\n"
                f"  Contact ∩ structure: {int(summary.n_fold1_contact_and_structural_hits)} / "
                f"{int(summary.n_fold2_contact_and_structural_hits)}\n"
                f"  First Fold1: {summary.first_fold1_structural_hit}\n"
                f"  First Fold2: {summary.first_fold2_structural_hit}\n"
                f"  True-contact gates: F1 "
                f"{'pass' if summary.true_fold1_control_pass else 'FAIL'} · F2 "
                f"{'pass' if summary.true_fold2_control_pass else 'FAIL'}\n\n"
                f"  No-contact baseline hit: "
                f"{'Fold1' if summary.no_contact_fold1_hit else ''}"
                f"{'Fold2' if summary.no_contact_fold2_hit else ''}"
                f"{'none' if not summary.no_contact_fold1_hit and not summary.no_contact_fold2_hit else ''}\n\n"
                "GDT-screen-selected structures\n"
                f"  Fold1 contact recall F1/F2: {chosen1.contact_recall_fold1_fs:.3f} / "
                f"{chosen1.contact_recall_fold2_fs:.3f}\n"
                f"  Fold1 TM-score F1/F2: {chosen1.tm_common_fold1:.3f} / "
                f"{chosen1.tm_common_fold2:.3f}\n"
                f"  Fold1 GDT: {chosen1.gdt_common_target:.3f} vs {chosen1.gdt_common_other:.3f} "
                f"(need {chosen1.gdt_threshold:.3f})\n"
                f"  Fold1 region: {chosen1.gdt_region_target:.3f} vs {chosen1.gdt_region_other:.3f}\n"
                f"  Fold1 CA RMSD: {chosen1.rmsd_common_target:.2f} Å\n"
                f"  Fold1 mean pLDDT: {chosen1.mean_plddt:.1f}\n"
                f"  Fold2 contact recall F1/F2: {chosen2.contact_recall_fold1_fs:.3f} / "
                f"{chosen2.contact_recall_fold2_fs:.3f}\n"
                f"  Fold2 TM-score F1/F2: {chosen2.tm_common_fold1:.3f} / "
                f"{chosen2.tm_common_fold2:.3f}\n"
                f"  Fold2 GDT: {chosen2.gdt_common_target:.3f} vs {chosen2.gdt_common_other:.3f} "
                f"(need {chosen2.gdt_threshold:.3f})\n"
                f"  Fold2 region: {chosen2.gdt_region_target:.3f} vs {chosen2.gdt_region_other:.3f}\n"
                f"  Fold2 CA RMSD: {chosen2.rmsd_common_target:.2f} Å\n"
                f"  Fold2 mean pLDDT: {chosen2.mean_plddt:.1f}",
                va="top", fontsize=8.0, linespacing=1.15)
            text_axis.text(
                0.58, 1,
                "Scatter definition\n"
                "TM-align each prediction to each reference\n"
                "using the full set of common Cα positions.\n"
                "The score is normalized by the reference\n"
                "length (tm_norm_chain2). No switching-region\n"
                "restriction is used. Higher is closer; 1 is\n"
                "a perfect match. The stars remain candidates\n"
                "chosen by the separately labeled GDT screen.",
                va="top", fontsize=8.0, linespacing=1.2, color="#444444",
            )
            save_page(pdf, fig, page_dir / f"{protein_index:03d}_{pair_id}_contacts.png")

            fig = plt.figure(figsize=(13.33, 7.5))
            grid = fig.add_gridspec(2, 3, left=0.04, right=0.98, bottom=0.06, top=0.86,
                                    hspace=0.22, wspace=0.08)
            fig.suptitle(f"{protein_index:02d}. {pair_id} — experimental and Helico structures",
                         x=0.05, ha="left", fontsize=19, fontweight="bold")
            add_image(fig.add_subplot(grid[0, 0]), images["references"],
                      "Experimental folds (aligned)")
            add_image(fig.add_subplot(grid[0, 1]), images["pred1"],
                      f"{status1.title()} · rollout {int(chosen1.rollout) + 1} vs Fold1")
            add_image(fig.add_subplot(grid[0, 2]), images["pred2"],
                      f"{status2.title()} · rollout {int(chosen2.rollout) + 1} vs Fold2")
            add_image(fig.add_subplot(grid[1, 0]), images["control1"],
                      "True Fold1 contacts → Helico")
            add_image(fig.add_subplot(grid[1, 1]), images["control2"],
                      "True Fold2 contacts → Helico")
            note = fig.add_subplot(grid[1, 2])
            note.set_axis_off()
            control1 = pair_scores.loc[f"{pair_id}__true_fold1"]
            control2 = pair_scores.loc[f"{pair_id}__true_fold2"]
            no_contacts = pair_scores.loc[f"{pair_id}__no_contacts"]
            note.text(0, 1,
                "Cross-reference metrics\n\n"
                "True Fold1-contact control\n"
                f"  GDT Fold1/Fold2: {control1.gdt_common_fold1:.3f} / {control1.gdt_common_fold2:.3f}\n"
                f"  region: {control1.gdt_region_global_fit_fold1:.3f} / "
                f"{control1.gdt_region_global_fit_fold2:.3f}\n\n"
                "True Fold2-contact control\n"
                f"  GDT Fold1/Fold2: {control2.gdt_common_fold1:.3f} / {control2.gdt_common_fold2:.3f}\n"
                f"  region: {control2.gdt_region_global_fit_fold1:.3f} / "
                f"{control2.gdt_region_global_fit_fold2:.3f}\n\n"
                "No-contact baseline\n"
                f"  GDT Fold1/Fold2: {no_contacts.gdt_common_fold1:.3f} / "
                f"{no_contacts.gdt_common_fold2:.3f}\n"
                f"  region: {no_contacts.gdt_region_global_fit_fold1:.3f} / "
                f"{no_contacts.gdt_region_global_fit_fold2:.3f}\n\n"
                "Blue/orange: prediction or corresponding reference\n"
                "Gray: target experimental reference\n"
                "Cartoons are visual aids; tables use exact common-position scoring.",
                va="top", fontsize=8.6, linespacing=1.3)
            save_page(pdf, fig, page_dir / f"{protein_index:03d}_{pair_id}_structures.png")
            print(f"rendered {pair_id}", flush=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
