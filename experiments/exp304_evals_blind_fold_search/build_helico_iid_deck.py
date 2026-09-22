#!/usr/bin/env python
"""Build a detailed two-page report for every fold-switching protein."""

import gzip
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from analyze_helico_iid import PRIMARY_GLOBAL_GDT, PRIMARY_REGION_MARGIN
from build_helico_targets import SOURCE, prepare_structure

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE / "_cache" / "helico_iid"
PREDICTIONS = ROOT / "results" / "predictions" / "iid1000"
DECK = ROOT / "deck"
OUTPUT = HERE / "plots" / "helico_iid_per_protein_deck.pdf"
BLUE, ORANGE, GRAY = "#1676b8", "#df7f1d", "#737373"


def contact_panel(axis, contacts: list, reference: list, title: str,
                  color: str) -> None:
    """Plot one MarinFold contact map over its experimental reference contacts."""
    if len(reference):
        ref = np.asarray(reference)
        axis.scatter(ref[:, 0], ref[:, 1], s=3, color="#d1d4d8", alpha=0.7,
                     rasterized=True, label="reference")
    if len(contacts):
        pred = np.asarray(contacts)
        axis.scatter(pred[:, 0], pred[:, 1], s=2.5, color=color, alpha=0.7,
                     rasterized=True, label="MarinFold")
    axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
    axis.set_xlabel("Residue index")
    axis.set_ylabel("Residue index")
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)


def reference_contact_panel(axis, fold1: list, fold2: list) -> None:
    """Plot the two experimental contact maps in opposing triangles."""
    first, second = np.asarray(fold1), np.asarray(fold2)
    axis.scatter(first[:, 0], first[:, 1], s=3, color=BLUE, alpha=0.65,
                 rasterized=True, label="Fold1")
    axis.scatter(second[:, 1], second[:, 0], s=3, color=ORANGE, alpha=0.65,
                 rasterized=True, label="Fold2 (mirrored)")
    axis.set_title("Experimental contact maps", loc="left", fontsize=10, fontweight="bold")
    axis.set_xlabel("Residue index")
    axis.set_ylabel("Residue index")
    axis.set_aspect("equal", adjustable="box")
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Render structure assets, per-protein pages, and the combined deck."""
    selections = pd.read_csv(DATA / "helico_iid_selected_rollouts.csv")
    summaries = pd.read_csv(DATA / "helico_iid_structural_per_protein.csv").set_index("pair_id")
    scores = pd.read_parquet(ROOT / "scores.parquet")
    inputs = pd.read_parquet(ROOT / "data" / "inputs.parquet").set_index("target_id")
    truth = pd.read_parquet(SOURCE / "eval_targets.parquet").set_index("pair_id")
    proteins = pd.read_csv(DATA / "helico_iid_proteins.csv").set_index("pair_id")
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
                 f"Primary structural hit: common-position GDT-TS ≥ {PRIMARY_GLOBAL_GDT:.2f} "
                 f"and switching-region GDT-TS advantage ≥ {PRIMARY_REGION_MARGIN:.2f}.\n"
                 "Every page reports raw GDT-TS, TM-score, and CA RMSD. All PDBs and score "
                 "tables are retained so this rule and the presentation can be changed.",
                 fontsize=13, va="top")
        fig.text(0.06, 0.20,
                 "Each protein has two pages: contact-map evidence and structural evidence. "
                 "Candidate selection is an oracle analysis against the references.",
                 fontsize=12, color="#444444")
        fig.set_axis_off()
        save_page(pdf, fig, page_dir / "000_intro.png")

        for protein_index, pair_id in enumerate(sorted(summaries.index), 1):
            target = truth.loc[pair_id]
            selected = selections[selections.pair_id == pair_id].set_index("fold")
            chosen1, chosen2 = selected.loc[1], selected.loc[2]
            pair_scores = scores[scores.pair_id == pair_id].set_index("target_id")
            summary = summaries.loc[pair_id]
            ref_paths = {}
            for fold in (1, 2):
                path = reference_dir / f"{pair_id}__fold{fold}.cif.gz"
                if not path.exists():
                    cif, _, _, _ = prepare_structure(str(target[f"fold{fold}"]), str(target.sequence))
                    path.write_bytes(cif)
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
            fig = plt.figure(figsize=(13.33, 7.5))
            grid = fig.add_gridspec(2, 3, left=0.06, right=0.97, bottom=0.09, top=0.82,
                                    hspace=0.36, wspace=0.27)
            fig.suptitle(f"{protein_index:02d}. {pair_id} — contact maps and structural scores",
                         x=0.06, ha="left", fontsize=19, fontweight="bold")
            fig.text(0.06, 0.865,
                     f"L={int(summary.L)} · Fold1={target.fold1} · Fold2={target.fold2} · "
                     f"1,000 individual rollouts · state at 1,000={int(summary.state_1000)} modes",
                     fontsize=10, color="#444444")
            reference_contact_panel(fig.add_subplot(grid[0, 0]), true1, true2)
            contact_panel(fig.add_subplot(grid[0, 1]), input1.contacts, true1,
                          f"Selected Fold1 map · rollout {int(chosen1.rollout) + 1}", BLUE)
            contact_panel(fig.add_subplot(grid[0, 2]), input2.contacts, true2,
                          f"Selected Fold2 map · rollout {int(chosen2.rollout) + 1}", ORANGE)
            axis = fig.add_subplot(grid[1, :2])
            iid = pair_scores[pair_scores.kind == "iid"]
            axis.scatter(iid.gdt_region_global_fit_fold1, iid.gdt_region_global_fit_fold2,
                         c=iid.rollout, cmap="viridis", s=9, alpha=0.5, rasterized=True)
            axis.scatter([pair_scores.loc[chosen1.target_id].gdt_region_global_fit_fold1],
                         [pair_scores.loc[chosen1.target_id].gdt_region_global_fit_fold2],
                         color=BLUE, s=70, marker="*", label="selected Fold1")
            axis.scatter([pair_scores.loc[chosen2.target_id].gdt_region_global_fit_fold1],
                         [pair_scores.loc[chosen2.target_id].gdt_region_global_fit_fold2],
                         color=ORANGE, s=70, marker="*", label="selected Fold2")
            axis.plot([0, 1], [0, 1], color="#999999", linewidth=1)
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Region GDT-TS vs Fold1",
                     ylabel="Region GDT-TS vs Fold2")
            axis.set_title("All 1,000 Helico predictions", loc="left", fontsize=10,
                           fontweight="bold")
            axis.legend(frameon=False, fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)
            text_axis = fig.add_subplot(grid[1, 2])
            text_axis.set_axis_off()
            text_axis.text(0, 1,
                "Primary structural screen\n"
                f"  Fold1 hits: {int(summary.n_fold1_structural_hits)}\n"
                f"  Fold2 hits: {int(summary.n_fold2_structural_hits)}\n"
                f"  First Fold1: {summary.first_fold1_structural_hit}\n"
                f"  First Fold2: {summary.first_fold2_structural_hit}\n"
                f"  True-contact controls: "
                f"{'pass' if summary.true_fold1_control_pass and summary.true_fold2_control_pass else 'FAIL'}\n\n"
                "Selected structures\n"
                f"  Fold1 GDT: {chosen1.gdt_common_target:.3f} vs {chosen1.gdt_common_other:.3f}\n"
                f"  Fold1 region: {chosen1.gdt_region_target:.3f} vs {chosen1.gdt_region_other:.3f}\n"
                f"  Fold1 CA RMSD: {chosen1.rmsd_common_target:.2f} Å\n"
                f"  Fold2 GDT: {chosen2.gdt_common_target:.3f} vs {chosen2.gdt_common_other:.3f}\n"
                f"  Fold2 region: {chosen2.gdt_region_target:.3f} vs {chosen2.gdt_region_other:.3f}\n"
                f"  Fold2 CA RMSD: {chosen2.rmsd_common_target:.2f} Å",
                va="top", fontsize=9.2, linespacing=1.4)
            save_page(pdf, fig, page_dir / f"{protein_index:03d}_{pair_id}_contacts.png")

            fig = plt.figure(figsize=(13.33, 7.5))
            grid = fig.add_gridspec(2, 3, left=0.04, right=0.98, bottom=0.06, top=0.86,
                                    hspace=0.22, wspace=0.08)
            fig.suptitle(f"{protein_index:02d}. {pair_id} — experimental and Helico structures",
                         x=0.05, ha="left", fontsize=19, fontweight="bold")
            add_image(fig.add_subplot(grid[0, 0]), images["references"],
                      "Experimental folds (aligned)")
            add_image(fig.add_subplot(grid[0, 1]), images["pred1"],
                      f"Selected rollout {int(chosen1.rollout) + 1} vs Fold1")
            add_image(fig.add_subplot(grid[0, 2]), images["pred2"],
                      f"Selected rollout {int(chosen2.rollout) + 1} vs Fold2")
            add_image(fig.add_subplot(grid[1, 0]), images["control1"],
                      "True Fold1 contacts → Helico")
            add_image(fig.add_subplot(grid[1, 1]), images["control2"],
                      "True Fold2 contacts → Helico")
            note = fig.add_subplot(grid[1, 2])
            note.set_axis_off()
            control1 = pair_scores.loc[f"{pair_id}__true_fold1"]
            control2 = pair_scores.loc[f"{pair_id}__true_fold2"]
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
                "Blue/orange: prediction or corresponding reference\n"
                "Gray: target experimental reference\n"
                "Cartoons are visual aids; tables use exact common-position scoring.",
                va="top", fontsize=9.5, linespacing=1.45)
            save_page(pdf, fig, page_dir / f"{protein_index:03d}_{pair_id}_structures.png")
            print(f"rendered {pair_id}", flush=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
