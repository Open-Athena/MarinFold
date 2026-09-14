"""Render a reproducible random montage of quality-passing production refolds."""

import argparse
import csv
import random
import tempfile
from pathlib import Path

import biotite.structure as struc
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import pymol
from matplotlib.patches import Patch
from pymol import cmd

from build_summary import save_plot_with_meta


CONDITION_LABELS = {
    "unconditional": "unconditional",
    "1.x.x.x": "alpha",
    "2.x.x.x": "beta",
    "3.x.x.x": "alpha/beta",
}
SECONDARY_COLORS = {
    "helix": (0.86, 0.22, 0.38),
    "sheet": (0.95, 0.67, 0.16),
    "loop": (0.20, 0.62, 0.78),
}


def sample_candidates(path: Path, count: int, seed: int) -> list[dict]:
    """Draw a deterministic simple random sample from quality-passing audit rows."""
    columns = [
        "stem",
        "sequence",
        "plddt",
        "scrmsd",
        "quality_pass",
        "audit_condition",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    eligible = [row for row in rows if row["quality_pass"]]
    if len(eligible) < count:
        raise ValueError(
            f"Requested {count} structures from {len(eligible)} eligible rows"
        )
    return random.Random(seed).sample(eligible, count)


def retention_metadata(path: Path) -> dict[str, dict]:
    """Index the completed audit annotations by stable candidate identifier."""
    with path.open() as handle:
        return {row["stem"]: row for row in csv.DictReader(handle)}


def assign_secondary_structure(selection: str) -> None:
    """Assign PyMOL cartoon states using P-SEA on the available C-alpha trace."""
    model = cmd.get_model(f"{selection} and name CA")
    atoms = struc.AtomArray(len(model.atom))
    atoms.coord = np.asarray([atom.coord for atom in model.atom])
    atoms.res_id = np.asarray([int(atom.resi) for atom in model.atom])
    atoms.res_name = [atom.resn for atom in model.atom]
    atoms.atom_name[:] = "CA"
    atoms.chain_id[:] = "A"
    atoms.element[:] = "C"
    annotation = struc.annotate_sse(atoms)
    cmd.alter(selection, "ss='L'")
    for psea_code, pymol_code in (("a", "H"), ("b", "S")):
        residue_ids = [
            atom.resi
            for atom, assigned in zip(model.atom, annotation, strict=True)
            if assigned == psea_code
        ]
        if residue_ids:
            cmd.alter(
                f"{selection} and resi {'+'.join(residue_ids)}",
                f"ss='{pymol_code}'",
            )
    cmd.rebuild(selection)


def render_structure(pdb_path: Path, output: Path) -> None:
    """Render one refold as an orthographic secondary-structure cartoon."""
    cmd.reinitialize()
    cmd.load(str(pdb_path), "refold")
    cmd.remove("solvent")
    cmd.hide("everything", "all")
    assign_secondary_structure("refold")
    cmd.show("cartoon", "refold")
    for name, rgb in SECONDARY_COLORS.items():
        cmd.set_color(f"secondary_{name}", rgb)
    cmd.color("secondary_loop", "refold")
    cmd.color("secondary_helix", "refold and ss H")
    cmd.color("secondary_sheet", "refold and ss S")
    cmd.set("orthoscopic", 1)
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 1)
    cmd.set("specular", 0.25)
    cmd.bg_color("white")
    cmd.orient("refold")
    cmd.zoom("refold", buffer=4)
    cmd.png(str(output), width=420, height=340, dpi=180, ray=1, quiet=1)


def write_sample(
    path: Path, rows: list[dict], annotations: dict[str, dict], seed: int
) -> None:
    """Write the exact sample and its audit annotations."""
    fields = [
        "panel",
        "seed",
        "stem",
        "length",
        "condition",
        "plddt",
        "self_consistency_rmsd",
        "sequence_excluded",
        "structure_excluded",
        "fine_cluster",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for panel, row in enumerate(rows, start=1):
            audit = annotations[row["stem"]]
            writer.writerow(
                {
                    "panel": panel,
                    "seed": seed,
                    "stem": row["stem"],
                    "length": len(row["sequence"]),
                    "condition": row["audit_condition"],
                    "plddt": row["plddt"],
                    "self_consistency_rmsd": row["scrmsd"],
                    "sequence_excluded": audit["sequence_excluded"],
                    "structure_excluded": audit["structure_excluded"],
                    "fine_cluster": audit["fine_cluster"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("/data/exp278/scale-review-18h/candidates.parquet"),
    )
    parser.add_argument(
        "--retention",
        type=Path,
        default=Path("data/scale-20260909/review-18h/retention.csv"),
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("/data/exp278/scale-review-18h/refolded"),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/scale-20260909/structure-montage-sample.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/quality-refolded-structures-8x8.png"),
    )
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2786401)
    args = parser.parse_args()
    side = round(args.count**0.5)
    if side * side != args.count:
        raise ValueError("Structure count must be a perfect square")

    rows = sample_candidates(args.candidates, args.count, args.seed)
    annotations = retention_metadata(args.retention)
    missing = [
        row["stem"]
        for row in rows
        if not (args.pdb_dir / f"{row['stem']}.pdb").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} sampled PDB files")
    write_sample(args.sample_output, rows, annotations, args.seed)

    pymol.finish_launching(["pymol", "-cq"])
    figure, axes = plt.subplots(side, side, figsize=(16, 17.2))
    with tempfile.TemporaryDirectory(prefix="exp278-montage-") as temp_dir:
        render_dir = Path(temp_dir)
        for panel, (axis, row) in enumerate(zip(axes.flat, rows, strict=True), start=1):
            image_path = render_dir / f"{panel:02d}.png"
            render_structure(args.pdb_dir / f"{row['stem']}.pdb", image_path)
            axis.imshow(plt.imread(image_path))
            axis.set_title(
                f"{panel:02d} | {len(row['sequence'])} aa | "
                f"{CONDITION_LABELS[row['audit_condition']]}\n"
                f"pLDDT {row['plddt']:.1f} | scRMSD {row['scrmsd']:.2f} A",
                fontsize=6.6,
                pad=1,
            )
            axis.axis("off")

    figure.subplots_adjust(
        left=0.01, right=0.99, bottom=0.035, top=0.94, wspace=0.01, hspace=0.10
    )
    figure.suptitle(
        "64 random quality-passing Proteina refolds — cartoons",
        fontsize=18,
        y=0.987,
    )
    figure.text(
        0.5,
        0.962,
        "Simple random sample from 1,838 quality passes in the stratified 18-hour audit pool; each panel is independently oriented and scaled",
        ha="center",
        fontsize=9,
    )
    figure.legend(
        handles=[
            Patch(color=SECONDARY_COLORS["helix"], label="alpha helix"),
            Patch(color=SECONDARY_COLORS["sheet"], label="beta strand"),
            Patch(color=SECONDARY_COLORS["loop"], label="loop / unassigned"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.006),
    )

    save_plot_with_meta(
        figure,
        args.output,
        caption=(
            "Fixed-seed simple random sample of 64 quality-passing ESMFold refolds "
            "from the production audit pool. PyMOL assigns secondary structure with DSS; "
            "cartoons are independently oriented and scaled, with helices in magenta, "
            "beta strands in gold, and loops/unassigned residues in blue. Panel "
            "labels report length, requested conditioning arm, mean pLDDT, and full-chain "
            "C-alpha self-consistency RMSD to the generated backbone."
        ),
        script="uv run --with pymol-open-source==3.2.0a0 python plot_structure_montage.py",
        args=[
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
        ],
        dpi=180,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
