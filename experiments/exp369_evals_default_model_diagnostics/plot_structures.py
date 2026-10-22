"""Show original AFDB and later experimental backbones on common sequence positions."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta
from confirm_training_errors import ca_coordinates

HERE = Path(__file__).resolve().parent


def superpose(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Least-squares align matched coordinates with a proper rigid rotation."""
    x, y = mobile - mobile.mean(axis=0), reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return x @ u @ correction @ vt + reference.mean(axis=0)


def main() -> None:
    """Plot common-residue C-alpha traces, breaking lines at missing residues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp369"))
    parser.add_argument("--monomers", action="store_true")
    args = parser.parse_args()
    prefix = "monomer" if args.monomers else "afdb"
    plan = json.loads((HERE / f"data/{prefix}_inference_plan.json").read_text())
    plot_dir = HERE / "plots" if args.monomers else HERE / "plots/superseded_complexes"
    fig = plt.figure(figsize=(14, 5 if args.monomers else 9), layout="constrained")
    for number, target in enumerate(plan["targets"], start=1):
        af = ca_coordinates(args.work / f"AF-{target['uniprot']}-F1-model_v4.cif", "A", target["sequence"])
        pdb = ca_coordinates(args.work / "pdb_chains" / f"{target['pdb']}_{target['chain']}.cif", target["chain"], target["sequence"])
        indices = np.array(sorted(set(af) & set(pdb)))
        a = np.array([af[i] for i in indices])
        p = np.array([pdb[i] for i in indices])
        a = superpose(a, p)
        combined = np.concatenate((a, p))
        center = combined.mean(axis=0)
        _, _, vt = np.linalg.svd(combined - center, full_matrices=False)
        vt[-1] *= np.linalg.det(vt)
        a, p = (a - center) @ vt.T, (p - center) @ vt.T
        ax = fig.add_subplot(1 if args.monomers else 2, 3, number, projection="3d")
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        segments = np.split(np.arange(len(indices)), breaks)
        for coords, color, label in ((a, "#CC6B25", "Original AFDB v4"), (p, "#167CA0", "Later experimental PDB")):
            for index, segment in enumerate(segments):
                ax.plot(*coords[segment].T, color=color, linewidth=1.4, alpha=.9, label=label if index == 0 else None)
        span = np.ptp(np.concatenate((a, p)), axis=0)
        span = np.maximum(span, 1)
        ax.set_box_aspect(span)
        ax.view_init(elev=22, azim=-65)
        ax.set_axis_off()
        name = target["protein_name"] if args.monomers else target["uniprot"]
        ax.set_title(f"{name} / {target['pdb'].upper()} chain {target['chain']}\nTM-score {target['tm_score_common_residues']:.2f} • {len(indices)} common residues", fontsize=10)
        if number == 1:
            ax.legend(loc="lower left", frameon=False, fontsize=8)
    context = "Verified monomers with moderate discrepancies; no catastrophic fold failure" if args.monomers else "Sequence-matched backbone overlays; assembly and alternate-state context matters"
    fig.suptitle("AFDB training structures and later experimental conformations\n" + context, fontsize=14)
    save_plot_with_meta(fig, str(plot_dir / f"{prefix}_pdb_backbones.png"),
                        caption="C-alpha traces are least-squares aligned using identical common sequence positions; gaps are not connected. Plot panels have independent physical scales. TM-scores in titles come from the separate structural alignment in confirm_training_errors.py. Pru p 7 uses deposited NMR model 1 here; every conformer is assessed separately. These overlays do not establish catastrophic fold errors.")
    plt.close(fig)


if __name__ == "__main__":
    main()
