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
    args = parser.parse_args()
    plan = json.loads((HERE / "data/afdb_inference_plan.json").read_text())
    fig = plt.figure(figsize=(14, 9), layout="constrained")
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
        a, p = (a - center) @ vt.T, (p - center) @ vt.T
        ax = fig.add_subplot(2, 3, number, projection="3d")
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
        ax.set_title(f"{target['uniprot']} / {target['pdb'].upper()} {target['chain']}\nTM-score {target['tm_score_common_residues']:.2f} • {len(indices)} common residues", fontsize=10)
        if number == 1:
            ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("AFDB training structures and later experimental conformations\nSequence-matched backbone overlays; assembly and alternate-state context matters", fontsize=14)
    save_plot_with_meta(fig, str(HERE / "plots/afdb_pdb_backbones.png"),
                        caption="C-alpha traces are least-squares aligned using identical common sequence positions; gaps are not connected. Plot panels have independent physical scales. TM-scores in titles come from the separate structural alignment in confirm_training_errors.py. These show geometric disagreement, not proof that every isolated AFDB monomer state is biologically impossible.")
    plt.close(fig)


if __name__ == "__main__":
    main()
