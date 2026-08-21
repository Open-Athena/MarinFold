# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw Figure 3a: what contact conditioning changes in Helico.

A schematic, but a faithful one — every element is checked against
`Open-Athena/helico` at `src/helico/model/helico.py` and `src/helico/model/features.py`:

* **The MSA is gone, by both routes.** `use_msa=False` skips the MSA module *and* zeroes the
  MSA-derived `profile` / `deletion_mean` columns inside `s_inputs`. Gating the module alone
  would leave alignment-derived conservation in the single representation; helico's own comment
  calls that "exactly the bug this argument exists to prevent".
* **Contacts enter the pair track, not the Pairformer blocks.** A three-state token x token
  matrix (present / absent / unknown) is one-hot encoded and added to the pair representation
  `z_init` through a zero-initialised 3 -> 128 projection. Drawing an arrow into each Pairformer
  block would be wrong: the blocks are untouched, the tensor they read is what changed.
* **It is re-added every recycle.** `z = z_init + ...` at the top of each cycle, so the contact
  signal reaches the template, Pairformer, distogram, diffusion and confidence paths on every
  iteration rather than decaying after the first.
* **Zero-initialised** projection: at step 0 the contact pathway is an exact no-op, which is what
  makes warm-starting from a Protenix checkpoint lossless.

    uv run python make_helico_architecture.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

KEPT = "#3C4A5A"
REMOVED = "#B03A3A"
ADDED = "#C44E52"
MUTED = "#7A8DA6"

CONTACT_COLORS = {"present": "#C44E52", "absent": "#E8E8E8", "unknown": "#BFC7D2"}


def box(axis, x, y, width, height, label, *, color=KEPT, face="white", fontsize=8,
        style="round,pad=0.02,rounding_size=0.02", lw=1.3, dashed=False, alpha=1.0):
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=style, linewidth=lw,
                                  edgecolor=color, facecolor=face, alpha=alpha,
                                  linestyle="--" if dashed else "-"))
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center",
              fontsize=fontsize, color=color, linespacing=1.35)


def arrow(axis, start, end, *, color=KEPT, lw=1.3, style="-|>", connection="arc3,rad=0"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                                   linewidth=lw, color=color, connectionstyle=connection,
                                   shrinkA=1, shrinkB=1))


def contact_matrix_icon(axis, x, y, size, seed=0):
    """A small three-state matrix, in the figure's own contact colours."""
    n = 9
    rng = np.random.default_rng(seed)
    state = np.full((n, n), 1)                      # absent
    for i in range(n):
        state[i, i] = 2                             # unknown band on the diagonal
    for i, j in rng.integers(0, n, size=(7, 2)):
        if abs(i - j) > 1:
            state[i, j] = state[j, i] = 0           # present
    palette = [CONTACT_COLORS["present"], CONTACT_COLORS["absent"], CONTACT_COLORS["unknown"]]
    cell = size / n
    for i in range(n):
        for j in range(n):
            axis.add_patch(plt.Rectangle((x + j * cell, y + (n - 1 - i) * cell), cell, cell,
                                         facecolor=palette[state[i, j]], edgecolor="white",
                                         linewidth=0.3))
    axis.add_patch(plt.Rectangle((x, y), size, size, facecolor="none", edgecolor=KEPT,
                                 linewidth=0.9))


def main() -> int:
    # A unit grid with equal aspect: 1 unit = 0.5 in in both directions, so circles are round and
    # a gap means the same thing horizontally and vertically.
    figure, axis = plt.subplots(figsize=(6.5, 2.6), layout="constrained")
    axis.set(xlim=(0, 13), ylim=(0, 5.2))
    axis.set_aspect("equal")
    axis.set_axis_off()

    # --- inputs ---------------------------------------------------------------------------------
    box(axis, 0.1, 3.5, 2.5, 1.1, "sequence,\nreference conformers,\ntoken bonds", fontsize=6.8)
    box(axis, 0.1, 1.85, 2.5, 0.85, "", color=REMOVED, face="#FBF1F1", dashed=True)
    axis.plot([0.45, 2.25], [1.95, 2.6], color=REMOVED, lw=1.3, alpha=0.75,
              solid_capstyle="round")
    axis.plot([0.45, 2.25], [2.6, 1.95], color=REMOVED, lw=1.3, alpha=0.75,
              solid_capstyle="round")
    axis.text(1.35, 2.275, "MSA", ha="center", va="center", fontsize=8.5, color=REMOVED,
              zorder=5, bbox=dict(boxstyle="round,pad=0.15", facecolor="#FBF1F1",
                                  edgecolor="none"))
    axis.text(1.35, 1.62, "removed — no alignment,\nand the MSA profile columns\nof s_inputs "
              "are zeroed too", ha="center", va="top", fontsize=6.2, color=REMOVED,
              linespacing=1.35)

    # --- embedder and the two representations ------------------------------------------------------
    box(axis, 3.3, 3.5, 1.9, 1.1, "input\nembedder", fontsize=7.5)
    arrow(axis, (2.6, 4.05), (3.3, 4.05))

    box(axis, 5.9, 4.15, 1.7, 0.75, "single  s", fontsize=7.5, color=MUTED)
    box(axis, 5.9, 3.05, 1.7, 0.75, "pair  z", fontsize=7.5)
    arrow(axis, (5.2, 4.25), (5.9, 4.5), color=MUTED)
    arrow(axis, (5.2, 3.85), (5.9, 3.5))

    # --- the contact pathway -------------------------------------------------------------------------
    contact_matrix_icon(axis, 3.35, 0.35, 1.1)
    axis.text(3.9, 0.15, "contacts, three-state\npresent / absent / unknown", ha="center",
              va="top", fontsize=6.2, color=ADDED, linespacing=1.35)
    box(axis, 5.05, 0.55, 2.0, 0.7, "linear 3 → 128\n(zero-initialised)", color=ADDED,
        face="#FDF4F4", fontsize=6.4)
    arrow(axis, (4.5, 0.9), (5.05, 0.9), color=ADDED)

    centre = (6.75, 2.15)
    axis.add_patch(plt.Circle(centre, 0.24, facecolor="white", edgecolor=ADDED, lw=1.3, zorder=3))
    axis.text(centre[0], centre[1], "+", ha="center", va="center", fontsize=9.5, color=ADDED,
              zorder=4)
    arrow(axis, (7.05, 0.9), (6.85, 1.91), color=ADDED, connection="arc3,rad=-0.3")
    arrow(axis, (6.75, 2.39), (6.75, 3.05), color=ADDED)

    # --- trunk ----------------------------------------------------------------------------------------
    box(axis, 8.3, 2.6, 2.3, 2.3,
        "Pairformer\nstack\n\ntriangle updates on z,\nattention on s", fontsize=7)
    arrow(axis, (7.6, 4.5), (8.3, 4.3), color=MUTED)
    arrow(axis, (7.6, 3.4), (8.3, 3.4))

    # Recycling: z_init — contacts included — is re-added at the top of every cycle.
    arrow(axis, (9.45, 2.6), (7.0, 2.05), color=ADDED, connection="arc3,rad=0.34", lw=1.0)
    axis.text(9.25, 1.75, "z re-initialised with the\ncontact term every recycle",
              ha="center", va="top", fontsize=6.2, color=ADDED, linespacing=1.35)

    # --- output -----------------------------------------------------------------------------------------
    box(axis, 11.0, 3.15, 1.85, 1.2, "diffusion\nmodule", fontsize=7.5)
    arrow(axis, (10.6, 3.75), (11.0, 3.75))
    axis.text(11.925, 2.95, "→  3D structure", ha="center", va="top", fontsize=7, color=KEPT)

    figlib.save_figure(figure, "helico_architecture", 300)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
