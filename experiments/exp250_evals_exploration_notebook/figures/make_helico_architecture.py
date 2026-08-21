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
    # A unit grid with equal aspect: 1 unit = 0.5 in both ways, so circles are round and a gap
    # means the same thing horizontally and vertically.
    figure, axis = plt.subplots(figsize=(6.5, 2.8), layout="constrained")
    axis.set(xlim=(0, 13), ylim=(0, 5.6))
    axis.set_aspect("equal")
    axis.set_axis_off()

    # --- inputs -----------------------------------------------------------------------------------
    box(axis, 0.1, 4.0, 2.6, 1.1, "sequence,\nreference conformers,\ntoken bonds", fontsize=6.8)
    box(axis, 0.1, 2.35, 2.6, 0.85, "", color=REMOVED, face="#FBF1F1", dashed=True)
    axis.plot([0.45, 2.35], [2.45, 3.1], color=REMOVED, lw=1.3, alpha=0.75, solid_capstyle="round")
    axis.plot([0.45, 2.35], [3.1, 2.45], color=REMOVED, lw=1.3, alpha=0.75, solid_capstyle="round")
    axis.text(1.4, 2.775, "MSA", ha="center", va="center", fontsize=8.5, color=REMOVED, zorder=5,
              bbox=dict(boxstyle="round,pad=0.15", facecolor="#FBF1F1", edgecolor="none"))
    axis.text(1.4, 2.1, "removed — no alignment,\nand the MSA profile columns\nof s_inputs are "
              "zeroed too", ha="center", va="top", fontsize=6.2, color=REMOVED, linespacing=1.35)

    # --- embedder and the two representations --------------------------------------------------------
    box(axis, 3.5, 4.0, 1.9, 1.1, "input\nembedder", fontsize=7.5)
    arrow(axis, (2.7, 4.55), (3.5, 4.55))

    # s and z are the same kind of object — one colour, so the eye does not read a distinction
    # the model does not make.
    box(axis, 6.0, 4.55, 1.6, 0.7, "single  s", fontsize=7.5)
    box(axis, 6.0, 3.5, 1.6, 0.7, "pair  z", fontsize=7.5)
    arrow(axis, (5.4, 4.7), (6.0, 4.9))
    arrow(axis, (5.4, 4.4), (6.0, 3.85))

    # --- the contact pathway, stacked directly under the pair representation --------------------------
    contact_matrix_icon(axis, 6.25, 0.25, 1.1)
    axis.text(5.95, 0.8, "contacts, three-state\npresent / absent / unknown", ha="right",
              va="center", fontsize=6.2, color=ADDED, linespacing=1.35)
    box(axis, 5.8, 1.65, 2.0, 0.65, "linear 3 → 128\n(zero-initialised)", color=ADDED,
        face="#FDF4F4", fontsize=6.4)
    arrow(axis, (6.8, 1.35), (6.8, 1.65), color=ADDED)

    centre = (6.8, 2.75)
    axis.add_patch(plt.Circle(centre, 0.24, facecolor="white", edgecolor=ADDED, lw=1.3, zorder=3))
    axis.text(centre[0], centre[1], "+", ha="center", va="center", fontsize=9.5, color=ADDED,
              zorder=4)
    arrow(axis, (6.8, 2.3), (6.8, 2.51), color=ADDED)
    arrow(axis, (6.8, 2.99), (6.8, 3.5), color=ADDED)

    # --- trunk ------------------------------------------------------------------------------------------
    box(axis, 8.5, 3.4, 2.2, 1.85, "Pairformer\nstack", fontsize=8)
    arrow(axis, (7.6, 4.9), (8.5, 4.75))
    arrow(axis, (7.6, 3.85), (8.5, 3.95))

    # Recycling: z_init — the contact term included — is re-added at the top of every cycle.
    arrow(axis, (8.5, 3.55), (7.04, 2.75), color=ADDED, connection="arc3,rad=0.3", lw=1.0)
    axis.text(8.75, 2.5, "z re-initialised with the\ncontact term every recycle", ha="left",
              va="top", fontsize=6.2, color=ADDED, linespacing=1.35)

    # --- output -------------------------------------------------------------------------------------------
    box(axis, 11.1, 3.9, 1.8, 1.1, "diffusion\nmodule", fontsize=7.5)
    arrow(axis, (10.7, 4.45), (11.1, 4.45))

    figlib.save_figure(figure, "helico_architecture", 300)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
