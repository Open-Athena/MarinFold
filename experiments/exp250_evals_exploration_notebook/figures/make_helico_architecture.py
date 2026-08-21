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

The three lines this figure is a picture of, pinned to `helico` main at dd1b0d4de621:

* the projection — `model/helico.py` L70-72
  https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/helico.py#L70-L72
* the injection into the pair representation — `model/helico.py` L130-136
  https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/helico.py#L130-L136
* the one-hot itself — `model/features.py` L98-110
  https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/features.py#L98-L110
* the MSA gate, zeroing profile and deletion_mean — `model/features.py` L119-145
  https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/features.py#L119-L145

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
    figure, axis = plt.subplots(figsize=(6.5, 2.03), layout="constrained")
    axis.set(xlim=(0, 13), ylim=(0.35, 4.4))
    axis.set_aspect("equal")
    axis.set_axis_off()

    # --- inputs, with the MSA struck out among them -------------------------------------------------
    # The MSA is listed where it used to be consumed rather than shown as a severed box: what
    # changed is one item of the model's input, not a module hanging off the side.
    box(axis, 0.1, 2.35, 2.9, 1.85, "")
    inputs = [("sequence", False), ("reference conformers", False), ("token bonds", False),
              ("MSA", True)]
    for row, (text, struck) in enumerate(inputs):
        y = 3.9 - row * 0.42
        colour = REMOVED if struck else KEPT
        axis.text(1.55, y, text, ha="center", va="center", fontsize=7, color=colour)
        if struck:
            axis.plot([1.28, 1.82], [y, y], color=REMOVED, lw=1.1, solid_capstyle="round")

    # --- embedder and the two representations ----------------------------------------------------------
    box(axis, 3.7, 2.85, 1.8, 1.0, "input\nembedder", fontsize=7.5)
    arrow(axis, (3.0, 3.35), (3.7, 3.35))

    box(axis, 6.1, 3.45, 1.5, 0.65, "single  s", fontsize=7.5)
    box(axis, 6.1, 2.45, 1.5, 0.65, "pair  z", fontsize=7.5)
    arrow(axis, (5.5, 3.5), (6.1, 3.775))
    arrow(axis, (5.5, 3.2), (6.1, 2.775))

    # --- the contact pathway, tucked under the pair representation ---------------------------------------
    # The contact row sits low enough that the arrow into the sum is a real arrow rather than a
    # nub, and the icon, the arrow and the projection box share one centre line so that arrow is
    # exactly horizontal. The projection is centred on the sum (and so on `pair z`) above it.
    CONTACT_Y = 1.05
    contact_matrix_icon(axis, 4.3, CONTACT_Y - 0.475, 0.95)
    axis.text(4.15, CONTACT_Y, "contacts, three-state\npresent / absent / unknown", ha="right",
              va="center", fontsize=6.2, color=ADDED, linespacing=1.35)
    box(axis, 5.85, CONTACT_Y - 0.3, 2.0, 0.6, "linear 3 → 128\n(zero-initialised)", color=ADDED,
        face="#FDF4F4", fontsize=6.4)
    arrow(axis, (5.25, CONTACT_Y), (5.85, CONTACT_Y), color=ADDED)

    centre = (6.85, 2.0)
    axis.add_patch(plt.Circle(centre, 0.22, facecolor="white", edgecolor=ADDED, lw=1.3, zorder=3))
    axis.text(centre[0], centre[1], "+", ha="center", va="center", fontsize=9, color=ADDED,
              zorder=4)
    arrow(axis, (6.85, CONTACT_Y + 0.3), (6.85, 1.78), color=ADDED)
    arrow(axis, (6.85, 2.22), (6.85, 2.45), color=ADDED)

    # --- trunk and output ----------------------------------------------------------------------------------
    box(axis, 8.4, 2.35, 2.1, 1.75, "Pairformer\nstack", fontsize=8)
    arrow(axis, (7.6, 3.775), (8.4, 3.7))
    arrow(axis, (7.6, 2.775), (8.4, 2.85))

    box(axis, 11.0, 2.85, 1.8, 1.0, "diffusion\nmodule", fontsize=7.5)
    arrow(axis, (10.5, 3.35), (11.0, 3.35))

    figlib.save_figure(figure, "helico_architecture", 300, tight=False)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
