"""Export native figures on white paper, preserving data colors and typography."""

from pathlib import Path

from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

from theme import PAPER


def save_poster_vectors(fig: Figure, directory: Path, name: str) -> None:
    """Restyle an owned figure for white paper and save its PDF/SVG variants.

    Call immediately before closing the figure: this changes its artists in
    place. Only the house paper color is replaced; heatmap values, missing-data
    shading, grid lines, and predictor colors retain their original meanings.
    Paper-colored annotation boxes, schematic boxes, marker outlines, and panel
    separators must follow the canvas color too.
    """
    paper_rgb = np.asarray(to_rgba(PAPER)[:3])
    fig.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    patches = fig.findobj(Patch)
    patches += [text.get_bbox_patch() for text in fig.findobj(Text) if text.get_bbox_patch() is not None]
    for patch in patches:
        for getter, setter in [(patch.get_facecolor, patch.set_facecolor),
                               (patch.get_edgecolor, patch.set_edgecolor)]:
            color = getter()
            if np.allclose(color[:3], paper_rgb):
                setter((1, 1, 1, color[3]))
    for line in fig.findobj(Line2D):
        color = to_rgba(line.get_color())
        if np.allclose(color[:3], paper_rgb):
            line.set_color((1, 1, 1, color[3]))
    for collection in fig.findobj(Collection):
        colors = collection.get_edgecolors().copy()
        if len(colors):
            mask = np.all(np.isclose(colors[:, :3], paper_rgb), axis=1)
            if mask.any():
                colors[mask, :3] = 1
                collection.set_edgecolors(colors)
    directory.mkdir(parents=True, exist_ok=True)
    for extension, metadata in [("pdf", {"CreationDate": None, "ModDate": None}),
                                ("svg", {"Date": None})]:
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight",
                    facecolor="white", edgecolor="white", metadata=metadata)
