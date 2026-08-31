# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose the manuscript's multi-panel SVG figures from the panels in ``output/``.

The plot notebooks own the panels; this owns only the arrangement and the letters. Nothing is
recomputed here, so a figure cannot disagree with the panel it is built from — rerun the relevant
``<n>_plot_*`` notebook and rerun this.

Panels are placed by their natural size, scaled to a common width per row, and lettered A, B, C…
in reading order. A panel that does not exist yet — an architecture diagram, say — is drawn as a
dashed placeholder carrying its letter, so the lettering is right before the artwork arrives and
the gap is obvious rather than silent.

    uv run --with svgutils python assemble_figures.py
"""

import argparse
import re
from pathlib import Path

from svgutils import compose, transform

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
FIGURES = HERE / "manuscript"

LETTER_SIZE = 13
LETTER_WEIGHT = "bold"
LETTER_FONT = "DejaVu Sans"
GUTTER = 5          # points between panels
MARGIN = 4          # points around the whole figure

#: figure name -> the panels it is built from, one list per row. A tuple is
#: (panel stem in output/, optional placeholder caption when the file is absent).
LAYOUT = {
    "figure_1": {
        "caption": ("Document format (a), and Top7's deposited structure beside its "
                    "ground-truth and predicted contact maps (b)."),
        # `training_composition` is still written by pair 5 and is not in any manuscript figure;
        # add it to a row here to give it one.
        "rows": [[("document_format", None), ("top7_maps", None)]],
    },
    "figure_2": {
        "caption": "Contact-prediction accuracy on natural monomers (a) and de novo designs (b).",
        "rows": [[("rprecision_natural", None), ("rprecision_designed", None)]],
    },
    "figure_3": {
        "caption": "Helico architecture (a); GDT-TS for natural monomers (b) and designs (c); lDDT for the same (d, e).",
        "rows": [[("helico_architecture", "Helico model architecture\n(supply artwork)")],
                 [("gdt_ts_natural", None), ("gdt_ts_designed", None)],
                 [("lddt_natural", None), ("lddt_designed", None)]],
    },
}


def panel_size(path: Path) -> tuple[float, float]:
    """Width and height of an SVG in points, resolving the unit suffix matplotlib writes."""
    svg = transform.fromfile(str(path))
    return _to_points(svg.width), _to_points(svg.height)


def _to_points(value) -> float:
    match = re.fullmatch(r"([0-9.]+)\s*(pt|px|in|cm|mm)?", str(value))
    if match is None:
        raise ValueError(f"cannot read SVG dimension {value!r}")
    number, unit = float(match.group(1)), (match.group(2) or "px")
    return number * {"pt": 1.0, "px": 0.75, "in": 72.0, "cm": 28.3465, "mm": 2.83465}[unit]


def placeholder(width: float, height: float, text: str, name: str) -> Path:
    """Write a dashed box standing in for artwork that does not exist yet, and return its path.

    A file rather than an in-memory element, so it composes through exactly the same code path as
    a real panel — the alternative is two element types with subtly different transform APIs.
    """
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="1" y="1" width="{width - 2:.1f}" height="{height - 2:.1f}" fill="#FAFAFA" '
        f'stroke="#B0B0B0" stroke-width="1.5" stroke-dasharray="6,4"/>',
    ]
    for offset, line in enumerate(text.split("\n")):
        lines.append(
            f'<text x="{width / 2:.1f}" y="{height / 2 + offset * 15 - 6:.1f}" '
            f'text-anchor="middle" font-family="{LETTER_FONT}" font-size="11" fill="#8A8A8A">'
            f"{line}</text>")
    lines.append("</svg>")
    destination = FIGURES / f"_placeholder_{name}.svg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines))
    return destination


def rasterise(svg: Path, dpi: int) -> Path | None:
    """Write a PNG beside an assembled SVG, for anything that will not take vector input.

    An SVG user unit is a pixel at 96 dpi, so the scale factor is dpi/96 — at the default that
    makes a 6.5 in figure 1,950 px wide.
    """
    try:
        import cairosvg
    except ImportError:
        print("    (no PNG: `pip install cairosvg` to rasterise the assembled figures)")
        return None
    png = svg.with_suffix(".png")
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=dpi / 96)
    return png


def assemble(name: str, specification: dict, width: float, dpi: int = 300) -> Path:
    """Lay the panels out row by row, letter them, and write `manuscript/<name>.svg`."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    elements, letters = [], iter("ABCDEFGHIJ")
    y = MARGIN
    missing = []

    for row in specification["rows"]:
        # A row is a list of panels, or a (panels, width-fraction) pair. Without the fraction a
        # single-panel row is scaled to the full width and dwarfs the rows beside it.
        row, row_fraction = row if isinstance(row, tuple) else (row, 1.0)
        sizes, sources = [], []
        for stem, fallback in row:
            path = OUTPUT / f"{stem}.svg"
            if path.exists():
                sizes.append(panel_size(path))
                sources.append((path, None))
            else:
                if fallback is None:
                    raise FileNotFoundError(
                        f"{path} is missing and has no placeholder caption — run the plot "
                        f"notebook that writes '{stem}' first")
                missing.append(stem)
                size = (460.0, 130.0)
                sizes.append(size)
                sources.append((placeholder(*size, fallback, stem), None))

        # Scale the row to the figure width, keeping every panel's aspect ratio.
        natural = sum(w for w, _ in sizes) + GUTTER * (len(sizes) - 1)
        scale = ((width - 2 * MARGIN) * row_fraction) / natural
        x = MARGIN
        row_height = 0.0
        for (path, fallback), (panel_width, panel_height) in zip(sources, sizes):
            letter = next(letters)
            drawn = compose.SVG(str(path))
            drawn.scale(scale)
            drawn.move(x, y)
            elements.append(drawn)
            elements.append(compose.Text(letter, x - 1, y + LETTER_SIZE, size=LETTER_SIZE,
                                         weight=LETTER_WEIGHT, font=LETTER_FONT))
            x += panel_width * scale + GUTTER * scale
            row_height = max(row_height, panel_height * scale)
        y += row_height + GUTTER

    figure = compose.Figure(width, y + MARGIN - GUTTER, *elements)
    destination = FIGURES / f"{name}.svg"
    figure.save(str(destination))
    rasterise(destination, dpi)
    # svgutils embeds panel content at compose time, so the placeholder files were scaffolding.
    for scaffold in FIGURES.glob("_placeholder_*.svg"):
        scaffold.unlink()
    note = f"  (placeholders: {', '.join(missing)})" if missing else ""
    print(f"wrote {destination.relative_to(HERE)} (+ .png at {dpi} dpi)  "
          f"{width:.0f}x{y + MARGIN - GUTTER:.0f} pt{note}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=468.0,
                        help="figure width in points (468 = 6.5 in, a US-letter text column)")
    parser.add_argument("--only", help="assemble just this figure")
    parser.add_argument("--png-dpi", type=int, default=300,
                        help="resolution of the PNG written beside each SVG")
    arguments = parser.parse_args()

    for name, specification in LAYOUT.items():
        if arguments.only and name != arguments.only:
            continue
        assemble(name, specification, arguments.width, arguments.png_dpi)
        print(f"    {specification['caption']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
