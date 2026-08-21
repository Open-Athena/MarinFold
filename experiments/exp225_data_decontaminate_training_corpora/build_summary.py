# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build ``plots/summary.pdf`` — this experiment's living presentation.

Run from the experiment dir::

    uv run python build_summary.py    # if the experiment has a pyproject
    python build_summary.py           # otherwise (pure-analysis dirs)

Contract (see ``experiments/AGENTS.md`` for the full story):

- Narrative section first: sourced from ``summary_narrative.md`` —
  one ``## `` heading per slide, body text below it. Edit this as
  the experiment progresses; it reflects what we're doing, why, and
  the current state of the results.
- Plot appendix last: auto-discovered from ``plots/*.{png,jpg,jpeg}``.
  Each plot's caption + generating script + args come from a
  sidecar ``plots/<plot>.<ext>.meta.json`` written by the plotting
  script via :func:`save_plot_with_meta` below.
- Plots without a sidecar still appear, with a placeholder caption
  nudging you to wire ``save_plot_with_meta`` in.

Regeneration is meant to be fast: no analysis is rerun, only PNGs
and text are assembled. Build time scales linearly with slide count.
"""

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Helper for plotting scripts to call when they save a plot.
# Importable without pulling in matplotlib (those imports are lazy in main()).
# ---------------------------------------------------------------------------

def save_plot_with_meta(
    fig,
    path: str | Path,
    *,
    caption: str,
    script: str | None = None,
    args: Sequence[str] | None = None,
    **savefig_kwargs,
) -> Path:
    """Save ``fig`` to ``path`` plus a ``<path>.meta.json`` sidecar.

    The sidecar carries the generating script's name, its argv, and a
    human-readable caption. ``build_summary.py`` reads these when
    assembling the slide appendix so the PDF can print, in small text
    on each plot page, exactly how to rerun the script.

    Defaults: ``script`` = ``sys.argv[0]``, ``args`` = ``sys.argv[1:]``.
    Any extra kwargs (``dpi``, ``transparent``, ...) pass through to
    ``fig.savefig``. ``bbox_inches="tight"`` is the default unless you
    override it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **savefig_kwargs)

    script = script if script is not None else Path(sys.argv[0]).name
    args = list(args) if args is not None else list(sys.argv[1:])

    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(
        {"script": script, "args": args, "caption": caption},
        indent=2,
    ))
    return path


# ---------------------------------------------------------------------------
# Module-level config.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE / "plots"
NARRATIVE_PATH = HERE / "summary_narrative.md"
OUTPUT_PATH = PLOTS_DIR / "summary.pdf"

# Letter landscape, in inches.
SLIDE_W, SLIDE_H = 13.33, 7.5

PLOT_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Narrative parsing.
# ---------------------------------------------------------------------------

@dataclass
class Slide:
    title: str
    body: str


def parse_narrative(path: Path) -> list[Slide]:
    if not path.exists():
        return [Slide(
            title="(narrative missing)",
            body=(
                f"Create `{path.name}` in the experiment dir.\n\n"
                "One `## ` heading per slide. Keep it updated as the experiment progresses."
            ),
        )]
    text = path.read_text()
    slides: list[Slide] = []
    current_title: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            if current_title is not None:
                slides.append(Slide(title=current_title, body="\n".join(buf).strip()))
            current_title = m.group(1).strip()
            buf = []
        elif current_title is not None:
            buf.append(line)
    if current_title is not None:
        slides.append(Slide(title=current_title, body="\n".join(buf).strip()))
    if not slides:
        return [Slide(
            title="(empty narrative)",
            body=f"Add `## ` headings to `{path.name}`.",
        )]
    return slides


# ---------------------------------------------------------------------------
# Plot discovery.
# ---------------------------------------------------------------------------

@dataclass
class PlotEntry:
    path: Path
    caption: str
    script: str
    args: list[str] = field(default_factory=list)

    @property
    def invocation(self) -> str:
        if not self.args:
            return self.script
        return self.script + " " + " ".join(self.args)


def load_plots(plots_dir: Path) -> list[PlotEntry]:
    if not plots_dir.exists():
        return []
    paths: list[Path] = [
        p for p in plots_dir.iterdir()
        if p.suffix.lower() in PLOT_EXTENSIONS
    ]
    paths.sort(key=lambda p: p.name)

    entries: list[PlotEntry] = []
    for img in paths:
        meta_path = img.with_suffix(img.suffix + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            caption = str(meta.get("caption", ""))
            script = str(meta.get("script", "?"))
            args = [str(a) for a in meta.get("args", [])]
        else:
            caption = (
                "(metadata missing — call `save_plot_with_meta(...)` in "
                "the generating script; see build_summary.py)"
            )
            script = "?"
            args = []
        entries.append(PlotEntry(path=img, caption=caption, script=script, args=args))
    return entries


# ---------------------------------------------------------------------------
# PDF rendering. matplotlib is imported lazily inside main() so that
# importers of `save_plot_with_meta` don't pay the matplotlib startup cost.
# ---------------------------------------------------------------------------

def _new_slide(plt):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor("white")
    return fig


def render_text_slide(plt, slide: Slide):
    fig = _new_slide(plt)
    fig.text(0.05, 0.92, slide.title, fontsize=26, fontweight="bold", va="top")

    paragraphs = [p.strip() for p in (slide.body or "").split("\n\n") if p.strip()]
    wrapped = "\n\n".join(textwrap.fill(p, width=90) for p in paragraphs)
    fig.text(0.05, 0.82, wrapped, fontsize=14, va="top")
    return fig


def render_plot_slide(plt, mpimg, entry: PlotEntry):
    fig = _new_slide(plt)
    fig.text(0.05, 0.94, entry.path.name, fontsize=18, fontweight="bold", va="top")
    if entry.caption:
        fig.text(
            0.05, 0.88,
            textwrap.fill(entry.caption, width=130),
            fontsize=11, va="top",
        )

    ax = fig.add_axes([0.06, 0.10, 0.88, 0.74])
    ax.set_axis_off()
    img = mpimg.imread(entry.path)
    ax.imshow(img)

    fig.text(
        0.05, 0.04,
        f"$ {entry.invocation}",
        fontsize=8, family="monospace", color="#555",
    )
    return fig


def main() -> int:
    # Lazy imports so `from build_summary import save_plot_with_meta`
    # stays cheap for plotting scripts.
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    narrative_slides = parse_narrative(NARRATIVE_PATH)
    plot_entries = load_plots(PLOTS_DIR)

    PLOTS_DIR.mkdir(exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        for slide in narrative_slides:
            fig = render_text_slide(plt, slide)
            pdf.savefig(fig)
            plt.close(fig)
        for entry in plot_entries:
            fig = render_plot_slide(plt, mpimg, entry)
            pdf.savefig(fig)
            plt.close(fig)

    try:
        rel = OUTPUT_PATH.relative_to(HERE.parent)
    except ValueError:
        rel = OUTPUT_PATH
    print(f"Wrote {rel}")
    print(f"  Narrative slides: {len(narrative_slides)}")
    print(f"  Plot slides:      {len(plot_entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
