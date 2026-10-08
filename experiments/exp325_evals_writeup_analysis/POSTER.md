# Poster-ready plot collection

[Browse individual public poster files](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/poster-2026-10-08).
[Download the complete ZIP](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-2026-10-08/poster_plots.zip), or browse
`poster_plots/pdf`, `poster_plots/svg`, and `poster_plots/png` for individual files.
The complete collection includes every static plot and metric variant, including
all individual-protein pages, new P@L/5 figures, focused KNN comparisons and
the four sparse-oracle GDT-TS/lDDT figures for the five proteins at MSA depth <10
(38 static figures total). Figures 02 and 05 now use the
[L/2 oracle-contact protocol](ORACLE_L2.md) across all four MSA-depth tiers.
Every poster PDF, SVG and PNG has a **plain white background**, including the
plot areas, annotation boxes and schematic boxes.

- **PDF:** native vector artwork with embedded fonts; recommended for print layouts.
- **SVG:** editable vector artwork; Lato font files and their license are included.
- **PNG:** 7,200 pixels wide, tagged 300 dpi. Print at up to **24 inches / 61 cm wide
  at 300 dpi** (or 48 inches at 150 dpi). Height follows each plot's aspect ratio.

The PNGs are rendered from native vector PDFs, not enlarged web images.
Vector files can scale beyond those raster dimensions. Preserve aspect ratio.
The slide deck remains a separate review document; use individual vectors in
your poster layout.

`data/poster_manifest.json` lists every figure, exact raster dimensions, physical
size, white background, blog source PDF and poster output checksums, and its original generating script,
arguments and caption. Numerical lineage remains in [FIGURES.md](FIGURES.md)
and [PL5_ANALYSIS.md](PL5_ANALYSIS.md).

Regenerate and upload the print package directly from the prepared tables:

```bash
uv run python export_poster.py --width-px 7200 --print-dpi 300 --upload
```

The exporter calls `render.py --poster-dir scratch/poster_vectors`, which uses
`poster_style.py` to replace the house paper color with white in native figure
artists before PDF/SVG export. Data colors and heatmap scales are preserved.
This only renders cached analysis; it does not rerun preprocessing or inference.
The blog plots retain their house style; rebuild those with `render.py` and
`build_summary.py` when needed.

The exporter requires `pdftocairo` (Poppler). For wider raster panels at the same
resolution, increase `--width-px`; for example, 10800 gives 36 inches at 300 dpi.
Large print files are saved under `scratch/poster_public/` and published to the
public project bucket; the source code and manifest remain on this branch.
