# Figure notebooks

Numbered pairs. `<n>_make_<name>_data.ipynb` produces a dataset; `<n>_plot_<name>.ipynb` draws it.
Same number, same figure. Nothing is recomputed at plot time.

| # | make | plot | needs | draws |
|---|---|---|---|---|
| 1 | [`1_make_top7_heatmap_data`](1_make_top7_heatmap_data.ipynb) | [`1_plot_top7_heatmap`](1_plot_top7_heatmap.ipynb) | **GPU** | `top7_heatmap_mirrored`, `top7_heatmap_side_by_side` |
| 2 | [`2_make_rprecision_data`](2_make_rprecision_data.ipynb) | [`2_plot_rprecision`](2_plot_rprecision.ipynb) | CPU | `rprecision_natural`, `rprecision_designed` |
| 3 | [`3_make_structure_accuracy_data`](3_make_structure_accuracy_data.ipynb) | [`3_plot_structure_accuracy`](3_plot_structure_accuracy.ipynb) | CPU | `structure_accuracy_natural`, `structure_accuracy_designed` |
| 4 | [`4_make_contamination_contrast_data`](4_make_contamination_contrast_data.ipynb) | [`4_plot_contamination_contrast`](4_plot_contamination_contrast.ipynb) | CPU | `contamination_contrast`, `contamination_contrast_scatter` |

Datasets land in `data/<n>_<name>/`, figures in `output/` as a 300 dpi PNG and a vector PDF. No
titles and no panel letters are baked into a figure — captions and lettering belong to the
document the panel goes into.

## The manuscript figures

[`assemble_figures.py`](assemble_figures.py) composes the panels above into the multi-panel
figures a manuscript needs, lettered A, B, C in reading order, and writes them to
[`manuscript/`](manuscript/) as SVG:

| figure | panels |
|---|---|
| [`figure_1.svg`](manuscript/figure_1.svg) | **A** document format · **B** Top7 observed vs predicted contact map |
| [`figure_2.svg`](manuscript/figure_2.svg) | **A** R-precision, natural monomers · **B** R-precision, de novo designs |
| [`figure_3.svg`](manuscript/figure_3.svg) | **A** Helico architecture *(placeholder)* · **B** GDT-TS, natural · **C** GDT-TS, designs |

```bash
uv run --with svgutils python assemble_figures.py           # all three
uv run --with svgutils python assemble_figures.py --only figure_2
```

The plot notebooks own the panels; this owns only the arrangement and the letters. Nothing is
recomputed, so an assembled figure cannot disagree with the panel it was built from — change a
panel, rerun its plot notebook, rerun this.

A panel that does not exist yet is drawn as a dashed placeholder carrying its letter, so the
lettering is settled before the artwork arrives and the gap is visible rather than silent.
`figure_3`'s panel A is one: a model architecture diagram is not something these notebooks can
produce. Drop a `helico_architecture.svg` into `output/` and re-run, and it takes the slot with no
other change.

## Why the split

Because a figure should be able to tell you where its numbers came from. Every `make` notebook
writes a `metadata.json` beside its data recording the checkout and whether it was dirty, the
machine and GPU, package versions, the exact inference recipe, every input with its sha256, and
every output with its sha256. Every `plot` notebook opens with `figlib.describe(...)`, which
prints that back before drawing anything.

So when a figure changes, `diff data/<name>/metadata.json` says whether the model changed, the
recipe changed, the input data changed, or only the plotting did. And a plot notebook run against
a stale dataset says so — it prints the commit the data was made at and warns when the current
checkout has moved.

Some of what that catches is not hypothetical:

- `rope_theta` is recorded for every model. A transformers-5 export states rope in a way our
  pinned transformers 4.x ignores, loading theta 10000 where the model was trained with 500000 —
  silently, no error, materially worse predictions ([#180](https://github.com/Open-Athena/MarinFold/issues/180)).
  A dataset whose metadata says 10000 came from a mis-loaded model, and `describe` flags it.
- Input digests. #245's per-protein table is the source for figures 2 and 4; if it is republished,
  the digest moves and the two figures stop matching each other silently.
- The dirty-checkout flag. A figure made from uncommitted code is not reproducible, and the
  metadata says so rather than leaving it to memory.

Figure 3 draws **GDT-TS and lDDT side by side** for each protein class. They disagree
usefully: GDT-TS superimposes the prediction and asks what fraction of residues land within a set
of cutoffs, so a correct fold with one domain rotated scores badly; lDDT never superimposes
anything, so it credits locally correct geometry when the global arrangement is wrong. Helico from
MarinFold contacts scores 0.48 GDT-TS against 0.61 lDDT on natural monomers — better locally than
globally, which neither metric shows on its own. Both panels share one bar order (`ORDER_BY`) so
they can be read across.

## Running them

```bash
uv venv --python 3.12 && uv pip install -e ../../marinfold[transformers] \
    jupyterlab pandas pyarrow scikit-learn matplotlib
jupyter lab            # then run make, then plot
```

Or headless, in order:

```bash
for nb in [0-9]_make_*.ipynb [0-9]_plot_*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Notebook 1 needs a GPU and a checkpoint (~6 GB, downloaded once from the public bucket). Its
`BACKEND="auto"` uses vLLM at compute capability >= 8.0 and transformers otherwise; `DTYPE="auto"`
is bfloat16 on any GPU, because **float16 does not work with these weights** — they were trained
in bfloat16 and fp16 overflows their residual stream, which surfaces as a CUDA device-side assert
inside sampling. 2, 3 and 4 are CPU-only and read published tables from the public bucket
anonymously, so they run anywhere with a network connection.

## Editing them

`figlib.py` holds what all of them share: provenance capture, the eval universes, #89's metric
implementation, the plotting style and `save_figure`. Change a figure's *appearance* in its plot
notebook; change what a figure is *measuring* in the make notebook, and regenerate. If you change
a make notebook, rerun it before its plot notebook — the plot will otherwise draw the previous
dataset and tell you (quietly, in the metadata header) that it did.
