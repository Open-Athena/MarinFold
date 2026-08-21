# Figure notebooks

Numbered pairs. `<n>_make_<name>_data.ipynb` produces a dataset; `<n>_plot_<name>.ipynb` draws it.
Same number, same figure. Nothing is recomputed at plot time.

| # | make | plot | needs | draws |
|---|---|---|---|---|
| 1 | [`1_make_top7_heatmap_data`](1_make_top7_heatmap_data.ipynb) | [`1_plot_top7_heatmap`](1_plot_top7_heatmap.ipynb) | **GPU** | `top7_heatmap_mirrored`, `top7_heatmap_side_by_side` |
| 2 | [`2_make_rprecision_data`](2_make_rprecision_data.ipynb) | [`2_plot_rprecision`](2_plot_rprecision.ipynb) | CPU | `rprecision_natural`, `rprecision_designed` |
| 3 | [`3_make_gdt_ts_data`](3_make_gdt_ts_data.ipynb) | [`3_plot_gdt_ts`](3_plot_gdt_ts.ipynb) | CPU | `gdt_ts_natural`, `gdt_ts_designed` |
| 4 | [`4_make_contamination_contrast_data`](4_make_contamination_contrast_data.ipynb) | [`4_plot_contamination_contrast`](4_plot_contamination_contrast.ipynb) | CPU | `contamination_contrast`, `contamination_contrast_scatter` |

Datasets land in `data/<n>_<name>/`, figures in `output/` as a 300 dpi PNG and a vector PDF. No
titles and no panel letters are baked into a figure — captions and lettering belong to the
document the panel goes into.

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
