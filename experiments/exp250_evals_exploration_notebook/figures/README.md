# Figure notebooks

Numbered pairs. `<n>_make_<name>_data.ipynb` produces a dataset; `<n>_plot_<name>.ipynb` draws it.
Same number, same figure. Nothing is recomputed at plot time.

| # | make | plot | needs | draws |
|---|---|---|---|---|
| 1 | [`1_make_top7_heatmap_data`](1_make_top7_heatmap_data.ipynb) | [`1_plot_top7_heatmap`](1_plot_top7_heatmap.ipynb) | **GPU** | `top7_heatmap_mirrored`, `top7_heatmap_side_by_side` |
| 2 | [`2_make_rprecision_data`](2_make_rprecision_data.ipynb) | [`2_plot_rprecision`](2_plot_rprecision.ipynb) | CPU | `rprecision_natural`, `rprecision_designed` |
| 3 | [`3_make_structure_accuracy_data`](3_make_structure_accuracy_data.ipynb) | [`3_plot_structure_accuracy`](3_plot_structure_accuracy.ipynb) | CPU | `structure_accuracy_natural`, `structure_accuracy_designed` |
| 4 | [`4_make_contamination_contrast_data`](4_make_contamination_contrast_data.ipynb) | [`4_plot_contamination_contrast`](4_plot_contamination_contrast.ipynb) | CPU | `contamination_contrast`, `contamination_contrast_scatter` |
| 5 | [`5_make_training_composition_data`](5_make_training_composition_data.ipynb) | [`5_plot_training_composition`](5_plot_training_composition.ipynb) | CPU | `training_composition` |

Datasets land in `data/<n>_<name>/`, figures in `output/` as a 300 dpi PNG and a vector PDF. No
titles and no panel letters are baked into a figure — captions and lettering belong to the
document the panel goes into.

## The manuscript figures

[`assemble_figures.py`](assemble_figures.py) composes the panels above into the multi-panel
figures a manuscript needs, lettered A, B, C in reading order, and writes them to
[`manuscript/`](manuscript/) as SVG **and PNG** (300 dpi by default, `--png-dpi` to change):

| figure | panels |
|---|---|
| [`figure_1.svg`](manuscript/figure_1.svg) | **A** document format, with the structure it describes · **B** training corpus by source · **C** Top7 ground truth · **D** Top7 predicted |
| [`figure_2.svg`](manuscript/figure_2.svg) | **A** R-precision, natural monomers · **B** R-precision, de novo designs |
| [`figure_3.svg`](manuscript/figure_3.svg) | **A** Helico architecture · **B** GDT-TS, natural · **C** GDT-TS, designs · **D** lDDT, natural · **E** lDDT, designs — the MarinFold arm is helico exp14's `mf_L_363k`, re-run on step-363000 contacts |

```bash
uv run --with svgutils python assemble_figures.py           # all three
uv run --with svgutils python assemble_figures.py --only figure_2
```

**`MarinFold` means one checkpoint everywhere: #232's `m2-p06` at step 363,000**
(`contacts-v1-exp232-m2-p06-train-1.5B`), the best model trained on FoldBench-decontaminated data.
Figure 1 folds Top7 with it, figure 2's bar is its R-precision, and figure 3's Helico arm
conditions on its contacts.

**The contaminated checkpoint is not drawn in any manuscript figure.** #199's cooldown scores
higher than everything we train on decontaminated data, but its corpus was never filtered against
FoldBench, so it is not a claim these figures make. It remains in the datasets — #245 published
those numbers and part 4 of the exploration notebook is built on the contrast — it simply is not
drawn.

**Where figure 2's MarinFold numbers come from.** #245 published every baseline and the *sweep*
checkpoint; #232 scored step-363000 on `eval-val` and `eval-denovo` but deliberately not on
`eval-test`. [`score_foldbench_rollouts.py`](../score_foldbench_rollouts.py) therefore scored all
333 monomers with it here, and reran the sweep checkpoint through the same pipeline as a control:
0.5240 against #245's published 0.5198 on the same 97 proteins (r = 0.995). The make notebook
recomputes that control every time and stores it in the dataset's `metadata.json`.

The plot notebooks own the panels; this owns only the arrangement and the letters. Nothing is
recomputed, so an assembled figure cannot disagree with the panel it was built from — change a
panel, rerun its plot notebook, rerun this.

`figure_3`'s panel A is drawn by [`make_helico_architecture.py`](make_helico_architecture.py),
checked line by line against `Open-Athena/helico`'s `model/helico.py` and `model/features.py`. Two
things it is careful about, because the intuitive version of each is wrong:

- **The MSA is removed by two routes, not one.** `use_msa=False` skips the MSA module *and* zeroes
  the MSA-derived profile / deletion-mean columns inside `s_inputs`. Gating the module alone would
  leave alignment-derived conservation in the single representation — helico's own comment calls
  that "exactly the bug this argument exists to prevent".
- **Contacts enter the pair representation, not the Pairformer blocks.** The three-state matrix is
  one-hot encoded and added to `z_init` through a zero-initialised 3 → 128 projection. The blocks
  are untouched; what changed is the tensor they read. `z_init` is re-added at the top of every
  recycle, which is why the figure draws that arrow rather than a single injection.

Verify the architecture panel against the source, pinned to `helico` main at `dd1b0d4de621`:
[the 3 → 128 projection](https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/helico.py#L70-L72),
[its injection into `z_init`](https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/helico.py#L130-L136),
[the one-hot of the three states](https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/features.py#L98-L110),
and [the MSA gate](https://github.com/Open-Athena/helico/blob/dd1b0d4de621267e4dee40cfbd014042555456d3/src/helico/model/features.py#L119-L145)
that zeroes the profile and deletion-mean columns.

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

Figure 3 reports **GDT-TS and lDDT in separate panels**, and they are *not* on a common scale —
do not subtract one from the other. GDT-TS superimposes the prediction and asks what fraction of
residues land within a set of cutoffs; lDDT never superimposes anything, comparing local
interatomic distances instead, so it credits locally correct geometry regardless of the global
arrangement and sits higher for the same structure. The `Helico, no contacts` arm shows the size
of that offset: **0.15 GDT-TS and 0.36 lDDT on identical predictions**.

Read each within itself. Normalised between its own metric's no-contact floor and oracle ceiling,
MarinFold contacts reach **45 %** on GDT-TS and **51 %** on lDDT — substantially the same story,
which a raw 0.48-against-0.61 comparison would misread as a 0.13 gap.

Every panel draws the arms in the same fixed order (the `ARMS` dict), not sorted by value, so a
bar can be compared with its neighbour across panels rather than the reader re-learning the axis
in each one.

## Running them

```bash
uv venv --python 3.12 && uv pip install -e ../../../marinfold[transformers] \
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
