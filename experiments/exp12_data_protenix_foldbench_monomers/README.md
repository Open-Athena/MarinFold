---
marinfold_experiment:
  issue: 12
  title: "exp: collect protenix results on subset of foldbench monomer set using modal"
  kind: data
  branch: exp/12-protenix-foldbench
---

# exp: collect protenix results on subset of foldbench monomer set using modal

**Issue:** [#12](https://github.com/Open-Athena/MarinFold/issues/12) · **Kind:** `data` · **Branch:** `exp/12-protenix-foldbench`

## Question

Collect distogram and structure (mmCIF) output for [Protenix v2](https://github.com/bytedance/Protenix),
run separately in single-sequence mode and with MSAs, on a subset of the
[FoldBench](https://github.com/BEAM-Labs/FoldBench) protein-monomer set.
Compute per-protein MAE (against expected distances from the distogram)
and dRMSD (predicted vs GT CA-CA pairwise distances), and produce a
comparison plot of single-seq vs MSA mode.

## Hypothesis

Just collecting data — eventual goal is to compare MarinFold's distograms
against Protenix in both modes. No hypothesis on the Protenix numbers
themselves beyond "single-seq performs worse than MSA."

## Background

- Protenix code: [bytedance/Protenix](https://github.com/bytedance/Protenix)
  (issue [#309](https://github.com/bytedance/Protenix/issues/309) — v2 weights
  download is 403'd at ByteDance; we use the HF mirror).
- Protenix weights: [TMF001/pxdesign-weights](https://huggingface.co/TMF001/pxdesign-weights)
  (`checkpoint/protenix-v2.pt` plus CCD cache).
- FoldBench targets: [BEAM-Labs/FoldBench](https://github.com/BEAM-Labs/FoldBench),
  monomer list at `targets/monomer_protein.csv` (**334 rows**, matching the
  issue's "~330"). Each row is `<pdb>-assembly1,<chain>`; we fetch the
  biological assembly mmCIF from RCSB and pull the canonical full sequence
  (incl. unresolved residues) from `_entity_poly.pdbx_seq_one_letter_code_can`.

## Approach

Modal-hosted Protenix inference, fan out across `(protein × {single_seq, msa})`.
H100 GPU. Weights + CCD cache + pre-computed MSAs persisted in a Modal Volume;
predictions land in a separate output Volume and rsync'd locally for inspection.

### Settings

- **5 seeds × 8 diffusion samples per seed** per the issue (= 40 structures, 5
  distograms per protein per mode). N_cycle (recycles) and N_step (diffusion
  steps) left at Protenix v2 defaults (10 and 200) — no clear evidence from the
  Protenix team of FoldBench-specific overrides.
- **Sample selection:** top-1 by Protenix's built-in `ranking_score`
  ([`sample_confidence.py`](https://github.com/bytedance/Protenix/blob/main/protenix/model/sample_confidence.py)),
  which is the same AF3-style aggregate of pTM / ipTM / disorder penalty
  Protenix's dumper sorts by. The kept distogram is the one from the seed of the
  top-1 sample.
- **MSAs:** pre-computed once via Protenix's `colabfold` backend (ColabFold
  MMseqs2 API) so reruns are deterministic and free; persisted to the weights
  Volume. Single-seq mode omits the MSA paths and passes `--use_msa false`.

### Distogram extraction

Protenix's inference code computes the distogram head's logits (shape
`[N_token, N_token, 64]`) but immediately collapses them to scalar contact
probs and discards the per-bin distribution. We capture the raw logits via
`torch.nn.Module.register_forward_hook` on `runner.model.distogram_head` — a
standard PyTorch idiom, no monkey-patching and no vendor fork. The hook fires
once per seed (the trunk runs once per seed before diffusion samples), so we
get 5 distograms per (protein, mode), one per seed; we keep the one matching
the seed of the top-ranked sample.

### Layout

```
experiments/exp12_data_protenix_foldbench_monomers/
├── cli.py              # subcommands: prepare-inputs, run, select-best, score, plot
├── prepare_inputs.py   # FoldBench CSV → per-protein Protenix JSON + GT CIF cache
├── modal_app.py        # Modal app: weights/CCD volume, output volume, .map fan-out
├── distogram_hook.py   # forward-hook util that captures distogram_head logits
├── select_best.py      # per protein × mode: rank by ranking_score, link best
├── score.py            # MAE on expected distances + dRMSD on CA-CA distances
├── plot.py             # PNG plots from data/scores.csv (no notebook — see note)
├── data/scores.csv     # committed
├── plots/*.png         # committed
├── pyproject.toml
└── tests/
```

**Deliverable for the plot is `plot.py` + a committed PNG, not a `.ipynb`** —
the issue says notebook but [`experiments/AGENTS.md`](../AGENTS.md) is explicit
that experiments use `.py` files, not jupyter. Going with the convention.
(If a notebook is actually required, easy to add later as a follow-up.)

### Per-protein expected runtime

~10-20 GPU-min per protein at 5 seeds × 8 samples on H100 (300 aa), based on
AF3-class model rules of thumb. 10 proteins × 2 modes ≈ 3-7 GPU-hours;
100 proteins × 2 modes ≈ 30-60 GPU-hours. (Eventual full set: 334 proteins.)

### CLI

```bash
# one-time: prepare inputs locally (no GPU)
uv run python cli.py prepare-inputs --n 10 --out inputs/

# run on modal (fan out across proteins × modes)
uv run python cli.py run --inputs inputs/ --modes single_seq,msa --outputs-volume foldbench-protenix-runs

# locally: pick top-1 sample per (protein, mode), then score against GT
uv run python cli.py select-best --runs outputs/ --out best/ --manifest inputs/manifest.csv
uv run python cli.py score --best best/ --inputs inputs/ --out data/scores.csv
uv run python cli.py plot --scores data/scores.csv --out plots/
```

## Success criteria

For ~10 proteins (initial test), per protein per mode:

1. Full 64-bin distogram (`.npz` with shape `[N_token, N_token, 64]`)
   for single-seq AND for msa.
2. Predicted structure (mmCIF) for single-seq AND for msa — the top-1 sample.
3. Confidence JSON (`summary_confidence` from Protenix) for single-seq AND for
   msa — the top-1 sample.
4. `data/scores.csv` with MAE (expected-distance vs GT) and dRMSD (CA-CA pairwise
   distance RMSD, no sequence-separation filter) per protein per mode.
5. `plots/protenix_ss_vs_msa.png` showing the per-protein scores.

After human sign-off on the 10-protein numbers, scale to 100 proteins.

**Raw outputs (CIFs, distograms) are not committed.** They stay in the Modal
output Volume + optionally rsynced locally; only the small score CSV + plot land
in git. After human review they go to the
[`open-athena/MarinFold`](https://huggingface.co/buckets/open-athena/MarinFold)
HF bucket under `data/protenix-foldbench-monomers/`.

## Results

**Status: 100 of 100 proteins, paired in both modes.** Full run
complete after a workspace switch mid-stream — first 48 paired
proteins ran on Modal workspace `timodonnell` (hit billing cycle
spend limit at 102 of 200 jobs); the remaining 52 ran on `open-athena`
via the same idempotent worker. Final `data/scores.csv` is a
union of the two runs (200 rows = 100 proteins × 2 modes).

First 100 rows of FoldBench's `monomer_protein.csv` (30-761 aa).
Top-1 sample per (protein, mode) by Protenix's `ranking_score`.
Five per-protein metrics (per-mode mean / median):

**Distance-based metrics on CB-CB (CA-for-GLY) — in-range pair set**
(GT in `[2.31, 21.84] Å`, Protenix v2's distogram range):

| Mode | MAE distogram CB | dRMSD distogram CB |
|---|---|---|
| single_seq | 2.69 / 2.96 | 3.75 / 4.15 |
| msa        | 0.47 / 0.39 | 0.79 / 0.66 |

**Distance-based metrics on CB-CB — contact regime** (GT ≤ 8 Å,
CASP convention):

| Mode | MAE distogram CB (contacts) | dRMSD distogram CB (contacts) |
|---|---|---|
| single_seq | 3.56 / 3.97 | 5.49 / 6.18 |
| msa        | 0.40 / 0.32 | 0.73 / 0.54 |

**Structure-based metrics (CA / all-heavy)**:

| Mode | MAE structure CA | dRMSD CA | RMSD CA (Kabsch) | RMSD all-heavy (Kabsch) |
|---|---|---|---|---|
| single_seq | 6.09 / 6.41 | 8.29 / 8.43 | 13.68 / 14.41 | 14.09 / 15.03 |
| msa        | 0.73 / 0.47 | 1.34 / 0.84 |  1.86 /  1.18 |  2.36 /  1.69 |

All in Å. Lower is better.

**CASP contact precision @ top L / top L/5** (higher is better; CASP14
convention, see issue [#12](https://github.com/Open-Athena/MarinFold/issues/12)):

| Mode | prec_long_L | prec_long_L_5 |
|---|---|---|
| single_seq | 0.37 / 0.28 | 0.61 / 0.67 |
| msa        | 0.91 / 0.99 | 0.99 / 1.00 |

(Short / medium / L/2 cuts also in `data/scores.csv` per the full schema.)

Source CSVs (all use the same column schema; the all-samples file just
has 40× the rows + a ``selected_as_best`` flag):

- [`data/scores.csv`](data/scores.csv) — 200 rows, top-1 per (protein,
  mode) by Protenix's ``ranking_score``. All rows have ``selected_as_best=1``.
- [`data/scores_all_samples.csv`](data/scores_all_samples.csv) — 8000 rows,
  every (protein, mode, seed, sample_idx). 200 of them have
  ``selected_as_best=1`` (the same selection as the top-1 file).
- [`data/scores_summary.csv`](data/scores_summary.csv) — per-mode
  mean/median/min/max (computed on the top-1 view).

Plots ([`plots/`](plots/)) — two PNGs per metric:
- `{metric}_per_protein.png` — grouped bar chart of metric per protein
- `{metric}_ss_vs_msa_scatter.png` — paired scatter, y=single_seq, x=msa

### Notes on the numbers

- **Distogram metrics filter out unrepresentable pairs.** Protenix v2's
  distogram covers `[2.31, 21.84] Å`; pairs with GT outside this range
  produce a clipping bias unrelated to model quality. We drop them
  from the in-range MAE / dRMSD computation, and the contact-regime
  variant tightens the filter further to GT ≤ 8 Å. **For cross-model
  comparison with MarinFold, use the intersection of the two models'
  distogram ranges** (Protenix is the narrower one at 21.84 Å, so it
  defines the upper bound).
- **All distance + structure metrics show ~80-90% MSA improvement**.
  Median MSA Kabsch CA-RMSD is 1.18 Å vs single-seq 14.4 Å —
  global-fold-correct in MSA mode, global-fold-wrong otherwise.
- **CASP contact precision (long range)**: MSA mode is near-perfect
  (median precision @ top L = 0.99, @ top L/5 = 1.00). Single-seq
  averages 0.37 / 0.61 — usable but well below MSA.
- **Designed peptides are mode-insensitive**: e.g. 5sbj_A (30 aa with
  ACE/NH2 caps) scores ~identically across modes — no natural-homolog
  signal for the MSA to exploit.

### Outputs not in git

Raw structures, distograms, and confidence JSONs live on the Modal
`foldbench-protenix-runs` Volume (the 48 originals on `timodonnell`
and the 52 backfills on `open-athena`). The curated `best/` tree
(top-1 sample per protein-mode, 200 entries) is ~2.5 GB on the local
sync; `.gitignore`'d. Slated for upload to
`huggingface.co/buckets/open-athena/MarinFold/data/protenix-foldbench-monomers/`
after human review — the two-workspace split is the only nuisance
in the eventual HF consolidation step (both sets of raw outputs can
be downloaded and uploaded together).

## Conclusion

On 100 FoldBench monomers, Protenix v2 with MSA produces near-native
structures (median Kabsch CA-RMSD 1.18 Å, median dRMSD 0.84 Å) — a
regime where downstream eval against MarinFold's predictions is
meaningful. Single-sequence mode degrades sharply on natural proteins
(median Kabsch CA-RMSD 14.4 Å); it's a viable "MSA-free" baseline
only for designed peptides.

The distogram is a strong distance signal when the model is otherwise
performing well. With the principled filter (drop pairs whose GT is
outside the distogram's expressible range), MSA-mode distogram MAE
on CB is **0.47 Å mean / 0.39 Å median**, and contact-regime MAE is
**0.40 Å / 0.32 Å**. CASP-style contact precision @ top L (long range)
is **0.91 mean / 0.99 median** in MSA mode. For cross-model
comparison with MarinFold, score MarinFold's distogram against the
same pair sets — see issue [#12](https://github.com/Open-Athena/MarinFold/issues/12)
for the full metric definitions and filter conventions. *(The first
write-up of this experiment used unfiltered distogram metrics, which
were dominated by clipping bias; that's why the earlier issue comments
showed much larger distogram MAEs.)*

The 100-protein numbers track the 48-paired interim view closely
(median RMSD_CA single_seq 12.49 → 14.41 Å; msa 1.24 → 1.18 Å), so
extending to the full 334 FoldBench monomers is unlikely to shift
the headline conclusions but would tighten the tail statistics.
