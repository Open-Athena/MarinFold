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
uv run python cli.py select-best --runs outputs/ --out best/
uv run python cli.py score --best best/ --gt inputs/gt/ --out data/scores.csv
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

10-protein subset (first 10 rows of FoldBench's `monomer_protein.csv`,
30-394 aa). Top-1 sample per (protein, mode) by Protenix's
`ranking_score`; MAE on the kept sample's distogram vs GT CB-CB
(CA-for-GLY); dRMSD on the kept sample's CA-CA distance matrix vs GT.

| Mode | MAE (Å), mean | MAE (Å), median | dRMSD (Å), mean | dRMSD (Å), median |
|---|---|---|---|---|
| single_seq | 5.09 | 5.47 | 6.39 | 6.31 |
| msa | 3.50 | 3.39 | 1.51 | 1.25 |

Source CSVs: [`data/scores.csv`](data/scores.csv) (per-protein),
[`data/scores_summary.csv`](data/scores_summary.csv) (per-mode summary).

Plots ([`plots/`](plots/)):
- `mae_per_protein.png` — grouped bar chart of MAE per protein, both modes
- `drmsd_per_protein.png` — same, for dRMSD
- `mae_ss_vs_msa_scatter.png` — paired scatter, y=single_seq, x=msa
- `drmsd_ss_vs_msa_scatter.png` — same, for dRMSD

### Notes on the numbers

- **MAE inflates with protein size** because Protenix's distogram has
  a 21.84 Å max bin — pairs with GT distance beyond that get clipped
  to the last bin midpoint (~21.7 Å). For a 400-aa protein many pairs
  are at 30-80 Å, so the floor of MAE on those pairs is large. dRMSD
  doesn't have this issue.
- **dRMSD improvement is dramatic**: single_seq → MSA reduces mean
  dRMSD 4.2× (6.39 → 1.51 Å). On the hardest single_seq target
  (7qsj_A, 373 aa, single_seq dRMSD 6.7 Å), MSA brings it to 0.26 Å.
- **Designed-peptide outlier**: 5sbj_A (30 aa with ACE/NH2 caps)
  scores nearly identically in both modes (~0.48 Å MAE, ~0.93 Å
  dRMSD) — no MSA signal to exploit on a designed sequence with no
  natural homologs.

### Outputs not in git

Raw structures (800 .cif), distograms (100 .npz), and confidence JSONs
(800) live on the Modal `foldbench-protenix-runs` Volume and rsynced
locally to `outputs/` (1.0 GB). The curated `best/` tree (top-1 sample
per protein-mode, 20 entries) is ~80 MB. Both are `.gitignore`'d. They
go to `huggingface.co/buckets/open-athena/MarinFold/data/protenix-foldbench-monomers/`
after human review.

## Conclusion

For the 10-protein subset, Protenix v2 in MSA mode produces near-native
structures (median dRMSD 1.25 Å) and the distogram is a meaningful
signal for distance prediction (mean MAE 3.50 Å, dominated by the
inability to represent >22 Å distances). Single-sequence mode degrades
sharply on natural proteins (mean dRMSD 6.4 Å) while remaining usable
on small designed peptides. The pipeline is ready to scale to 100
proteins once these numbers are approved by a human.

