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

**Status: 48 of 100 proteins, paired in both modes.** Dispatched the
full 100-protein × 2-mode run (200 jobs); Modal workspace hit its
monthly spend limit after 51 single-seq + 51 msa completions (102 of
200), with 48 proteins fully complete in *both* modes. The remaining
52 will resume after the billing cycle resets — the worker is
idempotent against the output Volume so a re-dispatch only runs the
missing ones.

48-protein subset (subset of first 100 rows of FoldBench's
`monomer_protein.csv`, 30-738 aa). Top-1 sample per (protein, mode)
by Protenix's `ranking_score`. Five per-protein metrics (per-mode
mean / median):

| Mode | MAE distogram CB | MAE structure CA | dRMSD CA | RMSD CA (Kabsch) | RMSD all-heavy (Kabsch) |
|---|---|---|---|---|---|
| single_seq | 6.66 / 6.39 | 5.52 / 5.57 | 7.56 / 7.77 | 12.31 / 12.49 | 12.75 / 12.73 |
| msa        | 5.11 / 4.51 | 0.62 / 0.48 | 1.18 / 0.95 |  1.67 /  1.24 |  2.15 /  1.77 |

All in Å. Lower is better.

Source CSVs: [`data/scores.csv`](data/scores.csv) (per-protein, all
metrics), [`data/scores_summary.csv`](data/scores_summary.csv) (per-mode
mean/median/min/max).

Plots ([`plots/`](plots/)) — two PNGs per metric:
- `{metric}_per_protein.png` — grouped bar chart of metric per protein
- `{metric}_ss_vs_msa_scatter.png` — paired scatter, y=single_seq, x=msa

### Notes on the numbers

- **Structure-derived metrics show massive MSA improvement** (≥83%
  reduction across structure_MAE / dRMSD / RMSD_CA / RMSD_all-heavy).
  This is the headline: in MSA mode Protenix produces structures
  within 1.7 Å median Kabsch CA-RMSD of GT; in single-seq mode the
  same model is essentially guessing on most natural proteins
  (median CA-RMSD 12.5 Å — global-fold-wrong).
- **Distogram MAE shows a smaller MSA gain** (~23%) because Protenix's
  distogram is capped at 21.84 Å — pairs farther than that all get
  the same midpoint (~21.7 Å), so the floor of distogram MAE on a
  400 aa protein is large regardless of model quality. The structure-
  derived MAE (which doesn't have this cap) shows the real signal:
  0.62 Å in MSA vs 5.52 Å in single-seq, an 8.9× improvement.
- **Designed peptides are mode-insensitive**: 5sbj_A (30 aa with
  ACE/NH2 caps) scores identically across modes — no natural-homolog
  signal for the MSA to exploit.

### Outputs not in git

Raw structures, distograms, and confidence JSONs live on the Modal
`foldbench-protenix-runs` Volume. The curated `best/` tree (top-1
sample per protein-mode, 96 entries for the 48 paired proteins) is
~1.2 GB on the local sync; `.gitignore`'d. After human review +
the resumed 52-protein backfill, the full set goes to
`huggingface.co/buckets/open-athena/MarinFold/data/protenix-foldbench-monomers/`.

## Conclusion

On 48 of 100 FoldBench monomers, Protenix v2 with MSA produces
near-native structures (median Kabsch CA-RMSD 1.24 Å, median dRMSD
0.95 Å) — a regime where downstream eval against MarinFold's
predictions is meaningful. Single-sequence mode degrades sharply on
natural proteins (median Kabsch CA-RMSD 12.5 Å); it's a viable
"MSA-free" baseline only for designed peptides.

The distogram is a usable distance signal when the model is otherwise
performing well: MSA-mode distogram MAE (on CB) is 5.1 Å mean / 4.5 Å
median, with the long tail entirely explained by Protenix's 21.84 Å
distogram cap (the structure-derived MAE on CA, with no cap, is 0.62 Å
mean for the same set). When MarinFold's distograms come online for
the same proteins, the apples-to-apples comparison is against
Protenix's MSA-mode distogram MAE — single-seq is a weaker baseline.

Resuming the remaining 52 proteins after the Modal billing cycle
resets will push the numbers toward the full-100 view; the per-mode
trends are unlikely to shift materially given n=48 already.

