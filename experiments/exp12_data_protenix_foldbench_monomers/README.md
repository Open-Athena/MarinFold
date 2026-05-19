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
  monomer list at `targets/monomer_protein.csv` (**427 rows** — the issue says
  "~330" but the upstream CSV has 427; we just take the first N in CSV order).

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
100 proteins × 2 modes ≈ 30-60 GPU-hours.

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

_(Fill in after the 10-protein run completes.)_

## Conclusion

_(Fill in after results are in.)_
