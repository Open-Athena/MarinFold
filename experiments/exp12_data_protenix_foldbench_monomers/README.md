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
  (`checkpoint/protenix-v2.pt` plus CCD cache `components.v20240608.cif`).
- FoldBench targets: [BEAM-Labs/FoldBench](https://github.com/BEAM-Labs/FoldBench),
  monomer list at `targets/monomer_protein.csv` (**334 rows**, matching the
  issue's "~330"). Each row is `<pdb>-assembly1,<chain>`; we fetch the
  biological assembly mmCIF from RCSB and pull the canonical full sequence
  (incl. unresolved residues) from `_entity_poly.pdbx_seq_one_letter_code_can`.

### Reproducibility pins

- **FoldBench commit:** [`4273f6877d82bd0b2fa476d1b2f34d121cbccc70`](https://github.com/BEAM-Labs/FoldBench/tree/4273f6877d82bd0b2fa476d1b2f34d121cbccc70)
  (HEAD of `main` as of our download, 2026-05-19 UTC). The
  `targets/monomer_protein.csv` file we used has sha256
  `43c2a5e9a73e84e00afb8d0108761547a8f9d6e52865e122792748c9c32bf595`.
- **Protenix:** `protenix==2.0.0` (PyPI, pulled by the Modal image). With
  `torch==2.7.1`, `biotite==1.4.0`, `gemmi==0.6.7`, `numpy==2.4.1`,
  Python 3.11.5 in the container.
- **Protenix v2 weights:** snapshot of `huggingface.co/TMF001/pxdesign-weights`
  taken at experiment time (the HF mirror; ByteDance's own download
  endpoint was returning 403 — see Protenix issue #309).
- **CCD cache:** `components.v20240608.cif` (+ its rdkit_mol pickle) from
  the same HF mirror.
- **MSA server:** ColabFold's MMseqs2 API at `https://api.colabfold.com`
  (set via `MMSEQS_SERVICE_HOST_URL` env var on the Modal image; the
  Protenix-server.com default doesn't synthesize the `non_pairing.a3m`
  files Protenix's inference path expects).
- **Scoring-side (local):** `python 3.11.15`, `gemmi 0.7.5`, `numpy 2.4.6`.

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
├── msa_depth.py        # a3m parsing + N_eff (Meff) reweighting (pure, testable)
├── plot.py             # PNG plots from data/scores.csv (no notebook — see note)
├── data/scores.csv     # committed
├── data/msa_depth.csv  # committed — per-protein MSA depth (n_seqs + N_eff)
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

## Metrics

This section is the reproduction spec. The implementation lives in
[`score.py`](score.py); this is what it does, with all constants
pinned. The CSV schema (200-row top-1 file
[`data/scores.csv`](data/scores.csv) and 8000-row per-sample file
[`data/scores_all_samples.csv`](data/scores_all_samples.csv)) is shared
across both files plus a `selected_as_best` flag (always 1 in top-1;
exactly 200 of 8000 in all-samples).

### Sample selection (per protein × mode)

5 trunk seeds × 8 diffusion samples per seed = 40 candidate structures
+ 5 distograms (the distogram is computed once per seed in the trunk;
the 8 diffusion samples within a seed share the same distogram).

- **Top-1** = max `summary_confidence.ranking_score` over the 40 samples.
- **Tie-breaking** when two samples tie on `ranking_score`: lower seed
  wins, then lower `sample_idx`. (Real tie observed once on this data:
  `5sbj_A/msa` seeds 1 vs 3.)
- The kept distogram is the one from the *seed* that produced the
  top-1 sample.

### Residue alignment

- Predicted CIF (from Protenix) + GT CIF (RCSB assembly1) parsed with
  `gemmi`, indexed by **`label_seq_id`** (the 1..N canonical sequence
  position from `_entity_poly_seq`).
- Both filtered to the **single L-peptide entity** (FoldBench monomers
  always have exactly one; we raise loudly otherwise).
- "Unresolved" residue (in `entity_poly_seq` but missing from
  `atom_site` for the relevant atom) → `None`, gets dropped from the
  per-metric pair set.

### Atom conventions

| Metric family | Atom on each side of each pair |
|---|---|
| Distogram MAE / dRMSD (in-range + contact) | CB (CA for GLY) — matches Protenix's distogram representative-atom convention |
| CASP contact precision (predicted score side) | distogram bins with **center ≤ 8 Å** summed → contact probability |
| CASP contact precision (GT side) | CB-CB ≤ 8 Å with both rep atoms resolved (CA for GLY) |
| Structure-distance MAE, dRMSD CA | CA |
| Kabsch RMSD CA | CA |
| Kabsch RMSD all-heavy | every shared `(label_seq_id, atom_name)` between pred and GT, **hydrogens excluded by element** (not by name) |

### Distogram bin scheme (pinned)

From Protenix v2 [`configs/configs_base.py`](https://github.com/bytedance/Protenix/blob/main/configs/configs_base.py):

```
min_bin = 2.3125 Å,  max_bin = 21.6875 Å,  no_bins = 64
```

Bin centers (matches Protenix's `get_bin_centers` in [`sample_confidence.py`](https://github.com/bytedance/Protenix/blob/main/protenix/model/sample_confidence.py)):

```python
bin_width = (max_bin - min_bin) / no_bins      # 0.302734375 Å
boundaries = linspace(min_bin, max_bin - bin_width, no_bins)
centers    = boundaries + 0.5 * bin_width
# centers[0]  = 2.464 Å,   centers[63] = 21.535 Å
```

Per-pair **expected distance** = `sum(p_bin × center_bin)` over the 64
bins (after softmax). The captured `.npz` already stores softmaxed
probabilities. Per-pair **contact probability** = `sum(p_bin)` over
bins whose `center ≤ 8 Å` — that's bins 0..18 (19 bins) for Protenix v2.

### Pair selection

For every distance-based metric, pairs come from `np.triu_indices(N, k=1)`
(upper triangle, no diagonal), then filtered:

| Metric | Filter |
|---|---|
| `mae_distogram_cb_angstrom` / `drmsd_distogram_cb_angstrom` (**option B**) | both rep atoms resolved in GT **AND** `2.3125 ≤ gt_cb_cb ≤ 21.6875 Å` |
| `mae_distogram_cb_contact_angstrom` / `drmsd_distogram_cb_contact_angstrom` (**option C1**) | both rep atoms resolved in GT **AND** `gt_cb_cb ≤ 8.0 Å` |
| `mae_structure_ca_angstrom` / `drmsd_ca_angstrom` | both CAs resolved in **both** GT and pred |
| `rmsd_ca_angstrom` (Kabsch) | shared CA residues between pred and GT (Kabsch is atom-set based, no upper-triangle restriction) |
| `rmsd_all_heavy_angstrom` (Kabsch) | shared `(label_seq_id, atom_name)` keys, hydrogens excluded |

**No sequence-separation filter** for the MAE/dRMSD metrics (includes
`|i-j|=1` bonded-neighbor pairs at ~3.8 Å). This compresses MAE/dRMSD
slightly relative to a `|i-j|≥6` cutoff used in some literature.

### Formulas

Let `iu = np.triu_indices(N, k=1)`, `usable` = filter mask per the table above:

```python
# Distogram MAE on CB (option B / in-range)
mae_distogram_cb = mean(|expected[iu][usable] - gt_cb_d[iu][usable]|)
drmsd_distogram_cb = sqrt(mean((expected[iu][usable] - gt_cb_d[iu][usable]) ** 2))

# Distogram MAE/dRMSD on CB (option C1 / contact-regime) — same formulas,
# different `usable` mask (gt ≤ 8 Å).

# Structure-derived MAE / dRMSD on CA — same diffs vector for both:
ca_diffs = pred_ca_d[iu][usable] - gt_ca_d[iu][usable]
mae_structure_ca = mean(|ca_diffs|)
drmsd_ca         = sqrt(mean(ca_diffs ** 2))

# Kabsch RMSD over CA atoms (and over all heavy atoms, via shared key set):
res = gemmi.superpose_positions(pred_xyz, gt_xyz)   # Kabsch under the hood
rmsd_ca = res.rmsd
```

### CASP contact precision (option C2)

CASP14+ convention. **Sequence-separation classes** (inclusive bounds):

- **Short**: `6 ≤ |i − j| ≤ 11`
- **Medium**: `12 ≤ |i − j| ≤ 23`
- **Long**: `|i − j| ≥ 24`

(Pairs with `|i − j| ≤ 5` are excluded — too easy.)

For each class:
1. Restrict to pairs (i, j) with `i < j`, separation in the class's
   range, both rep atoms resolved in GT.
2. Rank pairs by predicted **contact probability** descending. Stable
   mergesort for determinism.
3. For each `k ∈ {1, 2, 5}`: take the top `max(1, L // k)` pairs;
   `precision = (#true_contacts_in_topK) / topK`.
4. True contact = `gt_cb_cb ≤ 8 Å` with both rep atoms resolved.

Output columns: `prec_{short,medium,long}_{L,L_2,L_5}` plus
`n_{short,medium,long}_contacts` for denominator info. NaN if a class
has no eligible pairs (only happens for very short proteins, e.g. long-
range needs L ≥ 25).

### Adding new distogram-derived metrics

All raw artifacts needed to compute new per-(protein, mode) metrics are
already in `best/` (and uploaded to the HF bucket — see "Outputs"
below). To add a new metric, you don't need to re-run Protenix; just
load the existing files. Quick reference:

**Loading the distogram** (the most common starting point):

```python
import numpy as np
d = np.load("best/msa/7uk8_A/distogram.npz")["probs"]
# d.shape == (N, N, 64)  — N = protein length
# d.dtype == float32
# d sums to 1.0 along axis -1 (softmaxed probabilities)
# Bin k corresponds to center _DISTOGRAM_BIN_MIDPOINTS[k]
#   = 2.464 Å for k=0, ..., 21.535 Å for k=63
# See score.py for the exact bin-center formula.
```

The same `.npz` schema (`probs` key, `[N, N, 64]` float32) is in every
`best/{mode}/{stem}/distogram.npz`. The matrix is symmetric and the
upper triangle is the canonical pair set.

**Loading the predicted structure:**

```python
import gemmi
s = gemmi.read_structure("best/msa/7uk8_A/structure.cif")
# Same layout / atom naming as the GT CIFs in inputs/gt/.
# Per-atom b_factor = Protenix's pLDDT × 100 (per the Protenix dumper).
```

**Loading Protenix's summary confidence** (per-sample confidence; useful
for filters or alternate ranking criteria):

```python
import json
c = json.load(open("best/msa/7uk8_A/confidence.json"))
# Keys: plddt, gpde, ptm, iptm, chain_plddt, chain_ptm, chain_iptm,
#       chain_pair_iptm, chain_pair_plddt, chain_pair_iptm_global,
#       has_clash, disorder, ranking_score, num_recycles, ...
# All scalars are Python floats; chain_* are length-1 lists for monomers.
```

**Where to add a new metric in `score.py`:**

1. Compute it inside [`_compute_metrics()`](score.py) — that function
   already has `expected` (distogram-derived expected distances),
   `contact_probs`, `gt_coords` (CA + rep), `gt_rep_d / gt_rep_mask`
   (CB-CB distance matrix), `pred_coords`, and `gt_atoms` (heavy
   atoms) all in scope. Add your metric using those.
2. Add the column to `_MetricResult` (dataclass), `ProteinScore`
   (dataclass), `_FIELDS` (CSV header tuple), and `_format_csv_row()`
   (CSV writer). The smoke test in `tests/test_score_smoke.py` will
   catch most schema mistakes.
3. Re-score the top-1 view:
   `uv run python cli.py score --best best --inputs inputs --out data/scores.csv`
   (no Modal calls, no GPU, ~30 sec for 200 rows).
4. Re-score all 8000 samples:
   `uv run python _scripts/score_all_samples.py`
   (streams from the Modal `foldbench-protenix-runs` Volume; ~15 min for the full set).
5. Re-upload changed files:
   `uv run python _scripts/upload_to_hf.py --no-msa`
   (xet dedupes unchanged files; only the new CSVs / plots go over the wire).

**Some metric ideas the existing data supports out of the box:**

- **LDDT** (any atom set, GT-derived): compute from pred + GT structures,
  no distogram needed. Standard CASP convention is 15 Å inclusion
  radius, thresholds 0.5/1/2/4 Å.
- **Distogram entropy per pair** (or summed): `-(probs * log(probs)).sum(axis=-1)`
  — a model-confidence proxy that's independent of GT.
- **Calibration**: bin the predicted distance distribution against the
  empirical GT distribution. Tells you how well-calibrated the distogram
  is across distance ranges.
- **Variance of the distogram-implied distance**: `Σ p_bin × (center_bin − expected)²`
  — a per-pair uncertainty estimate.
- **Top-K contact-probability ROC / AUC**: alternative to the CASP top-L/k
  precision we already compute.
- **Inter-residue distance error distribution** (per-distance-bin MAE):
  bin GT distances and report MAE within each, to see where the model
  fails (near vs far pairs).

### LDDT (CASP convention, option D)

**Inclusion radius** 15 Å, **distance-difference thresholds**
`{0.5, 1, 2, 4} Å`, **sequence separation** `|i-j| ≥ 1` (only self
excluded). Per-residue LDDT = mean over thresholds of
(#preserved_at_t / #scored_pairs_for_residue). Global LDDT = mean over
residues with ≥1 scored pair. Range [0, 1].

Five variants in two families:

**Structure-derived** (predicted CIF vs GT CIF — analogous to
RMSD-Kabsch but local instead of global):

| Column | Atom set |
|---|---|
| `lddt_structure_ca` | CA only |
| `lddt_structure_cb` | CB (CA for GLY) |
| `lddt_structure_all_heavy` | All heavy atoms (intra-residue pairs excluded) |

The all-heavy variant uses a direct numpy implementation; may diverge
slightly from OpenStructure on symmetry-related atom labels (Asp
OD1/OD2, Leu CD1/CD2, etc.) — fine for internal Protenix-vs-MarinFold
comparison, use OpenStructure if you need byte-identity with
FoldBench-paper LDDT.

**Distogram-derived** (per-pair predicted distance distribution vs GT
CB-CB distance — there's no CA / all-atom variant because the distogram
only represents CB-CB):

| Column | What |
|---|---|
| `lddt_distogram_cb` | Point estimate: `expected_distance` from the distogram as the predicted distance |
| `lddt_distogram_cb_soft` | Probabilistic: per pair, score at threshold `t` is `Σ p_bin` over bins whose center is in `(gt - t, gt + t)`. Uses the full distribution; rewards calibrated uncertainty. |

The 15 Å inclusion radius is well inside the distogram's expressible
range (centers go up to 21.54 Å), so unlike the unfiltered distogram
MAE there's no clipping bias to worry about for LDDT.

### MSA depth (n_seqs + N_eff)

Per-protein MSA depth, so downstream plots can show predictor accuracy
as a function of how much homolog signal each target has. Computed from
the merged unpaired MSA Protenix actually feeds the model
(`{stem}/msa/0/0/non_pairing.a3m` on the `protenix-foldbench-msa`
Volume). Depth is a property of the *protein*, not the mode/seed/sample,
so it lives in its own [`data/msa_depth.csv`](data/msa_depth.csv) and is
joined to `scores.csv` on `(pdb_id, chain_id)` at plot time.

Implementation: [`msa_depth.py`](msa_depth.py) (pure numpy, unit-tested)
+ the `compute_msa_depth` Modal CPU function in
[`modal_app.py`](modal_app.py). Two metrics:

- **`n_seqs`** — raw number of sequences in the a3m (incl. the query as
  record 0). The simple "total entries" count.
- **`n_eff_0.8` / `n_eff_0.62`** — redundancy-reweighted effective count
  (Meff) at 0.8 (AlphaFold2 / HHblits convention) and 0.62 (common in
  the coevolution literature). Definitions, pinned:

  - **Match-state extraction**: a3m uppercase + `-` are match columns
    (aligned to the query); lowercase = insertions, dropped. Every
    sequence then has length `L` = query length.
  - **Pairwise identity** = (# columns both non-gap AND equal) / (# columns
    both non-gap) — identity over the overlapping region, so a partial-
    coverage fragment that matches its overlap clusters with the query
    rather than looking distant because of its gaps.
  - **Meff at threshold `t`**: `w_i = 1 / |{j : identity(i,j) ≥ t}|`
    (the set includes `i`); `n_eff = Σ w_i`. Fully diverse → `n_eff == n_seqs`;
    all-identical → `n_eff == 1`.

CSV columns: `pdb_id, chain_id, stem, n_residues, query_len, n_seqs,
n_eff_0.8, n_eff_0.62`. (`query_len` should equal `n_residues`; a
mismatch flags an a3m/sequence inconsistency.)

Generate it (CPU-only, reads the MSA Volume):

```bash
uv run python _scripts/compute_msa_depth.py --inputs inputs --out data/msa_depth.csv
```

Then add accuracy-vs-depth plots (`plots/{metric}_vs_msa_depth.png`,
both modes, log-x depth). **Accuracy is read out as LDDT** (the LDDT
family is the default `--depth-metrics`, led by LDDT-CA): it's bounded
[0, 1] and length-robust, so it compares cleanly across proteins of
different sizes — unlike raw RMSD/MAE. Override with `--depth-metrics`
if you want a different metric on the y-axis.

```bash
uv run python cli.py plot --scores data/scores.csv --out plots/ \
    --msa-depth data/msa_depth.csv --depth-col n_eff_0.8
```

### Cross-model comparison (e.g. MarinFold-side scoring)

When scoring MarinFold's distogram against these results:

1. Apply identical residue alignment + atom-pair logic (CB-CB / CA-for-GLY
   for distogram metrics, CA-CA for structure metrics).
2. Substitute MarinFold's own distogram bin range for the option-B
   filter — and for the head-to-head, use the **intersection** of the
   two ranges. MarinFold's `contacts-and-distances-v1` covers
   `[0, 32] Å` per [`exp1_document_structures_contacts_and_distances_v1`](../exp1_document_structures_contacts_and_distances_v1/README.md);
   intersection with Protenix is `[2.3125, 21.6875] Å`, so **score
   MarinFold on the same Protenix-defined pair set**.
3. For option C1, the 8 Å contact filter is identical across models.
4. For option C2, compute MarinFold's per-pair contact probability the
   same way (sum mass on bins with center ≤ 8 Å — bin indices differ
   because MarinFold's binning is different). The CASP precision
   formulas are then identical.

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

**LDDT** (CASP convention: 15 Å inclusion, thresholds 0.5/1/2/4 Å,
range [0, 1], higher is better):

| Mode | LDDT-CA (struct) | LDDT-CB (struct) | LDDT-all-heavy (struct) | LDDT-CB (distogram) | LDDT-CB (distogram, soft) |
|---|---|---|---|---|---|
| single_seq | 0.48 / 0.42 | 0.45 / 0.39 | 0.42 / 0.36 | 0.43 / 0.37 | 0.42 / 0.36 |
| msa        | 0.93 / 0.95 | 0.91 / 0.93 | 0.85 / 0.86 | 0.91 / 0.93 | 0.88 / 0.89 |

Structure-LDDT and distogram-LDDT on CB agree closely (~0.91 in MSA
mode for both) — meaning the distogram-derived expected distance is a
faithful proxy for the predicted CB-CB distance. The soft variant
(probabilistic, using the full bin distribution) is systematically
~0.04 stricter than the point-estimate version.

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

### Outputs

**Uploaded to HuggingFace:**
[`open-athena/MarinFold/blob/main/data/protenix-foldbench-monomers/`](https://huggingface.co/datasets/open-athena/MarinFold/tree/main/data/protenix-foldbench-monomers)

Layout:

```
data/protenix-foldbench-monomers/
├── README.md                      # this experiment's writeup mirror
├── scores.csv                     # 200 rows top-1 (= repo's data/scores.csv)
├── scores_all_samples.csv         # 8000 rows (= repo's data/scores_all_samples.csv)
├── scores_summary.csv             # per-mode aggregates
├── manifest.csv                   # 100 proteins (pdb_id, chain, n_residues, ...)
├── best/
│   ├── single_seq/{pdb}_{chain}/{structure.cif, confidence.json, distogram.npz, provenance.json}
│   └── msa/{pdb}_{chain}/{same layout}
├── gt/*.cif                       # 100 GT biological-assembly mmCIFs (~26 MB)
└── msa/{pdb}_{chain}/msa/         # pre-computed ColabFold MSAs
    └── 0/                         # Protenix's colabfold-mode layout
        ├── pairing.a3m            # stub for monomers (just the query)
        ├── 0/non_pairing.a3m      # the real unpaired MSA
        ├── bfd.mgnify30.metaeuk30.smag30.a3m
        └── uniref.a3m
```

**Raw 40-sample outputs (~50 GB) NOT uploaded.** They live on the Modal
`foldbench-protenix-runs` Volume (48 originals on workspace
`timodonnell`, 52 backfills on `open-athena`) — useful if you need to
re-rank or look at sample variance, but the curated top-1 in `best/`
plus the `scores_all_samples.csv` per-sample metrics cover the
expected downstream uses. Re-sync via
`modal volume get foldbench-protenix-runs <path> <local>` with the
right `MODAL_PROFILE` if needed.

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
