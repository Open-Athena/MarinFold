---
marinfold_experiment:
  issue: 20
  title: "exp: eval 1B model on 100 foldbench monomers"
  kind: evals
  branch: exp/20-marinfold-1b-foldbench-eval
---

# exp: eval 1B model on 100 foldbench monomers

**Issue:** [#20](https://github.com/Open-Athena/MarinFold/issues/20) · **Kind:** `evals` · **Branch:** `exp/20-marinfold-1b-foldbench-eval`

## Question

On the FoldBench monomer subset already collected in exp12 / PR #14, how does MarinFold’s `1B` model compare to Protenix v2 in single-sequence mode and Protenix v2 with MSAs when all three are scored on the same proteins with the same distogram-derived metrics?

## Hypothesis

The `1B` model will underperform Protenix for most proteins and metrics. But this will let us know where we stand.

## Background

This builds directly on issue #12 and PR #14, which collected Protenix outputs on FoldBench monomers and already produced paired per-protein metrics for `single_seq` and `msa` modes.

Relevant prior artifacts:

- Issue #12: Protenix-on-FoldBench data collection + scoring
- PR #14: experiment scaffold, scoring code, and results write-up for the Protenix dataset
- `experiments/exp12_data_protenix_foldbench_monomers/`: GT CIF handling, Protenix scoring logic, and the current metric conventions
- `experiments/exp9_evals_test_distance_heatmaps/`: prior zero-shot `1B` eval surface on held-out proteins
- `MODELS.yaml`: `1B` is the default MarinFold model for `contacts-and-distances-v1`

Important metric note from exp12: Protenix’s distogram range is narrower than MarinFold’s (`~2.31 Å` to `~21.84 Å`), so the fair comparison should use the same residue-pair filtering convention as exp12, i.e. compare on the intersection of the models’ expressible distance range, with Protenix’s upper bound defining the cap.

## Approach

### Inputs

- **FoldBench monomer set**: reuse exp12's 100-protein paired set; no
  Protenix re-runs. `fetch_protenix_data.py` pulls only what we need
  from the `open-athena/MarinFold` HF bucket (~26 MB of GT mmCIFs +
  small CSVs), via the bucket API in `huggingface_hub>=1.5` — note
  that the bucket APIs are incompatible with the `transformers <5`
  pin we need for `vllm 0.7`, so the fetcher is run via
  `uv run --with "huggingface_hub>=1.5" python fetch_protenix_data.py`
  in an ephemeral env, not the main exp20 venv.
- **Model**: `1B` from `MODELS.yaml`, resolved + snapshot-downloaded
  the same way exp9 does it (`allow_patterns=[f"{subdir}/*"]` so we
  only pull the matching subfolder, not all sibling models in the repo).

### Atom convention

CB-CB queries with CA fallback for any GLY (and our reader's UNK —
the entity_poly_seq's non-canonical residues map to UNK and we
fall back to CA there too, since CB conventions are undefined for
non-standard side chains). This matches Protenix's distogram
representative-atom convention exactly, so the cross-model
comparison is apples-to-apples.

### Inference

`run_1b_eval.py` (local) and `modal_app.py` (Modal). Both:

1. Read each protein's canonical 1..N residue sequence from the
   Protenix GT mmCIF (`canonical_sequence.py` — handles the
   biological-assembly multi-subchain case, maps non-canonical
   residues to UNK).
2. Build the zero-shot v1 prompt
   `<contacts-and-distances-v1> <begin_sequence> <AAs…> <begin_statements>`.
3. For all (i, j) with i < j, query
   `<distance> <p_i> <p_j> <CB|CA> <CB|CA>` with vLLM's prefix cache
   (one trunk forward, N²/2 tails).
4. Renormalize the top-128 logprobs over the 64 `<d_X.X>` bin tokens
   to a probability vector. Save symmetric `[N, N, 64]` to
   `outputs/{stem}/distogram.npz`.
5. Each protein also writes a `provenance.json` with
   `elapsed_seconds`, `n_residues`, `n_pairs`, batch size, the bin
   scheme, and a `hardware` block (`gpu_name`,
   `gpu_total_memory_gb`, `runner_tag`, hostname, torch version).
   These let `collect_timings.py` build a single
   `data/timings.csv` across local + Modal runs, used for the
   length-vs-runtime plot.

Idempotent: re-running skips any protein whose `distogram.npz`
already has the expected `[N, N, 64]` shape. Both the local and
Modal drivers respect this.

### Scoring

`score_marinfold.py` copies and parameterizes the distogram-only
portion of exp12's `score.py` so the bin scheme is a `BinScheme`
parameter — MarinFold's 0.5 Å bins (midpoints 0.25..31.75 Å)
plug into the same metric functions Protenix used. The
**in-range MAE/dRMSD pair filter uses Protenix's narrower range**
(`[2.3125, 21.6875]`) — the intersection rule from exp12's
"Cross-model comparison" section.

`score_comparison.py` merges MarinFold's per-protein scores with
Protenix's exp12 `scores.csv` into a single 300-row CSV (one row
per `(protein, method)` for `marinfold_1b`,
`protenix_single_seq`, `protenix_msa`). Headline metrics for the
3-way comparison are LDDT-distogram-CB (point + soft), MAE,
dRMSD, plus CASP contact precision. Structure-side metrics
(CA-RMSD, all-heavy RMSD, structure-LDDT) are **N/A** for
MarinFold and recorded as such — 1B doesn't emit a structure in
this eval, and the issue notes the non-parity is acceptable.

### Hypothesis verdict

The issue spells out that the hypothesis "1B underperforms
Protenix" counts as supported iff `marinfold_1b`'s aggregate mean
sits strictly between `protenix_single_seq` and `protenix_msa`
on at least 2 of 3 headline metrics. `score_comparison.py`
computes this verdict, keeps `data/scores_summary.csv` as a clean
CSV, and writes the verdict details to
`data/hypothesis_verdict.json`.

### Compute

- **Local**: RTX A5000 (24 GB) via `run_1b_eval.py`. Smoke run on
  3 proteins (30, 81, 245 aa) on this host confirmed end-to-end
  correctness; per-protein runtime grows quadratically (0.6 s,
  9.5 s, 221.3 s respectively — see `data/timings.csv`).
  Extrapolated to the longest FoldBench monomer (~760 aa,
  ~289k pairs), local runtime would be ~40-60 min/protein, putting
  the full 100-protein local run somewhere in the 10-15 GPU-hour
  range.
- **Modal**: H100 via `modal_app.py`. Worker class with
  `@modal.concurrent(max_inputs=1)` keeps vLLM resident across
  per-protein RPCs; weights cached in a named Volume so cold starts
  after the first run are fast. Same idempotent layout as local.
  Modal's H100 is ~3-5x faster than the local A5000 on this
  workload, so the full 100-protein run should land in
  ~2-4 hours.

Both runners save progress as they go (one `distogram.npz` per
protein), so the eval is partial-result-friendly per the issue's
"have it save progress as it goes" requirement.

### Files

- `fetch_protenix_data.py` — pull Protenix scores + GT CIFs from HF.
- `canonical_sequence.py` — read the 1..N entity_poly_seq from a
  Protenix biological-assembly GT mmCIF.
- `run_1b_eval.py` — local vLLM driver.
- `modal_app.py` — Modal driver.
- `download_outputs.py` — pull the Modal Volume back to local
  `outputs/` (idempotent; used when `modal run` was detached).
- `score_marinfold.py` — MarinFold-side per-protein scoring.
- `score_comparison.py` — merge + hypothesis verdict.
- `collect_timings.py` — aggregate `outputs/*/provenance.json` →
  `data/timings.csv`.
- `plot_comparison.py` — 3-way per-protein + aggregate plots + the
  length-vs-runtime plot + LDDT / prec-at-L swarm plots.
- `tests/test_smoke.py` — bin scheme, contact mask, CIF parser.

Zero-shot only — no seeded contacts, no hints.

## Running this experiment

Designed for re-runs with different checkpoints. Idempotency
(local + Modal) is keyed on `(stem, n_residues, model_nickname)`
— bumping `MODELS.yaml` to a new model automatically invalidates
the old `distogram.npz` files (provenance JSON's `model_nickname`
must match the requested one or the protein is recomputed).

### Prerequisites

- `uv` (Python toolchain).
- `modal` CLI authenticated for the Modal run (`modal token current`
  should resolve a workspace; we used `open-athena`).
- A GPU on the local host if you want to use `run_1b_eval.py` (any
  ≥24 GB card; an RTX A5000 is enough). Modal needs no local GPU.

### One-time setup

```bash
cd experiments/exp20_evals_marinfold_1b_foldbench
uv sync
# Fetch Protenix scores + GT mmCIFs from the open-athena/MarinFold
# HF bucket. The bucket API needs huggingface_hub>=1.5, which
# conflicts with our vllm/transformers pins, so we run it in an
# ephemeral env that is independent of the main exp20 venv.
uv run --with "huggingface_hub>=1.5" python fetch_protenix_data.py
```

That populates `protenix_data/data/protenix-foldbench-monomers/`
with `manifest.csv`, `scores.csv`, `scores_summary.csv`, and 100
`gt/*.cif` files (~30 MB total).

### Inference

Pick **one** of local or Modal — both write to the same
`outputs/<stem>/{distogram.npz, provenance.json}` layout so the
downstream scoring code doesn't care which produced them.

**Local (one machine, sequential):**

```bash
uv run python run_1b_eval.py                  # full 100 proteins
uv run python run_1b_eval.py --limit 5        # smoke
uv run python run_1b_eval.py --model 1B       # override MODELS.yaml nickname
```

**Modal (parallel H100s, recommended for full runs):**

```bash
# Smoke (3 proteins) — runs in the foreground, streams logs:
uv run modal run modal_app.py --limit 3

# Full run, detached (Modal keeps running even if your shell exits):
uv run modal run --detach modal_app.py

# Detached runs don't download outputs back; pull them when ready:
uv run python download_outputs.py
```

Both drivers skip proteins whose `distogram.npz` matches the
requested `model_nickname` already, so it's safe to re-run after a
crash or a partial smoke.

### Score + plot

```bash
uv run python score_marinfold.py     # writes data/marinfold_scores.csv
uv run python collect_timings.py     # writes data/timings.csv
uv run python score_comparison.py    # writes data/scores.csv + data/scores_summary.csv (incl. hypothesis verdict)
uv run python plot_comparison.py     # writes plots/*.png
```

The plotting step depends on `protenix_data/` (already fetched
above) for the structure-LDDT side of the 5-way swarm plot, and on
`../exp12_data_protenix_foldbench_monomers/data/timings.csv` (from
exp12) for the 3-way timing-vs-length plot. Both default paths;
override via `--protenix-scores` / `--protenix-timings`.

### Re-running with a new model

1. Edit `MODELS.yaml` (or add a new entry with a different
   nickname).
2. Re-run any of the inference commands above — both local and
   Modal will recompute proteins whose `provenance.json`
   `model_nickname` doesn't match what you requested.
3. The Modal outputs Volume (`marinfold-1b-foldbench-runs`) is
   shared across runs. If you want a clean slate per checkpoint,
   either delete the volume contents (`modal volume rm ...`) or
   change the volume name in `modal_app.py:OUTPUTS_VOLUME_NAME`
   before launching. The default behavior — overwrite per protein
   on a new model — is what we usually want.
4. Re-run the score + plot chain. CSVs and plots are regenerated
   in place.

### Smoke tests

```bash
uv run --extra test pytest tests/
```

Doesn't run the model — just checks bin midpoints, contact mask,
the canonical-sequence reader, and the smoke distogram (if present).

## Success criteria

1. We have a paired comparison for the same FoldBench proteins already used in exp12, with one row per `(protein, method)` for `marinfold_1b`, `protenix_single_seq`, and `protenix_msa`.
2. The experiment produces committed summary artifacts:
   - at least one CSV with per-protein comparison metrics
   - at least one plot showing the three-way comparison
3. Headline metrics are computed with the same scoring convention across methods:
   - distogram LDDT 
   - distogram MAE
   - distogram dRMSD
   - long-range contact precision (and any other contact cuts we decide to keep from exp12)
4. The hypothesis counts as supported if `1B` is better than Protenix single-seq but worse than Protenix+MSA on at least two of the three headline aggregate metrics above.
5. The README / issue write-up ends with a direct answer to “where does 1B sit relative to Protenix on this dataset?”

## Results

100 of 100 FoldBench monomers (30–761 aa) scored against the
exp12 Protenix v2 results in both `single_seq` and `msa` modes.
Headline aggregate (mean / median across the 100 proteins,
distogram-derived metrics on CB-CB / CA-for-GLY with the
intersection-range filter `[2.31, 21.69] Å`):

| metric | direction | marinfold_1b | protenix_single_seq | protenix_msa |
|---|---|---:|---:|---:|
| `lddt_distogram_cb` | higher | **0.271 / 0.233** | 0.432 / 0.367 | 0.912 / 0.929 |
| `lddt_distogram_cb_soft` | higher | **0.284 / 0.256** | 0.416 / 0.360 | 0.876 / 0.888 |
| `mae_distogram_cb_angstrom` (Å) | lower | **5.64 / 5.94** | 2.69 / 2.96 | 0.47 / 0.39 |
| `drmsd_distogram_cb_angstrom` (Å) | lower | **7.14 / 7.42** | 3.75 / 4.15 | 0.79 / 0.66 |
| `mae_distogram_cb_contact_angstrom` (Å) | lower | **5.75 / 5.92** | 3.56 / 3.97 | 0.40 / 0.32 |
| `drmsd_distogram_cb_contact_angstrom` (Å) | lower | **8.50 / 8.85** | 5.49 / 6.18 | 0.73 / 0.54 |
| `prec_long_L` | higher | **0.258 / 0.187** | 0.373 / 0.281 | 0.913 / 0.992 |
| `prec_long_L_5` | higher | **0.427 / 0.346** | 0.608 / 0.671 | 0.989 / 1.000 |

Source: [`data/scores.csv`](data/scores.csv) (300 rows: 100 proteins ×
3 methods). Per-method aggregates in
[`data/scores_summary.csv`](data/scores_summary.csv). The
verdict in [`data/hypothesis_verdict.json`](data/hypothesis_verdict.json)
is `not_supported`:
on every one of the 3 headline metrics (LDDT, MAE, dRMSD),
MarinFold's mean falls **outside** the
(`protenix_msa`, `protenix_single_seq`) interval — 1B is worse
than both Protenix modes on aggregate, not between them.

### Per-protein head-to-head

How many of the 100 proteins does 1B beat each Protenix mode on
each metric:

| metric | 1B beats protenix_single_seq | 1B beats protenix_msa |
|---|---:|---:|
| `lddt_distogram_cb` | 7 / 100 | 0 / 100 |
| `mae_distogram_cb_angstrom` | 5 / 100 | 0 / 100 |
| `drmsd_distogram_cb_angstrom` | 4 / 100 | 0 / 100 |
| `prec_long_L` | **33 / 100** | 1 / 100 |
| `prec_long_L_5` | **27 / 100** | 1 / 100 |

The picture isn't entirely flat: on **long-range contact precision**,
1B beats Protenix single-seq on ~30% of proteins, even though the
aggregate mean is still lower. The model's *ranking* of which
residue pairs are in contact is more competitive than its
*expected-distance accuracy* — consistent with the distogram head
being a softer, calibration-driven signal where MAE is dominated
by tail errors but contact precision sees the model produce a
correctly-ranked top-L set on a chunk of inputs.

### Timing

Modal H100 80GB, 100 proteins:

- Total GPU-time: **3.86 GPU-hours** (sum of per-protein elapsed).
- Wall-time on Modal: **~1h 15min** (Modal auto-scaled to ~3-5
  concurrent workers).
- Per-protein: mean 138.9 s, median 59.1 s, min 0.7 s (30 aa),
  max 1378.5 s (761 aa).
- Local A5000 (24 GB) extrapolation from 3-protein smoke: ~14
  GPU-hours sequentially. The full Modal run is ~3.6× faster on
  GPU-time alone, and the parallelism cuts wall time further.

Timing details per protein in [`data/timings.csv`](data/timings.csv).
The runtime-vs-length plot is
[`plots/timing_vs_sequence_length.png`](plots/timing_vs_sequence_length.png).

### Plots

- [`plots/headline_aggregate.png`](plots/headline_aggregate.png) —
  mean + median bar chart of the 3 headline metrics per method.
- [`plots/lddt_per_protein.png`](plots/lddt_per_protein.png) —
  per-protein grouped bars (100 proteins × 3 methods).
- [`plots/mae_per_protein.png`](plots/mae_per_protein.png) — same
  for MAE.
- [`plots/prec_long_L_per_protein.png`](plots/prec_long_L_per_protein.png)
  — same for CASP long-range contact precision @ L.
- [`plots/lddt_marinfold_vs_protenix_scatter.png`](plots/lddt_marinfold_vs_protenix_scatter.png) /
  [`plots/mae_marinfold_vs_protenix_scatter.png`](plots/mae_marinfold_vs_protenix_scatter.png) —
  paired scatters with `y = x` diagonal.
- [`plots/timing_vs_sequence_length.png`](plots/timing_vs_sequence_length.png) — per-protein wall-time vs `n_residues`, color by GPU.

### Non-parity notes

- **Structure-side metrics** (CA-RMSD, all-heavy RMSD, structure
  LDDT-CA/CB/all-heavy) are recorded for Protenix in
  `protenix_data/.../scores.csv` but are N/A for MarinFold 1B —
  the 1B model does not emit a structure in this eval. The
  merged `data/scores.csv` drops those columns to keep the schema
  consistent across methods.
- **Bin schemes** differ (Protenix v2: 64 bins from 2.31 to
  21.69 Å; MarinFold v1: 64 bins from 0 to 32 Å). All
  distance-based metrics use the **intersection range** as the
  pair filter (`[2.31, 21.69]`) and each model's own bin
  midpoints for its expected-distance / soft-LDDT computation.
- **Atom convention** is CB-CB on both sides, with CA fallback for
  GLY on either side AND for our reader's UNK (non-canonical
  residues at the entity_poly_seq level). Protenix uses CB-CB
  with CA-for-GLY by construction in its distogram head. The
  UNK→CA fallback is a small MarinFold-side concession; it
  affects a handful of designed peptides like 5sbj_A which have 4
  non-canonical positions.

## Conclusion

**MarinFold 1B underperforms Protenix v2 single-sequence on this
eval, not the "between single-seq and MSA" position the hypothesis
predicted.** On the 100 FoldBench monomers, scored with the same
distogram-derived metrics under exp12's CB-CB pair filter:

- `lDDT_distogram_cb` mean: **0.27 (1B)** vs 0.43 (`protenix_single_seq`) vs 0.91 (`protenix_msa`).
- `MAE_distogram_cb` mean: **5.64 Å (1B)** vs 2.69 Å (`protenix_single_seq`) vs 0.47 Å (`protenix_msa`).
- `dRMSD_distogram_cb` mean: **7.14 Å (1B)** vs 3.75 Å (`protenix_single_seq`) vs 0.79 Å (`protenix_msa`).

On per-protein head-to-head, 1B beats Protenix single-seq on
4–7 of 100 proteins for the headline distance/LDDT metrics, and
on 0 of 100 proteins for Protenix MSA. The hypothesis as stated
in the issue (≥2 of 3 headline metrics with 1B's mean strictly
between `protenix_msa` and `protenix_single_seq`) **is not
supported** by the data.

The one place 1B is more competitive is CASP-style long-range
contact precision: it beats Protenix single-seq on **33 / 100
proteins at top L** and **27 / 100 at top L/5**. So the model's
relative *ranking* of contact pairs is meaningfully informative on a
non-trivial fraction of inputs even when its absolute expected
distance is far off. That's a reasonable direction for follow-up
work (e.g. distill the contact-ranking signal into a stronger
distance prediction, or train a larger model with structure-
informed losses).

Where 1B sits: **below Protenix single-seq on aggregate, well
below Protenix-with-MSA on every protein, with a useful
contact-ranking signal on ~30% of inputs**.
