---
marinfold_experiment:
  issue: 26
  title: "exp: eval 1.5B model on 100 foldbench monomers"
  kind: evals
  branch: exp/26-marinfold-1_5b-foldbench-eval
---

# exp: eval 1.5B model on 100 foldbench monomers

**Issue:** [#26](https://github.com/Open-Athena/MarinFold/issues/26) · **Kind:** `evals` · **Branch:** `exp/26-marinfold-1_5b-foldbench-eval`

## Question

On the FoldBench monomer subset already collected in exp12 / PR #14, how does MarinFold's `1.5B` model compare to the `1B` baseline (from exp20 / PR #21), Protenix v2 single-sequence, and Protenix v2 with MSAs, when all four are scored on the same proteins with the same distogram-derived metrics?

## Hypothesis

`1.5B` will outperform `1B` on the headline distogram metrics (LDDT, MAE, dRMSD) and on long-range contact precision, narrowing the gap to Protenix single-seq but still well below Protenix+MSA. We will know whether the extra parameters at this training budget actually translate to better zero-shot distogram quality on FoldBench monomers.

## Background

This is a direct rerun of exp20 ([PR #21](https://github.com/Open-Athena/MarinFold/pull/21)) with the new `1.5B` checkpoint substituted for `1B`. The `1B` aggregate results from that run, for context:

| metric | marinfold_1b | protenix_single_seq | protenix_msa |
|---|---:|---:|---:|
| `lddt_distogram_cb` | **0.271** | 0.432 | 0.912 |
| `mae_distogram_cb_angstrom` (Å) | **5.64** | 2.69 | 0.47 |
| `drmsd_distogram_cb_angstrom` (Å) | **7.14** | 3.75 | 0.79 |
| `prec_long_L` | **0.258** | 0.373 | 0.913 |

So 1B sits below Protenix single-seq on aggregate, with a slightly more competitive ranking signal on long-range contacts (~33/100 proteins beat Protenix single-seq there). This experiment asks whether 1.5B is enough capacity to move us into (or past) the Protenix single-seq band.

Relevant prior artifacts:

- Issue #20 + PR #21: the 1B run this mirrors (re-use the launcher / scoring code with `--model 1.5B`)
- Issue #12 + PR #14 / `experiments/exp12_data_protenix_foldbench_monomers/`: GT CIF handling, Protenix scoring logic, the canonical metric conventions
- `experiments/exp9_evals_test_distance_heatmaps/`: prior zero-shot eval surface on held-out proteins
- PR #25 (merged): adds `1.5B` to `MODELS.yaml` and extends `marinfold.registry` to handle the `huggingface.co/buckets/...` URL shape, so `marinfold infer --model 1.5B` resolves by nickname.

The 1.5B checkpoint lives at
`https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999`.

Important metric note carried over from exp12 / exp20: Protenix's distogram range is narrower than MarinFold's (`~2.31 Å` to `~21.84 Å`), so the fair comparison uses the intersection of the models' expressible distance range with Protenix's upper bound defining the cap. Same convention here.

## Approach

### Inputs

- **FoldBench monomer set**: reuse exp12's 100-protein paired set; no
  Protenix re-runs. `fetch_protenix_data.py` pulls only what we need
  from the `open-athena/MarinFold` HF bucket (~26 MB of GT mmCIFs +
  small CSVs).
- **Model**: `1.5B` from `MODELS.yaml`, resolved + snapshot-downloaded
  via `marinfold.registry` (PR #25 wires the bucket URL shape).

### Compute: iris on TRC (not Modal)

This experiment runs on TRC via marin's iris orchestrator, matching
the marin protein-eval convention (see
`https://github.com/marin-community/marin/tree/protein-training-1b/experiments/protein`).
The iris launch shape is

```
uv run iris ... job run \
    --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 \
    --extra=vllm --extra=tpu \
    -- python run_eval.py --model 1.5B --output-gcs <prefix>
```

`run_eval.py` is a TPU-flavored derivative of exp20's
`run_1b_eval.py`: loads the model once, loops over all 100
proteins, writes `<stem>/{distogram.npz, provenance.json}` to GCS
on the fly so partial progress survives preemption.

**GCS layout** (per the AGENTS.md "GCS bucket" convention):

```
gs://marin-us-east5/protein-structure/MarinFold/exp26/protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers/
    <stem>/
        distogram.npz
        provenance.json
```

The local `outputs/` directory is populated by `gsutil -m rsync`
from that prefix before scoring runs.

### Atom convention

CB-CB queries with CA fallback for any GLY (and our reader's UNK —
the entity_poly_seq's non-canonical residues map to UNK and we
fall back to CA there too, since CB conventions are undefined for
non-standard side chains). Matches Protenix's distogram
representative-atom convention exactly, so the cross-model
comparison is apples-to-apples. Same as exp20.

### Inference (per protein)

Same body as exp20:

1. Read the canonical 1..N residue sequence from the Protenix GT
   mmCIF (`canonical_sequence.py` handles the biological-assembly
   multi-subchain case, maps non-canonical residues to UNK).
2. Build the zero-shot v1 prompt
   `<contacts-and-distances-v1> <begin_sequence> <AAs…> <begin_statements>`.
3. For all (i, j) with i < j, query
   `<distance> <p_i> <p_j> <CB|CA> <CB|CA>` with vLLM's prefix
   cache (one trunk forward, N²/2 tails).
4. Renormalize the top-128 logprobs over the 64 `<d_X.X>` bin
   tokens to a probability vector. Save symmetric `[N, N, 64]` to
   `outputs/{stem}/distogram.npz`.
5. Each protein also writes a `provenance.json` with
   `elapsed_seconds`, `n_residues`, `n_pairs`, batch size, bin
   scheme, and a `hardware` block (`gpu_name`,
   `gpu_total_memory_gb`, `runner_tag="iris"`, hostname, torch
   version). `collect_timings.py` rolls these into
   `data/timings.csv`.

Idempotent: re-running skips proteins whose `provenance.json`
matches `model_nickname: 1.5B` already.

### Scoring

`score_marinfold.py` is unchanged from exp20 — already
parameterized by `BinScheme`, so MarinFold's 0.5 Å bins (midpoints
0.25..31.75 Å) plug into the same metric functions Protenix uses.
The **in-range MAE/dRMSD pair filter uses Protenix's narrower
range** (`[2.3125, 21.6875]`).

`score_comparison.py` extends exp20's three-way join to four
methods:

- `marinfold_1_5b` (this run)
- `marinfold_1b` (joined in from `../exp20_evals_marinfold_1b_foldbench/data/scores.csv`)
- `protenix_single_seq` (from `../exp12_data_protenix_foldbench_monomers/data/scores.csv`)
- `protenix_msa` (same source)

Output is a 400-row `data/scores.csv` (100 proteins × 4 methods)
and per-method aggregates in `data/scores_summary.csv`.

### Hypothesis verdict

The issue counts the hypothesis as supported iff `1.5B` beats
`1B` on aggregate on at least **3 of the 4** headline metrics:
`lddt_distogram_cb` (↑), `mae_distogram_cb_angstrom` (↓),
`drmsd_distogram_cb_angstrom` (↓), `prec_long_L` (↑).
`score_comparison.py` computes this and writes
`data/hypothesis_verdict.json`.

### Files

- `fetch_protenix_data.py` — pull Protenix scores + GT CIFs from HF (unchanged from exp20).
- `canonical_sequence.py` — read the 1..N entity_poly_seq from a Protenix biological-assembly GT mmCIF (unchanged).
- `run_eval.py` — iris/TPU vLLM driver (derived from exp20's `run_1b_eval.py`).
- `launch_iris.sh` — wrapper around `uv run iris … job run … -- python run_eval.py …`.
- `score_marinfold.py` — MarinFold-side per-protein scoring (unchanged).
- `score_comparison.py` — 4-way merge + verdict.
- `collect_timings.py` — aggregate `outputs/*/provenance.json` → `data/timings.csv` (unchanged).
- `plot_comparison.py` — 4-way per-protein + aggregate plots.
- `tests/test_smoke.py` — bin scheme, contact mask, CIF parser (unchanged).

Zero-shot only — no seeded contacts, no hints.

## Running this experiment

### Prerequisites

- `uv` (Python toolchain).
- iris CLI authenticated for TRC (`uv run iris … job list` should
  return successfully).
- `gsutil` authenticated to read/write
  `gs://marin-us-east5/protein-structure/MarinFold/`.

### One-time setup

```bash
cd experiments/exp26_evals_marinfold_1_5b_foldbench
uv sync
# Fetch Protenix scores + GT mmCIFs from the open-athena/MarinFold
# HF bucket (ephemeral env, same as exp20).
uv run --with "huggingface_hub>=1.5" python fetch_protenix_data.py
```

### Inference

Launch the iris job:

```bash
./launch_iris.sh
```

The job loads `1.5B`, loops the 100 proteins, and writes each
result to GCS as it goes. Pull outputs back locally before
scoring:

```bash
gsutil -m rsync -r \
    gs://marin-us-east5/protein-structure/MarinFold/exp26/protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers/ \
    outputs/
```

`run_eval.py` is idempotent on `model_nickname`, so it's safe to
re-launch after a preemption.

### Score + plot

```bash
uv run python score_marinfold.py     # writes data/marinfold_scores.csv (method=marinfold_1_5b)
uv run python collect_timings.py     # writes data/timings.csv
uv run python score_comparison.py    # writes data/scores.csv + data/scores_summary.csv + data/hypothesis_verdict.json
uv run python plot_comparison.py     # writes plots/*.png
```

### Smoke tests

```bash
uv run --extra test pytest tests/
```

## Success criteria

1. Paired four-way comparison for the same FoldBench proteins from exp20 / exp12, one row per `(protein, method)` for `marinfold_1b`, `marinfold_1_5b`, `protenix_single_seq`, `protenix_msa`.
2. Committed summary artifacts:
   - at least one CSV with per-protein comparison metrics
   - at least one plot showing the four-way comparison
3. Headline metrics computed with the same scoring convention across all four methods.
4. Hypothesis counts as supported if `1.5B` beats `1B` on aggregate on at least 3 of the 4 headline metrics (`lddt_distogram_cb`, `mae_distogram_cb_angstrom`, `drmsd_distogram_cb_angstrom`, `prec_long_L`).
5. The README / final issue comment ends with a direct answer to "did 1.5B move us closer to Protenix, and by how much?"

## Results

100 / 100 proteins ran cleanly on v5p-8 via iris in a single ~4 h job.
Per-protein outputs live under
`gs://marin-us-east5/protein-structure/MarinFold/exp26/protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers/<stem>/{distogram.npz, provenance.json}`.

### Aggregate (mean across the 100 monomers)

| metric | marinfold_1_5b | marinfold_1b | protenix_single_seq | protenix_msa |
|---|---:|---:|---:|---:|
| `lddt_distogram_cb` ↑ | **0.288** | 0.272 | 0.432 | 0.912 |
| `mae_distogram_cb_angstrom` (Å) ↓ | **5.695** | 5.642 | 2.685 | 0.468 |
| `drmsd_distogram_cb_angstrom` (Å) ↓ | **7.202** | 7.143 | 3.750 | 0.787 |
| `prec_long_L` ↑ | **0.285** | 0.258 | 0.373 | 0.913 |

The 1.5B vs 1B deltas are small in both directions:
LDDT +0.016, prec_long_L +0.026, MAE +0.053 Å (worse), dRMSD +0.059 Å
(worse). Hypothesis-verdict counter (from
`data/hypothesis_verdict.json`) is **2 / 4 headline metrics support**,
which is below the 3 / 4 bar — so the issue's pre-registered
hypothesis is **not supported**.

### Per-protein head-to-head

The aggregate hides a sharper picture: on metrics that pool error
within a single bin (LDDT, contact precision) 1.5B wins on the
majority of proteins, but on the per-pair Å-distance metrics (MAE,
dRMSD) it loses on most.

| metric | 1.5B beats 1B | 1.5B beats Protenix-SS | 1.5B beats Protenix-MSA |
|---|---:|---:|---:|
| `lddt_distogram_cb` ↑ | 74 / 100 | 11 / 100 | 0 / 100 |
| `prec_long_L` ↑ | 70 / 100 | 33 / 100 | 1 / 100 |
| `mae_distogram_cb_angstrom` ↓ | 40 / 100 | 6 / 100 | 0 / 100 |
| `drmsd_distogram_cb_angstrom` ↓ | 38 / 100 | 6 / 100 | 0 / 100 |

vs Protenix-MSA, MarinFold 1.5B beats it on a single protein, on a
single metric (`prec_long_L`).

### Inference cost

Per-protein wall-time vs sequence length on log-log is in
`plots/timing_4way_vs_sequence_length.png`. Hardware caveat: 1.5B
ran on TPU v5p-8 via iris; 1B ran on H100 via Modal (exp20);
Protenix ran on H100 (exp12). Apples-to-apples the 1.5B pair sweep
on v5p-8 is roughly an order of magnitude slower per protein than
the 1B sweep on H100, but both show the same clean O(N²) scaling
shape; this is a hardware mix story, not an algorithmic regression.

### Artifacts

- `data/marinfold_scores.csv` — 100 per-protein 1.5B scores.
- `data/timings.csv` — 100 per-protein wall-times (iris/TPU).
- `data/scores.csv` — 400-row 4-way merge.
- `data/scores_summary.csv` — per-method aggregates (table above).
- `data/hypothesis_verdict.json` — programmatic 3/4 verdict.
- `plots/headline_aggregate.png`,
  `plots/lddt_5way_swarm.png`,
  `plots/lddt_marinfold_vs_protenix_scatter.png` (anchored on 1.5B),
  `plots/timing_4way_vs_sequence_length.png`, et al.

## Conclusion

**Did 1.5B move us closer to Protenix?** Slightly, in the same shape
1B already had — better on bin-pooled metrics (LDDT +1.6 pp,
prec_long_L +2.6 pp), no movement (in fact a small regression) on
the Å-distance metrics (MAE +0.05 Å, dRMSD +0.06 Å). The gap to
Protenix-single-seq is still wide on every metric (LDDT 0.288 vs
0.432) and to Protenix-MSA it remains ≈3×. On the pre-registered
3-of-4 bar, the hypothesis is **not supported** (2 / 4).

So the headline read for the issue: scaling 1B → 1.5B at the
current training budget gives a small but real bump on contact-class
metrics, no gain on the metric you'd actually use for downstream
structure work (per-pair Å error), and does not narrow the gap to
Protenix-single-seq in any meaningful way. The natural next move is
not "scale params again", it's the training-side levers (more
tokens, distance loss reweighting, MSA conditioning).
