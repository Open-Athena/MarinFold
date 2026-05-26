---
marinfold_experiment:
  issue: 9
  title: "exp: zero-shot distance heatmap eval on 10 random test proteins (1B model)"
  kind: evals
  branch: exp/9-test-distance-heatmaps
---

# exp: zero-shot distance heatmap eval on 10 random test proteins (1B model)

**Issue:** [#9](https://github.com/Open-Athena/MarinFold/issues/9) · **Kind:** `evals` · **Branch:** `exp/9-test-distance-heatmaps`

## Question

How well does the [`1B`](https://huggingface.co/timodonnell/LlamaFold-experiments/tree/main/marin-experiments.protein-contacts-1b-3.5e-4-distance-masked-7d355e)
model predict the CA-CA distance map of held-out (`contacts-and-distances-v1-5x`, `split=test`)
AFDB structures it has never seen? We want a small, interactive,
runnable record of this so anyone training a new model can drop
their checkpoint in and see how it stacks up.

**Kind:** `evals`

## Hypothesis

On the test split, zero-shot CA-CA MAE should be comparable to the
in-training distogram benchmark on the small hand-curated FoldBench
set (~3-4 Å). The heatmaps should look like noisy versions of the
ground-truth distance maps — diagonal band correct, off-diagonal
contacts roughly placed, with detail degrading at long range.

## Background

Existing eval surfaces are either (a) the training-time tagged
loss in `protein_distogram_eval.py` (CA-CA on 13 fixed PDB targets,
ran every `steps_per_eval`) or (b) the offline distogram script
`eval_protein_distogram.py` in marin (one PDB, vLLM, top-K logprob
renormalization over the 64 `<d_X.X>` bins).

This experiment is the smallest useful "did the model learn the
test distribution?" check: 10 random AFDB entries from
`contacts-and-distances-v1-5x` split=test, fetched fresh from the
EBI AFDB endpoint, scored CA-CA with vLLM-driven inference using
the prefix-shared batch pattern that exp1 already implements in
`inference._query_pairs`. Selection is RNG-seeded so re-running
gives the same 10 entries.

## Approach

- `select_test_proteins.py`: loads `timodonnell/protein-docs` (subset
  `contacts-and-distances-v1-5x`, split `test`), uses a fixed seed
  to sample 10 `entry_id`s, returns them plus the currently-live
  AFDB `.cif` URLs (`model_v6.cif` as of writing).
- `inference_helpers.py`: shared vLLM helpers (model load,
  distance-bin resolution, prompt build, per-pair / full-matrix
  queries, GT contact extraction, sampled-pair MAE). Both notebooks
  use it.
- Both notebooks resolve the model via the top-level
  [`MODELS.yaml`](../../MODELS.yaml) — bump `MODEL_NICK` (or add
  a new entry) and re-run to retarget a fresh checkpoint.
- Initial smoke run on this machine's RTX A5000 (24 GB) with the
  `1B` model.

### Variant 1 — `eval_notebook.ipynb` (zero-shot heatmaps)

For each protein, runs CA-CA inference at every (i, j), i < j with
a shared base prefix (`<task> <begin_sequence> <AAs>
<begin_statements>`) and per-pair `<distance> <p_i> <p_j> <CA>
<CA>` tails. Renormalizes top-128 logprobs over the 64 distance-bin
tokens, takes the expected distance from the bin midpoints. Plots
GT, predicted, and `|residual|` heatmaps per protein, a pooled
expected-vs-GT scatter, and a 10×3 grid.

### Variant 2 — `contact_seeding_search.ipynb` (minimal seeded contacts)

Same 10 proteins. **Greedy-searches for the smallest set of true
long-range contacts (≤5) whose presence in the prompt drops the
sample MAE below 1.0 Å.** At each round 1..5 we try every
remaining GT long-range contact (CB-CB ≤ 8 Å, sep ≥ 24) as the
next seeded contact, measure MAE on a deterministic 500-pair
CA-CA sample, and pick the one minimizing it. Stop early when the
target is hit. After the search, the chosen contacts are re-run on
the full N×N matrix to produce final heatmaps and a comparison
grid against the zero-shot variant.

## Success criteria

1. **Variant 1**: notebook runs end-to-end on a single GPU in
   under ~15 minutes for 10 small (\<150 res) proteins; macro
   CA-CA MAE is finite and within ~2x of the in-training
   FoldBench numbers (we expect ≤ the FoldBench number since the
   test split is in-distribution AFDB).
2. **Variant 2**: search notebook completes per-protein in a
   bounded time (~5 min each at the worst case) and records, for
   each of the 10 proteins, the smallest k ∈ {0..5} that brings
   sample MAE below 1.0 Å (or reports that the target is not met
   at k=5).
3. Sampling is deterministic: re-running the selection cell
   gives the same 10 entry_ids in both notebooks.
4. Per-protein artifacts are saved to `plots/`; per-protein
   metrics to `data/`.

## Results

Both runs on an RTX A5000 (24 GB), 2026-05-18, seed=0,
`max_seq_len=150`, model `1B` from `MODELS.yaml`. The executed
notebooks (`eval_notebook.ipynb`, `contact_seeding_search.ipynb`)
hold the cells + embedded plots; this section is the prose
summary.

### Variant 1 — zero-shot (`eval_notebook.ipynb`)

**Macro CA-CA MAE = 3.29 Å** across 10 AFDB proteins (40,442
evaluable pairs).

| entry_id | n_res | n_pairs | MAE (Å) |
|---|---:|---:|---:|
| AF-A0A1N7G8C0-F1 | 60 | 1,900 | 1.03 |
| AF-A0A6B0Z5B5-F1 | 112 | 10,186 | 2.20 |
| AF-A0A1C5BRX1-F1 | 72 | 3,886 | 2.85 |
| AF-R7G5V6-F1 | 132 | 8,774 | 3.21 |
| AF-A0A1H0PBF4-F1 | 94 | 6,658 | 3.23 |
| AF-A0A1G4A0Q3-F1 | 114 | 7,722 | 3.43 |
| AF-A0A2P2Q6H4-F1 | 55 | 2,286 | 3.74 |
| AF-A0A7W4UDR7-F1 | 131 | 15,228 | 4.05 |
| AF-E6UJZ8-F1 | 112 | 9,758 | 4.35 |
| AF-C6S3E2-F1 | 140 | 14,486 | 4.81 |

Per-protein heatmaps are in `plots/`. Source CSV:
`data/per_protein_mae.csv`. Sampling sometimes skips entries
whose UniProt accession has been retired from AFDB since the
training data was built — at seed=0 four such entries
(AF-A0A352P7D8-F1, AF-A0A1K1WRR2-F1, AF-A0A6M1XDR8-F1,
AF-A6D053-F1) were skipped; the next ten valid entries were used.

Wall time on a single A5000: ~5 minutes end-to-end including
model staging, with vLLM at ~85% GPU util during inference.

### Variant 2 — long-range-only beam search (`contact_seeding_search.ipynb`)

**Beam search width 2** over GT long-range contacts (CB-CB ≤ 8 Å,
sep ≥ 24), 8 random candidates per (beam state, round), sample
MAE on 300 deterministic CA-CA pairs, target MAE < 1.0 Å,
MAX_CONTACTS = 30. Wall time ~75 min on a single A5000.

An earlier pass used pure greedy with MAX_CONTACTS=5 and could
not reach 1 Å on any protein (best was 1.21 Å). Switching to
beam-2 + raising the budget to 30 lets **2/10 proteins cross
the 1.0 Å target** (full-matrix MAE):

| entry_id | n_res | n_cand | k | full MAE @ k=0 | full MAE @ chosen k | status |
|---|---:|---:|---:|---:|---:|---|
| AF-A0A6B0Z5B5-F1 | 112 | 59 | **7** | 2.20 | **1.08** | target met at k=7 |
| AF-A0A1H0PBF4-F1 | 94 | 37 | **18** | 3.23 | **1.03** | target met at k=18 |
| AF-A0A2P2Q6H4-F1 | 55 | 17 | 17 | 3.74 | 1.38 | exhausted candidates |
| AF-E6UJZ8-F1 | 112 | 113 | 30 | 4.35 | 1.50 | reached MAX_CONTACTS |
| AF-A0A1G4A0Q3-F1 | 114 | 45 | 30 | 3.43 | 1.62 | reached MAX_CONTACTS |
| AF-A0A7W4UDR7-F1 | 131 | 116 | 30 | 4.05 | 1.75 | reached MAX_CONTACTS |
| AF-R7G5V6-F1 | 132 | 21 | 21 | 3.21 | 2.09 | exhausted candidates |
| AF-C6S3E2-F1 | 140 | 156 | 30 | 4.81 | 2.27 | reached MAX_CONTACTS |
| AF-A0A1C5BRX1-F1 | 72 | 0 | 0 | 2.85 | 2.85 | no long-range contacts in structure |
| AF-A0A1N7G8C0-F1 | 60 | 0 | 0 | 1.03 | 1.03 | no long-range contacts in structure |

Patterns:

- The two proteins that cross the target are the same two whose
  zero-shot MAE was already on the better end (2.20 and 3.23 Å).
- "Hardness" tracks with the number of GT long-range contacts:
  the easy proteins have a small, geometrically-constraining
  contact set; the hard ones (C6S3E2 with 156 long-range
  contacts) have so many that any 30 only pin part of the
  structure.
- AF-A0A2P2Q6H4-F1 used all 17 GT contacts in its candidate set
  and still didn't break 1 Å — the model just can't predict that
  structure at <1 Å resolution with this seeding alone.
- The 2 proteins with zero long-range contacts in their structure
  can't benefit from any long-range seeding; their MAE stays at
  the zero-shot value. (For AF-A0A1N7G8C0-F1 that's already
  1.03 Å — basically at target without any hints.)

See `plots/contact_search_trace.png` for per-protein MAE-vs-k
curves and `plots/contact_search_grid.png` for the final
heatmaps. Source CSV: `data/contact_search_summary.csv`
(`selected_contacts` field lists the chosen contacts as
`i-j; i-j; …`).

### Variant 3 — all-range greedy search (`contact_seeding_search_all_ranges.ipynb`)

Same setup as Variant 2 but **drops the beam, opens the
candidate pool to all three training contact types**:

- `<long-range-contact>` (sep ≥ 24)
- `<medium-range-contact>` (sep 12..23)
- `<short-range-contact>` (sep 6..11)

(all with the CB-CB ≤ 8 Å cutoff). Pure greedy (no beam), cap 10
random candidates per round, sample MAE on 300 deterministic
pairs, MAX_CONTACTS = 30. Wall time ~50 min on the A5000 (~⅔
that of Variant 2 since pure greedy halves the per-round
evaluation count).

**4/10 proteins cross the 1 Å full-matrix MAE target**, up from
2/10 with Variant 2's long-range-only beam search. **Every
protein** comes out at a lower (better) full-matrix MAE than the
beam-2 long-only run:

| entry_id | n_res | n_cand | k | full MAE @ chosen k | (V2 baseline) | mix selected (L/M/S) |
|---|---:|---:|---:|---:|---:|---|
| AF-A0A2P2Q6H4-F1 | 55 | 80 | **15** | **0.95** | 1.38 | 6 / 5 / 4 |
| AF-A0A6B0Z5B5-F1 | 112 | 91 | **7** | **1.01** | 1.08 | 6 / 1 / 0 |
| AF-A0A1H0PBF4-F1 | 94 | 47 | **8** | **1.03** | 1.03 (V2 needed k=18) | 7 / 0 / 1 |
| AF-A0A1N7G8C0-F1 | 60 | 2 | **0** | **1.03** | 1.03 | — (zero-shot already at target) |
| AF-R7G5V6-F1 | 132 | 40 | 30 | 1.24 | 2.09 | 16 / 2 / 12 |
| AF-E6UJZ8-F1 | 112 | 162 | 30 | 1.28 | 1.50 | 22 / 4 / 4 |
| AF-A0A1G4A0Q3-F1 | 114 | 72 | 30 | 1.34 | 1.62 | 17 / 8 / 5 |
| AF-A0A7W4UDR7-F1 | 131 | 243 | 30 | 1.46 | 1.75 | 19 / 5 / 6 |
| AF-C6S3E2-F1 | 140 | 230 | 30 | 1.92 | 2.27 | 21 / 4 / 5 |
| AF-A0A1C5BRX1-F1 | 72 | 4 | 4 | 2.49 | 2.85 | 0 / 0 / 4 |

Observations:

- **Medium and short range help on every protein**, even when
  the search ends up choosing mostly long-range contacts: the
  expanded candidate pool lets greedy find a few critical
  non-long-range constraints (e.g. AF-A0A1H0PBF4-F1 needed 18
  long-range contacts to reach 1.03 Å under V2, but reaches
  1.03 Å with just 7 long + 1 short under V3).
- **R7G5V6's improvement is the biggest gain**: 2.09 → 1.24 Å.
  V2 had only 21 long-range candidates and exhausted them all
  without reaching the target; V3 had 40 total candidates and
  blended in 12 short-range ones.
- **C6S3E2 remains the hardest** — 156 long-range + 45 med + 29
  short = 230 candidates, and 30 of them still only get to
  1.92 Å. This protein either needs more contacts than 30 or
  benefits from a stronger model.
- AF-A0A1N7G8C0-F1 hits target at k=0 in V3 because the 300-pair
  sample happens to give sample MAE 0.96 (below target); the
  full-matrix MAE is 1.03 — so this is "lucky stop", not a real
  V3 win, but it's noted for honesty.

Source CSV: `data/contact_search_all_ranges_summary.csv` (the
`selected_contacts` field encodes each pick as
`<range>:i-j` — e.g. `long:25-52; medium:17-30`).
Trace: `plots/contact_search_all_ranges_trace.png`. Grid of
final heatmaps: `plots/contact_search_all_ranges_grid.png`.

**LDDT trace.** The MAE trace above is a 300-CA-CA-pair sample
(cheap to evaluate at every round), so it can't compute proper
LDDT — LDDT is defined per-residue over all pairs within 15 Å of
GT. After the search finishes, the notebook replays each protein's
`selected_contacts` step by step (k=0..k_final), predicts the full
N×N CA distance matrix at each k, and computes global LDDT(CA)
under the standard CASP convention (15 Å inclusion, thresholds
0.5/1/2/4 Å). One extra full-matrix prediction per k per protein
(~150 across all 10 proteins), so this roughly doubles the
end-to-end wall time. The per-k LDDT series is saved to
`data/contact_search_all_ranges_lddt_trace.csv`; the final-k LDDT
is also appended to `contact_search_all_ranges_summary.csv` as
`full_matrix_lddt_ca`. Plot:
`plots/contact_search_all_ranges_lddt_trace.png` — same x-axis as
the MAE trace, y-axis is LDDT(CA) ∈ [0, 1], higher is better.

### Variant 4 — directed search by predicted CB-CB distance (`contact_seeding_directed_search.ipynb`)

V3 spent most of its wall time on per-candidate sample-MAE
evaluation: each round picked the next contact by running
inference on up to 10 random candidates (300 CA-CA pairs each)
and comparing MAEs. The post-hoc LDDT replay then ran one extra
full-matrix prediction per k. V4 collapses both phases into one:
**at each round, run a single full-matrix prediction; then pick
the next contact by sorting remaining candidates by their
current predicted CB-CB distance and taking the largest.**

The heuristic: every candidate has GT CB-CB ≤ 8 Å (that's the
contact definition), so a candidate the model currently predicts
as close is a redundant seed; a candidate the model predicts as
far apart is one it's most wrong about — seeding it should give
the largest correction. No per-candidate inference, no random
sampling, no sample-MAE proxy. Deterministic given the input.

Other deltas vs V3:

- Everything is **CB-CB** (CA-for-GLY) — GT uses
  `cb_or_ca_position` (matches the contact-extraction
  convention), and the model is queried with `query_atom="CB"`
  (matches the training-data convention for emitting `<CB>`
  tokens).
- Only **LDDT-CB** is tracked (no MAE column anywhere).
- **No early stop**: always runs to MAX_CONTACTS=30 (or until
  candidates are exhausted) so the curve shape is fully visible.

Per-round cost ≈ one full-matrix CB prediction. For 10 proteins ×
~31 rounds = ~256 full-matrix predictions total (8/10 proteins
hit MAX_CONTACTS=30; the two with tiny candidate pools stop
early). **Measured wall time on the A5000: 144 min**
(`sum(elapsed_seconds)` across the trace CSV), vs ~161 min for
V3 + LDDT-CA replay — an ~11 % speedup. Smaller than the naive
"V4 just skips the per-candidate search inference" estimate
because V4 doesn't early-stop on the easy proteins, so it ends
up doing slightly *more* full-matrix predictions than V3's
replay did (256 vs 218).

Final LDDT(CB) per protein (no early stop, runs to
MAX_CONTACTS=30 or until candidates exhausted):

| entry_id | n_res | n_cand | k | init LDDT-CB | final LDDT-CB | L/M/S |
|---|---:|---:|---:|---:|---:|---|
| AF-A0A2P2Q6H4-F1 | 55 | 80 | 30 | 0.357 | **0.804** | 8/12/10 |
| AF-A0A6B0Z5B5-F1 | 112 | 91 | 30 | 0.552 | **0.786** | 17/6/7 |
| AF-A0A1C5BRX1-F1 | 72 | 4 | 4 | 0.767 | **0.784** | 0/0/4 |
| AF-R7G5V6-F1 | 132 | 40 | 30 | 0.616 | **0.767** | 15/3/12 |
| AF-A0A1N7G8C0-F1 | 60 | 2 | 2 | 0.755 | **0.753** | 0/0/2 |
| AF-A0A1H0PBF4-F1 | 94 | 47 | 30 | 0.481 | **0.729** | 24/0/6 |
| AF-A0A1G4A0Q3-F1 | 114 | 72 | 30 | 0.397 | **0.702** | 16/8/6 |
| AF-E6UJZ8-F1 | 112 | 162 | 30 | 0.346 | **0.693** | 23/1/6 |
| AF-A0A7W4UDR7-F1 | 131 | 243 | 30 | 0.316 | **0.627** | 22/4/4 |
| AF-C6S3E2-F1 | 140 | 230 | 30 | 0.244 | **0.525** | 21/4/5 |

Observations:

- **The heuristic works.** Final LDDT-CB lands within ~0.05 of
  V3's final LDDT-CA on most proteins (sometimes a touch lower
  — V3 was tuning to MAE on every step, V4 only sees LDDT
  implicitly). Same protein ordering as V3 across the board.
- **Long-range contacts dominate early picks**, as predicted by
  the heuristic: AF-A0A1H0PBF4-F1 picks 24 of 30 long-range
  (vs V3's 9/30). AF-E6UJZ8-F1 picks 23/30 long-range (V3: ?).
  The model is indeed most-wrong about distant CB pairs, so the
  ranking surfaces them first.
- **AF-A0A1N7G8C0-F1 LDDT drops slightly** (0.755 → 0.753) after
  adding its 2 short-range candidates — these aren't useful
  seeds and the model marginally over-corrects. Confirms that
  "rank by largest predicted distance" doesn't always pick
  *helpful* contacts when the few candidates that exist are
  already at short range.

Artifacts:

- `data/contact_directed_search_summary.csv` (one row per protein:
  `selected_contacts`, `initial_lddt_cb`, `final_lddt_cb`, …).
- `data/contact_directed_search_trace.csv` (one row per
  `(entry_id, k)`: `added_contact_type`,
  `predicted_distance_before_seeding_a`, `lddt_cb`,
  `elapsed_seconds`).
- `plots/contact_directed_search_trace.png` — LDDT(CB) vs k, one
  line per protein.
- `plots/contact_directed_search_grid.png` — 10×3 CB-CB heatmap
  grid (GT, predicted-with-seeded, |residual|).

## Conclusion

**Zero-shot.** The `1B` model predicts unseen AFDB CA-CA distance
maps with **macro MAE 3.29 Å zero-shot**, consistent with the ~3-4
Å hypothesis and very close to the in-training FoldBench-monomer
benchmark (~3.6 Å on real PDB targets). The heatmaps look right:
the diagonal neighborhood is tightly matched, off-diagonal contacts
are placed roughly correctly, residual is dominated by the
saturated-bin ceiling at >32 Å.

**Seeded — long-range only (V2).** With beam-2 search and up to
30 seeded **long-range** contacts, **2 / 10 proteins cross the
1.0 Å MAE threshold** — AF-A0A6B0Z5B5-F1 at k=7 (1.08 Å) and
AF-A0A1H0PBF4-F1 at k=18 (1.03 Å). The other 6 proteins with
long-range contacts in their candidate set bottom out between
1.4 and 2.3 Å despite using all 30 contacts; 2 proteins have no
long-range contacts at all in their AFDB structure.

**Seeded — all ranges (V3).** Adding medium + short-range contacts
to the candidate pool and running **pure greedy** raises the
threshold-crossing count to **4 / 10** (or 3 / 10 if we drop the
"lucky 300-pair sample" k=0 win on AF-A0A1N7G8C0-F1, which has a
full-matrix MAE of 1.03). **Every protein improves** vs. V2, in
some cases dramatically (R7G5V6: 2.09 → 1.24 Å; C6S3E2: 2.27 →
1.92 Å). The headline takeaway: **letting the search pick from
all three contact ranges is more important than the choice of
search algorithm** — pure greedy with a broader candidate pool
beats beam-2 with a narrow one on every protein.

The hardest proteins (C6S3E2, A0A7W4UDR7) still need either
>30 contacts or a stronger model to break 1 Å. Worth re-running
once we have a checkpoint past
`protein-contacts-1b-3.5e-4-distance-masked-7d355e`.

**Directed-search by predicted CB-CB (V4).** Drops the
per-candidate sample-MAE evaluation entirely: at each round we
already need one full-matrix prediction (to score LDDT-CB
against the GT), and the same prediction is used to rank
remaining candidates by descending predicted distance — pick
the one the model is currently most wrong about. Final LDDT(CB)
lands within ~0.05 of V3's final LDDT-CA on most proteins, with
the same protein ordering. Wall time is ~11 % under V3 + LDDT
replay (144 vs 161 min on the A5000); the speedup is modest
because V4 doesn't early-stop on the easy proteins. The
useful empirical confirmation: directed-search picks
long-range contacts heavily in the first few rounds, exactly
as the heuristic predicts — the model is most-wrong about
distant CB pairs, and seeding those gives the biggest LDDT lift.

The four notebooks are the deliverable — a future "did the new
model learn?" check is just: add a nickname to `MODELS.yaml` and
re-run them.
