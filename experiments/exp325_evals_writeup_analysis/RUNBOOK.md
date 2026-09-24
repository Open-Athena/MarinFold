# Reproduce or restyle

From this experiment directory, install the locked lightweight analysis environment:

```bash
uv sync --locked
uv run python prepare.py
uv run python render.py
uv run python build_summary.py
uv run pytest -q
```

`prepare.py` runs joins, complete-case selection and uncertainty once. It reads
committed per-protein results and a pinned MSA snapshot; no GPU or credentials.
Afterwards, changing `theme.py` or `render.py` requires only the last rendering
and PDF commands. `render.py` verifies prepared hashes and does no network I/O.
Open `site/index.html` for an offline interactive preview. `plotly.min.js` is
generated locally from the locked Plotly package and ignored by git.

The static deliverables are `plots/*.svg`, `plots/*.png` and `plots/summary.pdf`.
The PDF uses five `02c_accuracy_confidence_protein_<stem>.png` pages, one per
protein with all four accuracy views. The combined website scatter exports have
`include_in_summary: false` in their sidecars, so they are replaced by the
individual pages in the PDF. `render.py` writes these sidecars automatically.
Plotly assets are `site/*.json`, including mobile variants. The house palette and
Lato font come from Open Athena's site at commit
`0618887cbd74bc89d0e4575716d262cb1c6c179d`. The eventual integration is
`DRAFT.md` → `content/blog/` and figure JSON →
`static/assets/images/blog/marinfold-single-sequence-contacts/`. The site's native
`plotly` shortcode accepts these data/layout/config envelopes. No site changes,
deployment or pull request are part of this branch.

# Regenerate predictions (expensive; not needed for styling)

The fixed checkpoint, targets and controls live in `generation/` and
`data/*inputs.json`. The run needs Modal access to the existing Helico checkpoint
and CCD volumes. The Helico checkout must be at
`b10385d736673c81b10e70d1099962af6f2573c0`. Test reads are publication-only and
recorded in exp245's `eval_test_reads.md`; do not use this launcher for selection.

```bash
uv run --project generation modal run generation/run_contacts.py
uv run --project generation python generation/score_contacts.py
uv run --project generation python generation/prepare_helico.py --phase confidence
PYTHONPATH=. SWEEP_TARGET_LIMIT=1 uv run --project generation modal run generation/run_helico.py
PYTHONPATH=. uv run --project generation modal run generation/run_helico.py
uv run --project generation python generation/prepare_helico.py --phase folding
PYTHONPATH=. WRITEUP_PHASE=folding SWEEP_TARGET_LIMIT=1 uv run --project generation modal run generation/run_helico.py
PYTHONPATH=. WRITEUP_PHASE=folding uv run --project generation modal run generation/run_helico.py
```

The contact launcher stages the published checkpoint once (5.9 GB) into a Modal
volume, validates SHA256 including both weight shards and tokenizer, and runs a
one-target smoke before eight independent H100 shards in `us-east`. It uses
vLLM 0.9.2 and the exp277-pinned MarinFold document generator. Each protein gets
100 fresh prompt realizations, T=1, top-p=.95, top-k disabled and budget 6L+128
capped by context. The batch contains one protein so timings are per protein.
Raw completions, vote tables, status and timings persist in `marinfold-exp325`.
`--collect-only` on `run_contacts.py` recovers them without inference.

Helico uses the original step-6000 checkpoint, SHA256
`779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644`,
three diffusion samples, six recycles, seed 42, and single-sequence input.
`run_helico.py` accepts an explicit contact-state matrix through the public
`run_inference` API. It validates checkpoint model/training fields and does not
patch Helico. The folding cut is exactly L, fixed before looking at test scores.
Pairs becoming closer than six residues after mapping are discarded by Helico, as in exp311; requested and effective counts are recorded separately. Unknown pairs remain unknown. Every sample is scored and saved; only confidence
selects the displayed prediction. Per-map coordinates and contact states persist
in the `marinfold-exp325-helico-results` Modal volume.

The confidence target list is frozen before inference. Randomization preserves
known/unknown pairs and positive/negative counts. A second control also preserves
counts at sequence separations 6–11, 12–23 and ≥24, using residue coordinates
even when token indices include non-protein atoms. Seeds 1–5 give five maps per
control; each map receives the same three-sample diffusion budget. The oracle
and each random map are ranked by their own highest-confidence sample.

`score_contacts.py` checks that raw completions reproduce every saved vote before
scoring. It calls exp89's unchanged metric implementation. All/long R-precision
use resolved-residue candidate pairs, minimum separation 6/24, pyconfind degree
≥.001, stable vote ties, and R equal to the true-contact count. Main contact
scores exclude unfinished rollouts from votes as exp277 does. The sampling
diagnostic follows exp321: all parsed maps vote, while invalid individual maps
score zero; short individual maps retain denominator R. This difference is
explicit and prevents silently mixing the two recipes.

Timing scope: inference and model-load times are direct measurements. The initial
contact run inherited exp277’s setup-plus-inference sum for `total_seconds`,
which omits CPU prompt/parse/write overhead; this is recorded in its manifest.
Helico’s total includes inference and scoring but is recorded before coordinate
serialization. These figures make no speed comparisons.

# Public artifacts

Raw contact completions/votes and every new Helico diffusion sample are archived
under `data/exp325-writeup-analysis/exp277-step266344/v6-ptm-ranking` in the public
`open-athena/MarinFold` HF bucket. `data/publication_manifest.json` records file
sizes and SHA256 digests. The package includes the offline preview, local font
and Plotly bundle, figures, prepared tables and experiment scripts. Existing
experiment source tables are available in the MarinFold repository.

To rebuild that package after an intentional analysis update:

```bash
uv run --project generation modal run generation/archive_helico.py
uv run --project generation modal run generation/archive_helico.py --structured
uv run python publish_to_hf.py --upload
```

The archive commands recover already-computed coordinates from the durable volume;
they do not run a predictor. The final command publishes only this experiment's prefix.

# AlphaFold baseline extension

`generation/prepare_alphafold.py` pins the 333 sequences, exact existing MSA files,
private parameter digests and all sampling choices. It verifies that every MSA
depth equals the count already used in the plots. AF2 uses ColabFold's model API;
AF3 uses the official unmodified source revision in `data/alphafold_inputs.json`.
Both use protein-chain-only inputs and no templates. No ground-truth structures
are sent to the predictor workers. Current workstation paths are explicit in the
preparer and staging script; parameters are copied only to the private Modal volume.

```bash
uv run --project generation python generation/prepare_alphafold.py
uv run --project generation python generation/stage_alphafold.py
AF_VARIANT=af2 uv run --project generation modal run generation/run_af_baselines.py --limit 2
AF_VARIANT=af3 uv run --project generation modal run generation/run_af_baselines.py --limit 2
# After both real-protein smoke runs pass scoring:
AF_VARIANT=af2 uv run --project generation modal run generation/run_af_baselines.py
AF_VARIANT=af3 uv run --project generation modal run generation/run_af_baselines.py
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_alphafold.py --variant af2
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_alphafold.py --variant af3
uv run python prepare.py
uv run python render.py
uv run python build_summary.py
```

Add `--smoke` to the scorer for the initial two-protein validation. Predictor
`--collect-only` recovers results without GPU work; completed proteins are skipped
when a run resumes. Each timing row separates model load and inference calls;
inference includes cold-shape JIT compilation, explicitly labeled.
Total time includes model initialization and per-protein work through candidate
writes; it excludes parameter-integrity checks, the completion marker and volume
commit. These runs are not used for a speed comparison. All candidates,
confidence summaries and selected structures are kept; AF3's quadratic PAE files
are omitted because these analyses use coordinates and summary confidence only.
`publish_to_hf.py` includes the new input/output archives without any parameter files.
Exports read completed target directories in bounded concurrent batches to avoid
serial object-storage latency. This changes packaging only, not prediction bytes.

## Boltz-2 baseline

Uses official source `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` (2.2.1) and
public structure checkpoint revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`.
The fixed recipe is one seed (42), 25 diffusion samples, ten recycles, 200
sampling steps, up to five parallel samples, step scale 1.5, bf16 mixed precision.
The archived MSA is parsed with upstream deduplication and an 8,192-sequence cap, without optional
MSA subsampling. No templates, constraints, physical potentials or affinity run.
The upstream confidence score selects one structure (for these monomers,
0.8 × complex pLDDT + 0.2 × pTM). This follows the
[documented 25-sample/ten-recycle option](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/docs/prediction.md);
matching AF3's sample count is not a claim of equal compute or identical seeds.

```bash
uv run --project generation modal run generation/run_boltz2.py --stage-only
uv run --project generation modal run generation/run_boltz2.py --smoke
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_boltz2.py --smoke
uv run --project generation modal run generation/run_boltz2.py
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_boltz2.py
uv run python prepare.py
uv run python render.py
uv run python build_summary.py
uv run pytest -q
```

The smoke proteins are fixed by input properties: `5sbj_A` (unknown residues)
and `8wnj_A` (longest, 917 residues). No accuracy-based recipe selection.
`--collect-only` recovers completed outputs. The separate Modal volume
`marinfold-exp325-boltz2` stores weights and outputs; staging extracts the 21
canonical protein/unknown CCD entries from the checksum-verified public archive; the existing AlphaFold
volume supplies unchanged input MSAs. Every completion validates the protocol
and MSA hashes; scorer validation checks all candidates and selected bytes.
The public archive excludes weights, processed feature arrays and unused full
PAE/PDE matrices (the upstream writer emits these even with its write flags off),
retaining all
sample CIFs, confidence JSON, pLDDT arrays, input identity, selection and timings.

Timing uses synchronized Lightning prediction-batch callbacks before file
writing. Input featurization, model loading and output writing are excluded from
`elapsed_seconds`; `model_load_seconds` is recorded separately. `total_seconds`
adds model setup to per-target preparation, prediction and writing, excluding the
final completion-marker write and volume commit. Timing is retained for audit;
the figures make no speed comparison.

The full Boltz-2 run uses up to sixteen resident GPU workers in `us-east`.
Modal requests H100 GPUs; actual allocations (including H200 upgrades) are
recorded per protein, along with hardware and model-load time.

## Plausible contact-map decoys (Figure 02b)

The population is fixed: `8ii8_A`, `8oxk_A`, `8qoh_A`, `8ux2_A`, `8wrx_A`.
These are all five natural FoldBench monomers at archived query-inclusive MSA
depth <10, all in the authorized test split. ESMFold2 receives the same resolved
protein queries as the new AF2/AF3/Boltz-2 baselines, without MSAs or templates.
Native ESM source `43b4548b86762edfa747b07d5f440aad3c33acee` loads the existing
exp78 weight snapshot `1ebf0e3481a5184eb6171d40615c79e384b48796` from the
`esmfold2-weights` Modal volume. The native implementation's documented LM
dropout is explicitly 0.3; seeds 0–99, 20 loops and 100 diffusion steps are fixed.
Every sample is retained. This new sampling run does not replace the archived
ESMFold2 baseline scores used by the other panels.

```bash
uv run --project generation modal run generation/run_esmfold2_decoys.py --smoke
uv run --project generation modal run generation/run_esmfold2_decoys.py
uv run --project /home/bizon/git/helico --no-sync python generation/prepare_structured_decoys.py
uv run --project /home/bizon/git/helico --no-sync --with scikit-learn python generation/score_esmfold2_decoys.py
PYTHONPATH=. WRITEUP_PHASE=structured HELICO_DRY_RUN=1 uv run --project generation python generation/run_helico.py
PYTHONPATH=. WRITEUP_PHASE=structured uv run --project generation modal run generation/run_helico.py
uv run --project generation modal run generation/archive_helico.py --structured
uv run python prepare.py
uv run python render.py
uv run python build_summary.py
```

Contact extraction uses Helico's existing `oracle_contact_state` for both
predicted and experimental structures. Exact query identity is required before
mapping. Pyconfind geometry, degree cutoff and sequence separation are identical.
All maps share the oracle's eligible-pair mask; present/absent counts may vary.
Full non-contact information is supplied for both sources. The mask uses
experimental residue eligibility, so this is a controlled diagnostic, not an
end-to-end deployable selection benchmark.

Helico source, checkpoint and inference settings match the earlier control:
three diffusion samples per map, six recycles, seed 42, no MSA. The 505 maps
are dispatched in small disjoint blocks to eight resident H100 workers in
`us-east`. In preprocessing, highest pTM selects one sample from each map
before sorting maps by pTM. Neither ipTM nor the clash flag affects selection;
within-map ties use the lowest sample index. The raw run manifest describes the
original inference-time ranking_score selection; the prepared manifest records
this later pTM reanalysis. Every sample already has pTM and measured TM-score,
so no inference or structural rescoring is required. pLDDT is omitted because
it was saved only for the original ranking_score winner.
Ties are reported as a rank interval and a midrank, with no random tie-break.
CSV rows retain inference timings, seeds, source rows and structure/map hashes.
Native ESMFold2 timings include input preparation and result decoding inside
`builder.fold`; they are retained for audit and are not a speed comparison.
Restyling only requires `render.py` and `build_summary.py`.

Figure 02c joins each original ESMFold2 structure's TM-score to Helico's
pTM after using that structure's contacts. Its alternate views use the
measured downstream Helico accuracy. The oracle source is the experimental
structure, so its source accuracy is one by definition; its downstream Helico
accuracy is scored normally. The scorer verifies all coordinate digests and
requires full CA coverage, reusing `structure_scores` from the AF2/3/Boltz-2
analysis without a different alignment or metric implementation.

The pTM selection comparison is cached in `structured_selection_comparison.csv`.
Figure 02c uses pTM on x and measured TM-score on y. Each protein gets a PDF
page with original ESMFold2 and reconstructed Helico panels. Other structural
benchmark figures retain their archived selection protocols.
