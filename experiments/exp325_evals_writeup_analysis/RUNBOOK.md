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
under `data/exp325-writeup-analysis/exp277-step266344/v1` in the public
`open-athena/MarinFold` HF bucket. `data/publication_manifest.json` records file
sizes and SHA256 digests. The package includes the offline preview, local font
and Plotly bundle, figures, prepared tables and experiment scripts. Existing
experiment source tables are available in the MarinFold repository.

To rebuild that package after an intentional analysis update:

```bash
uv run --project generation modal run generation/archive_helico.py
uv run python publish_to_hf.py --upload
```

The first command recovers already-computed coordinates from the durable volume;
it does not run a predictor. The second publishes only this experiment's prefix.
