# Task: independently re-measure contact R-precision for the exp199 contacts-v1 model

We have two disagreeing numbers for the **same** checkpoint and need a third,
independent measurement. Please run this yourself and report the number you get.
Do not read our analysis first — we want an uncontaminated replication.

Context: MarinFold issue #209 (https://github.com/Open-Athena/MarinFold/issues/209).

## What to measure

Mean **R-precision, "all" range (sequence separation >= 6)**, over the fixed
554-protein contact eval set, using the **rollout-consensus** inference recipe.

## The checkpoint — pick ONE and say which

All three carry **byte-identical weights** (LFS sha256 `e8db3b66…` for shard 1,
`a7a38503…` for shard 2). They differ only in `config.json` / `tokenizer_config.json`:

| | location | rope stated as | `tokenizer_class` |
|---|---|---|---|
| A | `open-athena/marinfold-exp199` @ revision `ed7103bfd7dac3f75ba759e5ec827da3d75ff0ed`, subfolder `prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199` | `rope_parameters` only | `TokenizersBackend` |
| B | `hf://buckets/open-athena/MarinFold/checkpoints/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199` (the `MODELS.yaml` default) | `rope_theta` + `rope_scaling` **and** `rope_parameters` | `TokenizersBackend` |
| C | `timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199` | both forms | `PreTrainedTokenizerFast` |

**This matters and is the most likely source of a spurious discrepancy:**

- If your stack runs **transformers 5.x**, A / B / C all load correctly.
- If it runs **transformers 4.x**, A and B fail: 4.x ignores `rope_parameters`
  and *silently* loads `rope_theta = 10000` against a model trained at **500000**
  (wrong, no error), and `TokenizersBackend` is unresolvable so
  `AutoTokenizer.from_pretrained` raises outright.

**Before generating anything, print and report:** your transformers version, the
resolved `rope_theta`, `vocab_size` (expect **2845**), and the ids of
`<contact>`, `<begin_statements>`, `<end>`, `<p0>` (expect **5, 9, 10, 143**).
If `rope_theta` is not 500000, stop — the run would be meaningless.

Weights must be **bfloat16** at load (`dtype="bfloat16"`). The A/B copies are
fp32 on disk; casting at load is fine and is what vLLM does anyway.

## The eval code — use exactly this, do not reimplement

Repo `Open-Athena/MarinFold`, branch `main` (pin commit
`dd7670d2ecdaa415d9f23fac536075a4aa1bde4e`):

1. `experiments/exp82_evals_contacts_v1_contact_prediction/score_rollout_worker.py`
   — generation + vote matrices. Runs unmodified on CUDA and on TPU; all I/O is
   fsspec, so `s3://`, `gs://` and local paths all work. Shard it however suits
   your hardware (`--shard i/n`); it is resumable by `(dataset, stem)`.
2. `.../fetch_cw_scores.py` — sparse vote parquets -> `[L,L]` npz score matrices.
   It enforces 554 units and refuses any protein appearing in more than one part.
3. `.../build_rollout_rows.py` — per-protein metrics. This carries exp89's
   `compute_metrics` verbatim; **do not substitute another metric implementation.**

**There is an `eval-checkpoint` skill in the repo. Use it for the surrounding
discipline (checkpoint identity, compute/storage locality, provenance records) but
NOT for the scorer**: that skill specifies exp89's *pairwise* scorer, which
measures a different quantity and lands near 0.33. We need the *rollout* recipe
above, which lands near 0.6.

## Exact recipe

```
--n-rollouts 100 --temperature 1.0 --top-p 0.95 --top-k -1 --contact-mult 6 --seed 0
```

Add `--no-per-request-seed` **only if** you are on TPU (the JAX backend rejects
per-request seeds). Please report which you used, and your `--max-num-seqs`
(the worker's default is 512; the run we are trying to reproduce used **128**).

**Do not change `--top-p`.** This is a replication: both numbers under dispute
were measured at 0.95, and every published MarinFold rollout number uses it, so
it is what makes your result comparable to them.

For context, since 0.95 is a convention rather than a tuned optimum and you may
wonder: the only sweep behind it (exp82, `data/results_rollout_sweep_dev.txt`)
covered **16 dev proteins** and read, in long-band R-precision — pairwise
reference 0.160; T1.0/p0.95 **0.189**; T0.7/p0.95 **0.192**; T0.7/p1.0 **0.191**;
T1.0/p1.0 **0.184**. The p=0.95 and p=1.0 rows are within noise of each other at
that sample size, and the sweep ran `top_k 50`, which the settled recipe later
dropped to off (#142 traced under-generation to a finite top_k) — so no row in it
is exactly the recipe in use today. The robust signal there is rollout >>
pairwise, not the specific p. Whether 0.95 is the right operating point is a
worthwhile separate question and wants its own sweep on all 554 proteins; it is
explicitly **not** what this task is for.

## The prompts are resampled per rollout — this is not a knob, and not a variance source

`score_rollout_worker.py` builds a **fresh serialization of the protein for every
rollout**:

```python
doc = build_document(f"{stem}:r{k}", residues, [], config=GenerationConfig())
```

`build_document` is pure and deterministic given its first argument, which is the
RNG seed, and the seed differs per rollout `k`. Two things vary across the 100:

1. **Position-token offset** — `start = rng.randrange(num_indices)`, so the `<pN>`
   numbering is rotated differently each time. Absolute position tokens carry no
   fixed meaning across rollouts.
2. **Sequence-statement order** — `rng.shuffle(seq_statements)` shuffles the
   per-residue assignments and both termini.

So the 100 rollouts are 100 *different serializations of the same protein*, not
100 samples from one prompt. That resampling is the diversity the consensus vote
is built on. The worker keeps a per-rollout `{position_token -> seq_index}` map
and decodes every prediction back to sequence indices before voting, so votes
accumulate in a common frame.

Consequence worth knowing: prompt construction is **fully determined by the
stem**, so it is reproducible across runs and machines and cannot be a source of
disagreement between your number and ours. The only sampling randomness is the
engine seed.

## Fixed inputs — fetch, do not rebuild

- Ground truth universe:
  `https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/contacts-v1-model-eval-exp169/gt_universe.jsonl`
  Verify **554** `(dataset, stem)` units and **552** unique stems.
- Eval targets parquet (`dataset`/`stem`/`L`/`input_seq`), whichever is local to you:
  - TPU / GCS: `gs://marin-us-central1/protein-structure/MarinFold/exp169/eval_targets.parquet`
  - CoreWeave / S3: `s3://marin-us-east-02a/MarinFold/exp167_eval/eval_targets.parquet`

## Steps

1. Preflight the checkpoint (the printouts listed above).
2. Smoke-test one shard with `--limit 2` end to end before the full fan-out.
3. Generate all 554 proteins x 100 rollouts.
4. `fetch_cw_scores.py --parts <out>/<label> --out <matrices>` — must report
   `completeness OK: 554/554`.
5. `build_rollout_rows.py --gt gt_universe.jsonl --model <label>=<matrices>
   --out rows.csv.gz --summary summary.csv`
6. Report the mean of `precision` over rows with `cut == "R"` and
   `range == "all"` (n=554), and the same for `range == "long"` (n=553).
   Please also attach `rows.csv.gz` so we can compare per protein rather than
   only on the mean.

## What we expect, and what would be informative

Two prior measurements of this checkpoint disagree: **0.5873** and **0.6103**
(all-range), both at the recipe above. The gap is ~10x the reproducibility span of
this recipe — four evaluations of one unchanged checkpoint span 0.0023 — so your
number should land clearly on one side rather than in between.

Please report the number **before** looking at issue #209's analysis, and include:
transformers version, resolved `rope_theta`, `vocab_size`, the four token ids,
accelerator, `--max-num-seqs`, per-request-seed setting, and which checkpoint copy
(A/B/C) you used.
