---
marinfold_experiment:
  issue: 254
  title: 'exp: seed each rollout with a top-ranked pairwise contact — does conditioning on our own high-confidence predictions beat i.i.d. sampling?'
  kind: evals
  branch: claude/contact-probability-inference-eval-a2d2ea
---

# exp: seed each rollout with a top-ranked pairwise contact — does conditioning on our own high-confidence predictions beat i.i.d. sampling?

**Issue:** [#254](https://github.com/Open-Athena/MarinFold/issues/254) · **Kind:** `evals` · **Branch:** `claude/contact-probability-inference-eval-a2d2ea`

## Question

Does **seeding** each of N contacts-v1 rollouts with a distinct high-confidence
contact — taken from the *pairwise* `P(contact)` readout, one seed per rollout —
beat plain i.i.d. rollout sampling, measured both by consensus R-precision and by
oracle best-of-N?

## Hypothesis

Two things are known and they pull in opposite directions.

1. **Conditioning on true partial structure is enormously informative.**
   [#163](https://github.com/Open-Athena/MarinFold/issues/163) found that
   prompting a rollout with *ground-truth* partial contacts lifted R-precision
   from 0.145 to 0.556. The joint signal is there; the problem has always been
   that we have no oracle at inference time.
2. **The pairwise readout is a weak stand-in for that oracle.** Under exp82's
   comparison the same weights score ~0.086 *lower* in R-precision under pairwise
   than under rollout voting, so the pairwise top-N is a noisy prior — good
   enough to be better than chance, not good enough to be trusted.

So the pre-registered expectation is:

- **Consensus:** roughly a wash, possibly slightly *worse* than i.i.d. A wrong
  seed poisons a whole rollout, and the seeded pairs additionally get a
  structural +1 vote each, which imports pairwise's ranking errors into the
  consensus. I do not expect this arm to clear i.i.d. by more than noise
  (0.005).
- **Oracle best-of-N:** seeded should be **better**. Forcing N *distinct*
  starting pairs decorrelates the rollouts, which is a broader search of the
  contact-map posterior — exactly what a best-of-N readout rewards. If seeding
  buys anything, it should show up here first, and that would make it a
  candidate reranking/RFT signal rather than a deployable decoder on its own.

The interesting negative result is also informative: if seeded oracle best-of-N
does *not* beat i.i.d. oracle best-of-N, then the per-rollout diversity we
already get from realization resampling is saturating the search, and the
headroom #163 identified is not reachable by conditioning on our own noisy
predictions.

## Background

- [#82](https://github.com/Open-Athena/MarinFold/issues/82) settled
  rollout+resample as the best LM-only contact inference and measured the
  pairwise-vs-rollout gap.
- [#163](https://github.com/Open-Athena/MarinFold/issues/163) measured the
  true-partial-conditioning ceiling (0.145 → 0.556) and found unconditional
  consensus over noisy candidates worth ~nothing (0.22 tie).
- [#232](https://github.com/Open-Athena/MarinFold/issues/232) /
  [#245](https://github.com/Open-Athena/MarinFold/issues/245) produced the
  decontaminated `m2-p06` checkpoint and the eval-val / eval-test / eval-denovo
  split. This experiment uses **`m2-p06` only**, and **eval-val only** — no
  eval-test read.

## Approach

Model: `contacts-v1-exp232-m2-p06-1.5B`
(`prot-exp232-cw-cv1-decontam-s02-m2-p06-aug` step-145199), from the public HF
bucket. Eval set: **eval-val, 97 natural FoldBench monomers** (#245). Sampling
recipe fixed to exp82's: T=1.0, top-p 0.95, top-k disabled, token budget
`6L+128`, N=100 rollouts per protein, one fresh document realization per rollout.

Three phases, all on one local A5000 via vLLM:

1. **Pairwise pass** (`rank_pairwise.py`) — the `P(contact)` readout from
   `marinfold.document_structures.contacts_v1.inference` on one canonical
   realization per protein; keep the top-100 pairs (sep ≥ 6) per protein as the
   seed list.
2. **Rollout pass** (`score_rollouts.py`), two arms over the same 97 proteins:
   - `iid` — the published exp82 recipe, unchanged. The control.
   - `seeded` — rollout *r* starts from realization *r* whose structure section
     is pre-filled with the *r*-th ranked pairwise contact, written in that
     realization's position tokens with a coin-flipped orientation (training
     randomizes pair orientation, so a fixed orientation would be off-manifold).
   Both arms dump **per-rollout, order-preserving contact lists** as well as the
   aggregated vote matrix, which is what the oracle readout needs.
3. **Scoring** (`build_metrics.py`, `plot.py`) — exp89's `compute_metrics.py`
   for consensus, exp82's `build_oracle_best_rollout.py` definition for oracle
   best-of-N. For the seeded arm, consensus is reported **both** with and
   without the seeded pair counted in its own rollout's vote, since the +1 is an
   artifact of the construction rather than a model prediction.

Deliverable: one plot on eval-val with all four MarinFold arms (iid consensus,
iid oracle-best-of-100, seeded consensus, seeded oracle-best-of-100) plus the
standard #245 predictors (Protenix-v2 single-seq and +MSA, ESMFold, ESMFold2,
both seq-KNN nulls, and the #199 cooldown / #232 checkpoints).

## Success criteria

All on eval-val (n=97), all-range R-precision, with long-range reported
alongside. **Differences under 0.005 are ties** (#204's four-replicate span is
0.0023).

- **Sanity gate:** the `iid` arm must reproduce #245's published m2-p06 eval-val
  R-precision of **0.520** within 0.005. If it does not, the harness is wrong and
  nothing else is reportable.
- **Primary:** seeded consensus − iid consensus. Preregistered prediction: ≤ 0,
  i.e. no win.
- **Secondary:** seeded oracle-best-of-100 − iid oracle-best-of-100.
  Preregistered prediction: > 0.
- Paired per-protein deltas with bootstrap intervals for both, not just the
  macro means.

## How to run

One local A5000, three GPU phases plus scoring. The `marinfold` package goes on
`PYTHONPATH` rather than into the venv: its base install pins
`transformers>=4.40,<5`, and the published m2-p06 export declares
`tokenizer_class: TokenizersBackend`, which only transformers 5.x resolves. This
is the same split exp82's CoreWeave worker uses (`pip install marinfold
--no-deps` into a vLLM image).

```bash
export PYTHONPATH=$PWD/../../marinfold
uv sync --extra test && uv run pytest test_exp254.py -q

MODEL=/data/exp_contactseed/model_m2_p06     # the published bucket export
RUN=/data/exp_contactseed/run

.venv/bin/python rank_pairwise.py  --model "$MODEL" --out "$RUN"
.venv/bin/python score_rollouts.py --model "$MODEL" --out "$RUN" --arm iid
.venv/bin/python score_rollouts.py --model "$MODEL" --out "$RUN" --arm seeded \
    --seeds "$RUN/seeds.parquet"
.venv/bin/python build_metrics.py --run "$RUN" --out data
.venv/bin/python plot.py --data data --out plots
```

The weights come from the public bucket, anonymously:
`hf://buckets/open-athena/MarinFold/checkpoints/prot-exp232-cw-cv1-decontam-s02-m2-p06-aug/hf/step-145199`
(`huggingface_hub>=1.5`; `list_bucket_tree` / `download_bucket_files` — buckets
are invisible to `snapshot_download`). Rollouts and pairwise matrices stay on
local disk under `$RUN`; only the derived CSVs and the figure are committed.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
