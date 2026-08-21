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

Ran 2026-08-21 on one workstation A5000 (vLLM 0.19.1, bf16). 97 proteins x 100
rollouts per arm, 33.7 min per arm; the pairwise pass took 4.2 min for all 97.
**Zero of 9,700 rollouts hit the token cap in either arm, and zero emitted
nothing** -- the `6L+128` budget never binds on this set.

### The gate

| | this run | #245 published | Δ |
|---|---:|---:|---:|
| m2-p06, eval-val, all-range R-precision, rollout consensus | **0.5217** | 0.520 | +0.0017 |

Inside the 0.005 tie band, so the harness reproduces the published recipe and
the arms below are comparable to everything else on this axis.

### Headline

All on eval-val (n=97). "Oracle best-of-100" needs the ground truth to pick a
rollout and is a headroom diagnostic, not a recipe.

| readout | R (all) | R (long) | AUC (all) |
|---|---:|---:|---:|
| pairwise `P(contact)` (the seed source) | 0.4585 | 0.4407 | 0.9433 |
| **i.i.d. consensus** (control) | 0.5217 | 0.5042 | 0.9310 |
| **seeded consensus** | 0.5234 | 0.5081 | 0.9307 |
| seeded consensus, seed vote removed | 0.5237 | 0.5087 | 0.9306 |
| *i.i.d. oracle best-of-100* | *0.5199* | *0.5415* | — |
| *seeded oracle best-of-100* | *0.5341* | *0.5433* | — |

Paired per-protein differences, all-range R-precision, 95 % bootstrap over
proteins:

| comparison | Δ | 95 % CI | proteins won |
|---|---:|---|---:|
| seeded − i.i.d., consensus | **+0.0017** | [−0.0011, +0.0046] | 40 % |
| seeded − i.i.d., consensus (seed vote removed) | +0.0020 | [−0.0007, +0.0049] | 39 % |
| seeded − i.i.d., oracle best-of-100 | **+0.0142** | [+0.0055, +0.0247] | 58 % |
| i.i.d. consensus − pairwise | +0.0632 | [+0.0516, +0.0767] | 90 % |

**Primary (consensus): a tie**, as preregistered — +0.0017 with an interval
straddling zero and well inside #204's 0.005 noise floor. The seed's own vote
is not what carries it: removing it moves the number by +0.0003.

**Secondary (oracle best-of-100): a real but small win**, as preregistered —
+0.0142 [+0.0055, +0.0247] all-range. It is confined to the all-range cut; at
long range the two oracles are indistinguishable (+0.0018 [−0.0156, +0.0158]).

### Why: one contact is almost no conditioning

The rollout index *is* the pairwise rank of the seed it was handed, so the
experiment contains its own dose-response curve. Seed accuracy collapses down
the ranking and rollout quality does not follow:

| seed rank | seed is a true contact | R-precision of those rollouts |
|---|---:|---:|
| 1–10 | 79.2 % | 0.3931 |
| 11–20 | 69.6 % | 0.3898 |
| 21–40 | 64.6 % | 0.3921 |
| 41–70 | 55.7 % | 0.3917 |
| 71–100 | 46.0 % | 0.3890 |
| *unseeded (i.i.d.)* | — | *0.3868* |

A 33-point swing in whether the model was told the truth moves the rollout it
then writes by 0.004. Contrasting within each protein — where both kinds of seed
occur against the same ground truth — a true seed is worth **+0.0124**
[+0.0082, +0.0168] over a false one (73 % of proteins), against 0.3861 for a
false seed and 0.3868 for no seed at all. So a wrong seed costs nothing
measurable and a right one buys about a hundredth, which is exactly the size of
the consensus tie.

**The pooled version of that split is a trap and is not the result.** Pooled
across proteins it reads +0.182 (0.4668 true vs 0.2847 false), which looks like
a spectacular conditioning effect and is almost entirely protein difficulty: a
protein the model handles well supplies both more correct seeds and better
rollouts. `exp254_seed_conditioning_summary.csv` carries both numbers so the
confound is visible rather than inferred.

### Two things worth noticing on the side

- **Consensus over 100 rollouts is as good as the best single rollout**, at
  all-range: 0.5217 versus an oracle 0.5199 (Δ −0.0018 [−0.0164, +0.0137]).
  Best-of-N leaves essentially no all-range headroom on this set. At long range
  it does — oracle beats consensus by +0.0374 [+0.0129, +0.0660] — so the
  headroom that exists is in the long-separation contacts.
- **The pairwise readout has the better AUC and the worse R-precision** (0.9433
  vs 0.9310 AUC; 0.4585 vs 0.5217 R). It ranks the whole candidate universe
  better and the top of the list worse, which is a coherent description of a
  marginal readout that never commits to a joint structure.

![seeded vs i.i.d. on eval-val](plots/seeded_vs_iid_eval_val.png)

Artifacts: `data/exp254_per_protein.csv.gz` (5 readouts x 97 x 4 ranges x 5
cuts), `data/exp254_headline.csv`, `data/exp254_paired_deltas.csv`,
`data/exp254_seed_conditioning.csv.gz` (19,400 individual rollouts),
`data/exp254_seed_conditioning_summary.csv`, `data/exp254_seed_rank.csv`,
`data/timings.csv`. The rollout dumps and pairwise matrices stay on local disk
under `/data/exp_contactseed/run` (~250 MB); nothing needed publishing.

**Timing caveat.** `data/timings.csv` for this run was parsed from the run log
by `collect_timings.py` (`source = run_log`) rather than emitted during it;
`score_rollouts.py` now writes the same CSV at eval time. In both cases the
timing unit is the **chunk** of 8 proteins, not the protein: vLLM schedules a
chunk's 800 rollouts together, so there is no separable per-protein inference
time to record. Do not sum `elapsed_seconds`.

## Conclusion

**Seeding rollouts with our own top-ranked pairwise contacts does not improve
contact prediction.** Consensus R-precision is 0.5234 against a 0.5217 control —
a tie by any reading, and both preregistered predictions held.

The reason is more interesting than the headline. It is not that the seeds are
bad: 58 % of the top-100 pairwise pairs are true contacts, and 79 % of the top
ten are. It is that **one contact is almost no conditioning at all.** Handing a
rollout a true contact instead of a false one changes what it writes by +0.012,
and handing it a false one instead of nothing costs +0.001. Against #163's
finding that conditioning on *true partial contact sets* lifts R-precision from
0.145 to 0.556, the implication is that the signal lives in the joint structure
of many contacts, not in any single anchor — a single pair is simply not enough
constraint to move a 1.5B model that has already read the whole sequence.

That reframes the seeding idea rather than killing it. The lever this experiment
tested was per-rollout accuracy and it barely exists; the lever that did move was
**diversity** — forcing 100 distinct starting pairs made the best-of-100 better
by +0.014 without making the average better, which is what a broader search of
the same posterior looks like. A follow-up worth running is therefore seeding
with **k > 1 contacts at once** — a partial map rather than an anchor —
where #163 says the conditioning signal actually is, accepting that at 58 % seed
accuracy a k-contact seed is right only 0.58^k of the time and so needs either a
sharper prior (the top-10 seeds are 79 % accurate) or a model trained to
retract, as #158/#175 built.

Two negative results are worth keeping on their own: consensus over 100 rollouts
already matches the oracle best single rollout at all-range on this set, and the
remaining best-of-N headroom is entirely long-range.
