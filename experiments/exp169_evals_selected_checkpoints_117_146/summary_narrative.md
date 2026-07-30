# Summary slides — exp: evaluate selected checkpoints from #117 and #146

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Scoring the three checkpoints issue #169 selected — the final and early-stop
winners of #117 (1.5B, 16 epochs) and the #146 3B — on the 554-protein contact
eval set, and reporting **R-precision** for each.

The measurement spec is unchanged from what the project already publishes:
exp89's ground truth, candidate-pair universe and metric implementation; exp82's
rollout+resample inference recipe (100 resampled contacts-v1 rollouts per
protein, top-k off), run through exp82's worker unmodified.

## Why

All three checkpoints were picked by `eval/tokenized/contacts-v1-val/loss`, and
they sit within **0.008 nats** of each other. So the question underneath "what
R-precision do they get?" is whether val loss is still a useful selection signal
at that granularity, or whether the loss to accuracy relationship has gone flat.

The prior evidence is a steep slope: 2.7566 to 2.7037 (0.053 nats) bought +0.11
R-precision. Extrapolated locally that predicts +0.016 for the 0.0076-nat gap
between the two #117 checkpoints — small, but resolvable with a paired test over
554 proteins.

## How the comparison is kept honest

Every checkpoint scores the same 554 proteins, so differences are reported
**paired**. That matters: the between-protein spread of R-precision is ~0.3, so
the unpaired SEM (~0.013) is wider than the entire effect being measured. The
paired standard error is an order of magnitude smaller, because protein-to-protein
variance cancels.

Two other guards. The #117 final checkpoint is re-scored rather than reused from
its published run, so all three numbers come from one submission — and
reproducing the published 0.535 is itself the harness check. And
`verify_prepared_exports.py` gates the run on the three checkpoints sharing one
vocabulary and one set of special-token ids, since a silent tokenizer shift would
not fail loudly; it would just produce confident wrong numbers.

## Results

554/554 proteins for all three checkpoints, 0 of 166,200 rollouts truncated.

| checkpoint | val loss | R (all) | R (long) | AUC (all) |
|---|---:|---:|---:|---:|
| #117 · 1.5B · final | 2.7037 | **0.534** | **0.482** | 0.933 |
| #117 · 1.5B · early stop | 2.6961 | 0.532 | 0.481 | 0.933 |
| #146 · 3B · E8 | 2.7025 | 0.512 | 0.459 | 0.925 |

The #117 final row reproduces its published 0.5350 to 0.0006 — the harness check.

## What it says

**Early stopping on val loss bought nothing.** The early-stop checkpoint has
0.0076 lower loss and is indistinguishable on contacts: Δ +0.0026 favouring the
final checkpoint, CI [-0.0010, +0.0062], win rate 48.6%. The exp82 slope
predicted +0.016. The loss to accuracy relationship is steep across training
generations and flat inside one run's last 2,000 steps.

**Matched val loss does not mean matched contact accuracy across model sizes.**
The 3B's loss is 0.0012 *better* than the 1.5B final, and its R-precision is
0.023 *worse* — resolvable, CI [+0.017, +0.028], and the largest effect in the
comparison. (Confounded with epochs 8 vs 16 and wd 0.4 vs 0.2, so it is about
this checkpoint, not scale as such.)

Practical consequence: `contacts-v1-val/loss` is not a usable tie-breaker below
~0.01 nats, and must not be compared across model sizes to pick a contact
predictor.
