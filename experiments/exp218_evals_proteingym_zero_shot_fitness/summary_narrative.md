# exp218 — contacts-v1 as a bidirectional protein language model

Issue #218, kind evals. Branch claude/proteingym-marinfold-tasks-ce0cc6.

Every contacts-v1 document opens with the protein's residues as a randomly ordered
list of <pN> <AA> statements. Prompt the model with every residue but one and it
returns a distribution over the one left out — the same conditional ESM-1v and
ESM-2 compute with a mask token, and the object ProteinGym's zero-shot DMS
benchmark scores.

This experiment asks whether that conditional is any good, and reports the answer
on the community-standard scale.

Headline: 0.2964 average Spearman over 212 assays. That is ESM2-35M territory,
rank 88 of 97, and 0.119 behind ESM-2 650M. Real signal, not competitive.

## Why this is not a party trick

Roughly half of contacts-v1 pretraining is sequence modelling. From exp53: 4,213,203
documents, ~4.67B train tokens, mean 1,131 tokens/doc, mean ~200 contacts/doc.
Contacts cost 3 tokens each and the frame is 4, so the sequence section is ~527
tokens — about 47% of all training tokens, half of which are the <AA> targets.
exp199's 152B-token run therefore spent on the order of 35B tokens predicting
amino acids from other amino acids.

There is no loss masking: marinfold_models trains a plain LM loss over packed
documents, so the sequence section is inside it.

The default model was trained as an any-order model on purpose. exp199's winner is
its "-aug" arm, which the exp166 sweep defines as giving every packed example a
fresh re-permutation of its <pN> <AA> statements, ramped from probability 0 to 1
over training. Under a uniform shuffle any statement can be last, so asking for
P(residue i | all the others) asks something the model saw directly.

## One pass per ordering, not one per residue

A document is an ordinary causal-LM sequence, so a single teacher-forced pass yields
the conditional at every residue at once: the slot holding a <pN> token already
carries P(amino acid at N | every statement before it). An L-residue protein costs
one forward pass per ordering rather than L.

What each residue is conditioned on varies with where its statement landed in the
shuffle, so the readout also returns a per-slot context size. That turns what would
be a nuisance into two knobs no masked LM has: how many orderings to ensemble, and
how much context to require.

The whole 212-assay benchmark took 29.1M tokens and 54.5 minutes on one A5000, at
8,912 tokens/sec. The cost forecast in the issue said 29.1M tokens.

## Preflight: three ways this could have lied

The aggregation is exact. ProteinGym's headline is not a mean over assays; it is a
mean within UniProt id, then within function category, then over the five
categories. Our implementation reproduces every published leaderboard number from
their own per-assay file to within 0.0005.

Mutation indexing is right. Across 212 assays and 2,438,361 variants, the stated
wild-type letter matches target_seq at the stated position every time.

The rope config survives the load. The bucket copy of exp199 carries both
transformers-4 and transformers-5 rope, so the silent theta-10000 fallback, worth
0.76 nats/token, does not fire.

## Phase 0: the conditional is real

Top-1 accuracy at context >= 0.8 is 0.345, against a 0.132 composition floor and
0.088 for a scrambled-sequence control. Perplexity 8.15 versus 18.72.

The scrambled arm is the load-bearing control. It stays flat as context grows
(0.094 to 0.088) while the model arm nearly triples (0.103 to 0.345). Composition is
identical in both arms by construction, so everything the model gains from seeing
more of the protein is real sequence structure.

The conditional is also soft rather than memorised: P(wt) is only 0.231 at high
context, with entropy 2.15 of a possible 3.00 nats. So the ranking of the other 19
amino acids, which is the only thing a variant-effect score reads, sits in a broad
distribution. An early ubiquitin smoke test had returned top-1 0.98, which would
have implied a memorised spike; across 14 diverse proteins that is an outlier.

## Phase 1: both knobs work, and double the score

K=1 with every slot scores 0.147. K=200 with context >= 0.9 scores 0.2964.

The ordering ensemble saturates fast: most of its gain is in by K=16 (0.147 to
0.225), and K=64 to K=200 buys 0.0011. The context threshold keeps paying to the
end. The pre-registered primary rule is also the grid's best cell, so nothing here
rests on test-set selection.

The readout approximation is not what costs us the gap. At context >= 0.9 a residue
is never conditioned on exactly all-but-one, but the context increments are
diminishing (+0.038, +0.018, +0.009), so exact masked marginals is worth 0.01 or
less against a 0.119 gap. The gap is model quality, not estimator error.

## The structure-tilt prediction fails

The issue predicted a structure-trained model would tilt toward Stability. It does.
So does an equally weak sequence-only model, which is the control that makes the
prediction falsifiable.

MarinFold scores Stability at 1.36x its own average and OrganismalFitness at 0.71x.
ESM2-35M, a sequence-only model at a matched overall level, scores 1.36x and 0.71x.
ESM-IF1, a genuine structure model, scores 1.47x.

MarinFold's profile is indistinguishable from ESM2-35M's and unlike ESM-IF1's.
Whatever this model has learned presents as a small sequence model, not a structure
model wearing a sequence readout. The Stability tilt is a property of weak
variant-effect predictors generally, not evidence of structural knowledge.

## Where it wins and where it is worst

MarinFold beats ESM-2 650M on 34 of 212 assays, 16%, median delta -0.106.

Best stratum is Stability, 29% win rate, 0.403 against 0.523. Worst are
OrganismalFitness at 7% and viruses at 7%.

Viruses are the standout weakness: 0.111 against ESM-2's 0.272. That is worth
flagging because the prior ran the other way — exp199 trains on ESM-Atlas
metagenomic data, so one might have expected better generalisation to
under-represented sequence space. It does not materialise.

Per-assay correlation with ESM-2 is r = 0.696, decorrelated enough that an ensemble
is worth testing. That test was not run: it needs ProteinGym's per-variant archive
(1.9 GB) and the workstation is at 100% disk with 7.8 GB free.

## Conclusion and what is next

contacts-v1 is a real but weak bidirectional protein language model. The conditional
is genuinely contextual and converts to 0.2964 zero-shot Spearman, well clear of the
0.188 broken-readout floor — but that is ESM2-35M territory.

The more interesting finding is the negative one. On the axis that was supposed to
reveal structural knowledge, the function-category profile, MarinFold is
indistinguishable from a small sequence-only model. Training on a structure
objective bought sequence understanding roughly in proportion to the sequence tokens
seen, and nothing extra this benchmark can detect.

Two things stay open and are both cheap now the harness exists. Exact joint scoring
for multi-mutants: 1.77M of 2.44M variants are multi-mutant and every single-sequence
baseline scores them additively. And the p03-base versus p03-aug ablation, exp199's
matched pair that ties on contact R-precision, where ProteinGym measures exactly what
the order augmentation was for.
