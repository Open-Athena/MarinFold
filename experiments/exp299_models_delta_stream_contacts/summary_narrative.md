## Sequence before contacts

The delta-stream format places the whole amino-acid sequence in a causal prefix before any contact target:

`DOC_START AA_0 ... AA_(L-1) CONTACTS_BEGIN DELTA* STOP ... DOC_END`.

The contact suffix contains one signed-delta segment per residue, in sequence order. The model can therefore see every amino acid before predicting any contact, matching the conditioning structure used by contacts-v1.

## Evaluation

The reference metric is exp82's rollout-vote evaluation: sample 100 sequence-conditioned contact suffixes, vote emitted pairs, and run the unchanged exp82/exp89 candidate-universe and R-precision code.

## Canonical result at 28% schedule progress

We evaluated V2 step 22,000 and exp177 contacts-v1 step 20,000 with 100
sequence-conditioned rollouts and unchanged exp82/exp89 metrics. V2 uses about
10% more tokens per protein, so these checkpoints match both estimated protein
exposure and schedule progress: each is approximately 28% through its full
cosine schedule.

| subset | proteins | model | R/all | R/long | AUC/all |
|---|---:|---|---:|---:|---:|
| legacy | 554 | V2 step 22,000 | **0.4691** | **0.4028** | **0.8886** |
| legacy | 554 | exp177 step 20,000 | 0.0250 | 0.0223 | 0.5995 |
| eval-val | 97 | V2 step 22,000 | **0.3321** | **0.2942** | **0.8467** |
| eval-val | 97 | exp177 step 20,000 | 0.0219 | 0.0197 | 0.5880 |
| eval-denovo | 19 | V2 step 22,000 | **0.4383** | **0.3935** | **0.8830** |
| eval-denovo | 19 | exp177 step 20,000 | 0.0284 | 0.0340 | 0.5866 |

On the legacy universe, the matched V2 checkpoint is 18.8× higher on all-range
R-precision and 18.1× higher on long-range R-precision. It has already reached
92% of exp177-final all-range R-precision at only 28% of matched exposure. The
large advantage also transfers to the 19 held-out de-novo proteins, so the
result is not confined to the older evaluation universe.

All 670 proteins were scored. The V2 run produced no malformed rollouts among
67,000 samples.

## Step-30,000 progress check

At 38.2% of the full schedule (approximately 27,273 contacts-v1-equivalent
steps), V2 reaches **0.5091 R/all** and **0.4390 R/long** on the legacy universe.
That is 99.6% and 95.5% of the confirmed exp177-final values, 0.5113 and 0.4595.
On `eval-denovo`, V2 improves from 0.4383 to **0.4881 R/all** and from 0.3935 to
**0.4224 R/long** between steps 22,000 and 30,000. All 670 targets were covered,
with no malformed generations among 67,000 rollouts.

## Step-36,000 crossover

At step 36,000 (45.9% schedule; approximately 32,727 contacts-v1-equivalent
steps), V2 reaches **0.5130 R/all** on the legacy universe, narrowly exceeding
exp177 final at 0.5113. The paired protein-bootstrap interval for this narrow
R/all difference is [-0.0067, 0.0103], so it is not yet a resolved lead. V2 has
not crossed the contacts-v1 reference on long-range R-precision (0.4432 versus
0.4595) or AUC. On `eval-denovo`, V2 reaches 0.5261 R/all and 0.4626 R/long.

## Step-42,000 regression

Despite lower validation CE, step 42,000 falls to 0.4761 R/all and 0.4055
R/long on the legacy universe. An independent second 100-rollout run reproduces
the result at 0.4756 and 0.4058. Paired protein-bootstrap intervals for the
step-42,000 minus step-36,000 difference exclude zero: [-0.0475, -0.0269] for
R/all and [-0.0490, -0.0268] for R/long. The rollout curve is therefore
non-monotonic even while teacher-forced validation CE improves.

## Step-48,000 recovery and new high

Step 48,000 recovers to **0.5418 R/all** and **0.4748 R/long** on the legacy
universe, exceeding exp177 final on both R-precision metrics. Paired-bootstrap
V2-minus-exp177 intervals are [0.0207, 0.0407] and [0.0044, 0.0265],
respectively. AUC remains below exp177 final at 0.9131 all-range and 0.8878
long-range. On `eval-denovo`, V2 reaches 0.5272 R/all and 0.4595 R/long.

## Final mature-run result

The fresh run completed successfully at step 78,499 after 112.9 hours on 2 × 4
GB200 GPUs. It processed 82.31B packed tokens, approximately 94.73M protein
exposures, at mean throughput 205,396 tokens/s. Final train CE was 0.9190 and
final/best native validation CE was 0.91425. Its approximately 71,363
contacts-v1-equivalent steps nearly exactly match exp177 final at 71,359.

On `legacy_554`, final V2 reaches **0.5763 R/all** and **0.5160 R/long**, versus
0.5113 and 0.4595 for exp177 final. Paired-bootstrap V2-minus-exp177 intervals
are [0.0547, 0.0757] and [0.0450, 0.0684], establishing a resolved advantage.
AUC is effectively tied: 0.9245 / 0.9028 for V2 versus 0.9246 / 0.9033 for
exp177, with paired intervals spanning zero. On `eval-val`, final V2 scores
0.5147 / 0.4842 R/all / R/long; on the 19-protein `eval-denovo` set it scores
0.5494 / 0.4779. The best late `eval-denovo` point estimate is step 72,000 at
0.5534 / 0.4871, illustrating residual sampling/checkpoint noise at n=19.

## Why V2 CE is low

Paired teacher-forced scoring on the same 670 R-precision proteins confirms
that the contact suffix is much easier to predict token-by-token than the
amino-acid prefix. At step 32,000, V2 has **2.2575 CE on amino acids** and
**0.4086 CE on contact tokens**. The exp177-final reference has 2.6024 AA CE
and 2.2301 contact CE on exactly the same proteins.

This means V2's low aggregate CE partly reflects an easier serialization. The
rollout R-precision result remains independent evidence that it also learns the
contact task better and faster. Full token losses are paired by `(dataset,
stem)` for protein-level analysis.

## Next steps

Review the approved format/objective ablations before launching more training:
one-sided contact-loss masking, preferred-triangle serialization, absolute
partner positions, and removing forced per-residue `STOP` tokens. Measure
strict grammar-constrained decoding separately.
