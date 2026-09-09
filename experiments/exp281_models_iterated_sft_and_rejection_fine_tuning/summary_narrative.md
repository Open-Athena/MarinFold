# exp281 — Iterated contact synthesis SFT

## Question and design

Can prolonged SFT teach a model to combine multiple structural hypotheses, then improve its histories through rejection selection?

Warm up the multi format and explicit final-prediction token. Later use hypothesis weight 0.1 with full reference-answer supervision and refreshed corpora. Rejection ranks generated final answers but trains against reference contacts.

## First full-model trial completed

Current exp232-derived 1.5B model; one 8-H100 node on cw-us-east-02a, with all large I/O in its existing S3 bucket. Iris selected placement from live fleet capacity.

2,048 source-balanced proteins: 2,023 training, 25 sequence-hash-held-out. Sixteen bootstrap drafts, 50% plain rehearsal, hypothesis weight 1.0. Global batch 32, 256 optimizer steps, 8,192 training documents processed.

W&B: open-athena/MarinFold/exp281-trial-s03. Source: bab3f50a. Final model: trial-s03/checkpoints/exp281-trial-s03/step-256 under the exp281 S3 prefix.

## Result: format gate failed

At step 256, sample eight completions per held-out protein in each mode. Natural seed 281; forced seed 282 and budgets 0/256/1024/2048.

Natural: 0/200 valid. Forced: 14/200 valid (7%). Both fall below the fixed 99% validity gate. The natural multiple-nonempty-hypothesis gate also fails.

188/200 natural samples contain multiple raw section markers, but only 8/200 contain the final marker. In forced mode, 185/200 restart a hypothesis after the final marker. Repeated sections have appeared; reliable finalization has not.

## Lower loss did not imply finalization

Internal teacher-forced validation loss decreases from 2.781 at step 32 to 2.244 at step 256. Final training minibatch loss: 1.869.

The corpus has only 500 supervised natural final markers per pass; the other 1,054 supervised markers are plain-mode starts. Aggregate contact-token loss can improve while the rare switching behavior remains unreliable.

Invalid trajectories retain zero final F1: natural 0.0000, forced 0.03193. These are engineering/format diagnostics, not evidence of improved protein accuracy.

## Throughput and recovery

Peak allocated memory: 35.45 GB per H100. Median optimizer step: 1.049 seconds, excluding checkpoint publication and validation. A separate eight-step batch-8 profile reached 58,480 aggregate tokens/s after step 1.

The original trial stalled publishing its step-128 optimizer, then a waiting rank aborted. Resume from complete step 64 succeeded through step 256. Long runs need bounded checkpoint retries. The canonical training CSV uses the successful resumed trajectory; W&B explicit-step logging suppressed replayed metrics after rollback.

## Next gate and open research questions

Extend format warm-up toward the proposed 2,000 steps and track final-marker/termination losses separately before synthesis or rejection training. A short-history curriculum and stronger transition supervision are additional candidate interventions. This 256-step pilot does not test the intended long SFT horizon.

The first trial establishes full-model execution and checkpoint recovery. It does not establish useful diversity, improved contact accuracy, or the benefit of refreshed SFT or rejection selection. FoldBench eval-test remains untouched.

The public HF report and committed per-input metrics, diagnostics, timing CSVs, and plotting inputs preserve this negative result for reproduction.
