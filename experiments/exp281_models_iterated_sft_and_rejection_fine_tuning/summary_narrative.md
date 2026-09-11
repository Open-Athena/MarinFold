# exp281 — Iterated contact synthesis SFT

## Question and design

Can prolonged SFT teach a model to combine multiple structural hypotheses, then improve its histories through rejection selection?

Warm up the multi format and explicit final-prediction token. Later use hypothesis weight 0.1 with full reference-answer supervision and refreshed corpora. Rejection ranks generated final answers but trains against reference contacts. Forced switches occur between complete statements.

## Two format trials completed

Exp232-derived 1.5B model. The pilot trained 256 steps on eight H100s in US-EAST-02A. A continuation reached global step 2,000, preserving Adam state, data positions and RNG, with a 100-step LR rewarmup and decay.

Both use the same 2,023 training and 25 sequence-hash-held-out documents, sixteen bootstrap drafts, 50% plain rehearsal, hypothesis weight 1 and global batch 32. This is extended repetition of a small frozen corpus, not the proposed larger refreshed SFT experiment.

W&B: exp281-trial-s03 and exp281-format-s02 in open-athena/MarinFold. Training sources: bab3f50a and 4721c76c.

## Result: extended training still fails the gate

Same 25 proteins, eight completions per mode: 400 total at each checkpoint. Natural seed 281; forced seed 282, budgets 0/256/1024/2048. Invalid outputs remain in the denominator.

Natural validity: 0/200 at both step 256 and step 2,000. Forced validity: 14/200 (7%) falls to 4/200 (2%). Both fail the fixed 99% validity gate. No valid natural trajectory has two nonempty hypotheses.

Natural final-marker emission rises from 8/200 to 98/200, but complete documents remain malformed. In forced mode, 136/200 outputs fail with an unexpected hypothesis-start token. More marker emission is insufficient for reliable finalization.

## The frozen corpus overfits

Held-out weighted token loss rises from 2.2445 at step 256 to 6.0468 at step 2,000. Final training minibatch loss is 0.00117.

Natural final-marker loss rises from 6.585 to 16.193; final-answer termination loss rises from 4.273 to 13.608. Top-1 accuracy remains zero at all checks. These sparse diagnostics contain six natural markers and nine multi-document end tokens; the independent generation evaluation establishes the format failure.

Final F1 with invalid outputs scored zero: natural 0.0000; forced 0.01220, down from 0.03193 in the pilot. These are internal format/engineering diagnostics, not FoldBench accuracy claims.

## Recovery completed on RNO2A

The first continuation attempt timed out publishing step 1,750 after five earlier saves completed in 97-101 seconds. The 300-second process deadline bounded the failure. Cleanup removed one abandoned 6.81 GB multipart upload.

US-EAST-02A recovery remained scheduler-blocked. The user approved approximately 53 GB of cross-region checkpoint I/O; the queued job was cancelled before resuming step 1,500 on eight RNO2A H100s. Both remaining checkpoints published in 218.6 seconds. Recovery worker duration: 1,349 seconds. The intermittent storage stall remains unresolved.

Median optimizer step: 1.064 seconds, excluding save/validation; peak allocation: 35.46 GB/GPU. Final model and tokenizer: format-s02/checkpoints/exp281-format-s02/step-2000 under the exp281 S3 prefix.

## Evaluation, reproducibility and next decision

The final evaluation and scoring succeeded on one RNO2A H100 in 472 seconds, sharing one 5.886 GB model download across both modes. All 400 candidates and 50 captured timing rows are accounted for; raw-output and scoring audits agree.

The 2.19 MB final report is public in the open-athena/MarinFold HF bucket at data/exp281/format-s02/report.zip. Anonymous download checksum verification passed. Raw histories, metrics, timings, checkpoint checksums, plots and run history preserve the result.

No synthesis or rejection round was launched. More repetitions of this small corpus are insufficient. A larger and more varied format corpus, shorter histories or stronger transition supervision are candidates for the next design, not tested remedies. The refreshed-SFT and rejection hypotheses remain open; FoldBench eval-test is untouched.
