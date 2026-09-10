## One model to test sequence redesign
Train contacts-v1 Qwen3 1.5B from scratch on native and ProteinMPNN-redesigned AFDB and ESM-Atlas documents. Companion to Zack's broader issue #274 sweep; tracked in issue #277.

## Fixed first-pass recipe
Use exp232's winning m2-p06 optimizer: LR 0.001, weight decay 0.2. Preserve 5.9522% AFDB / 94.0478% ESM sampling and split each source equally between native and redesigned documents. Train for 145,200 steps (152.253B tokens), retaining amino-acid order augmentation and the original WSD schedule.

## Placement and current status
Iris inventory identified US-EAST-02A as the regional home of all four inputs. At placement time, 208 H100s were unallocated across 26 nodes. The isolated 16-node / 128-H100 smoke succeeded: ten updates, finite losses, and verified native/HF step-9 checkpoints with tokenizer files. Full redesigned tokenization is complete: 162.6 million documents / 174.1 billion tokens. The production run was submitted at 19:32 UTC on 2026-09-09 and passed startup verification: step 23, finite loss 6.70270, 0.864 seconds/step on all 128 H100s. Training is in progress. The first full LM validation passed at step 2,114 (loss 3.76525). Contact-prediction quality is not yet evaluated.

## Interpretation
Compare first with exp232's matched-budget native-only checkpoint. The longer-trained best model is a separate reference. One seed and one mixture can provide an early answer but cannot settle the broader mixture sweep.


## Validation completed
Both tokenizer smoke caches exactly match fresh tokenization: 160,000 AFDB documents / 21.58M tokens and 39,180 ESM documents / 40.62M tokens. Two tests check the resolved production configuration and separation of smoke/production cache identities. Full AFDB preparation used streaming document-only reads and 32 GB workers after an 8 GB memory failure. ESM finished with 512 workers at 8 GB each. Both final cache totals were verified before production submission.

## Thirty-percent milestone
At 07:01 UTC on 2026-09-10 the run reached step 43,870 / 145,200. Validation loss declined across twenty passes from 3.76525 to 3.12627. Permanent checkpoints at steps 14,520, 29,040, and 43,560 are saved. The stable LR is 0.001. No production restarts.
