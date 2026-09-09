## One model to test sequence redesign
Train contacts-v1 Qwen3 1.5B from scratch on native and ProteinMPNN-redesigned AFDB and ESM-Atlas documents. Companion to Zack's broader issue #274 sweep; tracked in issue #277.

## Fixed first-pass recipe
Use exp232's winning m2-p06 optimizer: LR 0.001, weight decay 0.2. Preserve 5.9522% AFDB / 94.0478% ESM sampling and split each source equally between native and redesigned documents. Train for 145,200 steps (152.253B tokens), retaining amino-acid order augmentation and the original WSD schedule.

## Placement and current status
Iris inventory identified US-EAST-02A as the regional home of all four inputs. At placement time, 208 H100s were unallocated across 26 nodes. The isolated 16-node / 128-H100 smoke succeeded: ten updates, finite losses, and verified native/HF step-9 checkpoints with tokenizer files. Full document tokenization is in progress. No model-quality result is available yet.

## Interpretation
Compare first with exp232's matched-budget native-only checkpoint. The longer-trained best model is a separate reference. One seed and one mixture can provide an early answer but cannot settle the broader mixture sweep.


## Validation completed
Both tokenizer smoke caches exactly match fresh tokenization: 160,000 AFDB documents / 21.58M tokens and 39,180 ESM documents / 40.62M tokens. Two tests check the resolved production configuration and separation of smoke/production cache identities. Full AFDB preparation uses streaming document-only reads and 32 GB workers after an 8 GB memory failure.
