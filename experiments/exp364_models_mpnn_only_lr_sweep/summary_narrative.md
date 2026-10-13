## ProteinMPNN-only learning-rate sweep

Experiment #364 asks whether the default 1.47B contacts-v1 model can train on redesigned sequences alone.

162,574,724 documents: 31,702,680 AFDB + 130,872,044 ESM-Atlas designs.

## Controlled training recipe

Three learning rates: 0.0005, 0.001, 0.002. Two complete epochs, trained from scratch.

Preserve exp277 architecture, tokenizer, batch 128, context 8192, optimizer settings, WSD schedule fractions, validation and amino-acid order augmentation.

No native training documents. Fresh global shuffle each epoch.

## Coverage and execution

23,921,189 packed examples per epoch; 373,769 updates for two passes.

Live regional audit passed: both source ledgers, raw tokens, packing and clipping exactly match the default corpus subset. Six CPU tests passed.

The 64-H100 smoke passed ten updates with finite losses. Native checkpoint and HF weights plus tokenizer were verified.

All three production trials were launched on 32 H100s each, 96 total, using regional cached data. Saved W&B configurations match the default except for the requested changes.

Production startup is in progress; the two-epoch runs are not yet complete.
