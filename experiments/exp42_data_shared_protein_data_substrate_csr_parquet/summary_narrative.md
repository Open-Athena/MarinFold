# Summary slides — exp: shared protein-data substrate (CSR parquet) + on-the-fly training-time dataloader

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we build a shared protein-data substrate (a CSR-encoded parquet of parsed structures) + a generic on-the-fly training-time dataloader that lets us rapidly remix document formats across experiments — without re-parsing CIFs or pre-generating a per-format static corpus each time?

## Why

Yes. A precomputed CSR parquet of `ParsedStructure` rows + a `CSRDocumentDataset` parameterized by a `DocumentGenerator` callback gives every future doc-format experiment a one-line training integration over the same substrate, with per-epoch RNG variation as free data augmentation.

Already measured on the WIP branch (`u/alxmrs/exp5-jit`, stacked on #39):

- **~225 docs/sec per CPU** end-to-end (parquet read → reconstruct `ParsedStructure` → generate v2 doc) on real AFDB structures. At 16 dataloader workers per node that's ~3,600 docs/sec — comfortably enough to feed an 8-GPU training job consuming ~256 docs/sec/GPU.
- **CSR artifact is ~8.4 % of source CIF size** (~12× smaller). Extrapolated full 1.6M corpus: ~25-40 GB compressed, fits on training-node NVMe.
- **CSR-reader cost is 0.17 ms/structure** (zero-copy column buffers via pyarrow.dataset); the rest is the doc generator itself.
- **Byte-identity-through-CSR** test passes: `CIF → ParsedStructure → generate` produces docs SHA1-equal to `CIF → ParsedStructure → CSR → ParsedStructure → generate`. So the substrate is a faithful re-serialization, not a lossy compression — anything testable against exp34's reference oracle transitively holds through the CSR path.

## Results so far

_(Fill in as results come in.)_
