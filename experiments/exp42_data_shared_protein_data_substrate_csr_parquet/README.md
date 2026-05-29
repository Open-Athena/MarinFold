---
marinfold_experiment:
  issue: 42
  title: "exp: shared protein-data substrate (CSR parquet) + on-the-fly training-time dataloader"
  kind: data
  branch: u/alxmrs/exp5-jit
---

# exp: shared protein-data substrate (CSR parquet) + on-the-fly training-time dataloader

**Issue:** [#42](https://github.com/Open-Athena/MarinFold/issues/42) · **Kind:** `data` · **Branch:** `u/alxmrs/exp5-jit`

## Question

Can we build a shared protein-data substrate (a CSR-encoded parquet of parsed structures) + a generic on-the-fly training-time dataloader that lets us rapidly remix document formats across experiments — without re-parsing CIFs or pre-generating a per-format static corpus each time?

## Hypothesis

Yes. A precomputed CSR parquet of `ParsedStructure` rows + a `CSRDocumentDataset` parameterized by a `DocumentGenerator` callback gives every future doc-format experiment a one-line training integration over the same substrate, with per-epoch RNG variation as free data augmentation.

Already measured on the WIP branch (`u/alxmrs/exp5-jit`, stacked on #39):

- **~225 docs/sec per CPU** end-to-end (parquet read → reconstruct `ParsedStructure` → generate v2 doc) on real AFDB structures. At 16 dataloader workers per node that's ~3,600 docs/sec — comfortably enough to feed an 8-GPU training job consuming ~256 docs/sec/GPU.
- **CSR artifact is ~8.4 % of source CIF size** (~12× smaller). Extrapolated full 1.6M corpus: ~25-40 GB compressed, fits on training-node NVMe.
- **CSR-reader cost is 0.17 ms/structure** (zero-copy column buffers via pyarrow.dataset); the rest is the doc generator itself.
- **Byte-identity-through-CSR** test passes: `CIF → ParsedStructure → generate` produces docs SHA1-equal to `CIF → ParsedStructure → CSR → ParsedStructure → generate`. So the substrate is a faithful re-serialization, not a lossy compression — anything testable against exp34's reference oracle transitively holds through the CSR path.

## Background

- This work was prototyped on top of #39 (exp5: write v2 docs to HF) on branch `u/alxmrs/exp5-jit`. @timodonnell's review feedback on #39 was that the on-the-fly / training-time pipeline belongs in a *separate* experiment, with exp5 staying focused on its issue-#5 scope (publish v2 docs to HF). This issue is that separate experiment.
- The substrate is what makes the long-term "rapidly remix doc formats" property work: one canonical published parquet on GCS, every new doc-format experiment composes it with a 3-line `CSRDocumentDataset(dataset_path=..., generator=<bespoke>)`.
- Per `AGENTS.md`'s graduation rule, the shared modules (`csr_store.py`, `dataset.py`, the `ParsedStructure` schema in `parse.py`) live inside this experiment for now and **graduate to a new top-level `data/` kind library** when a *second* experiment uses them (most likely an "actually train a model from on-the-fly docs" experiment — natural follow-on).

## Approach

Self-contained experiment dir with the substrate's reader/writer + the
training-time dataloader. **Doc-format-agnostic by design** — no
generator is bundled; every caller supplies one.

### Layout

```
experiments/exp42_data_shared_protein_data_substrate_csr_parquet/
├── README.md          # prose record
├── parse.py           # ParsedStructure (frozen dataclass) + gemmi extractor
├── vocab.py           # AMINO_ACIDS + ATOM_NAMES (re-exported from marinfold v1)
├── csr_store.py       # CSR parquet schema + zero-copy batched reader
├── dataset.py         # CSRDocumentDataset + DocumentGenerator callback API
├── cli.py             # `parse-to-csr` Zephyr subcommand (precompute only)
├── pyproject.toml     # gemmi/numpy/pyarrow/marin-zephyr/fsspec/gcsfs + optional torch
└── tests/             # roundtrip + dataloader plumbing (callback + subclass)
```

### Pipeline shape

1. **One-time precompute** — `cli.py parse-to-csr` runs as a Zephyr
   job over the AFDB manifest, fetches CIFs concurrently per shard,
   parses with gemmi, writes CSR parquet shards into
   `gs://marin-us-central1/protein-structure/MarinFold/csr-v1/`.
   Inherits exp5's thread-pool-fetch + in-region-bucket lessons.
2. **Training-time read** — `CSRDocumentDataset(dataset_path=..., generator=...)`
   is a `torch.utils.data.IterableDataset` (or a plain iterable when
   torch isn't installed). Reads through `pyarrow.dataset` with
   predicate pushdown (`filter=pc.field("split") == "train"`), column
   projection, and native fragment discovery for `gs://` / local /
   `file://`. The user passes one path (directory, single file, or
   glob); pyarrow handles the rest.
3. **Two extension points** for varying doc *type* / *shape*:
   - **callback** (composition): pass any
     `Callable[[ParsedStructure], str | None]` to `generator=`. This is
     where the doc-format experiment plugs in (e.g. exp5 exposes a
     `v2_generator(cfg, context_length)` factory matching this signature).
   - **subclass** (template method): override
     `CSRDocumentDataset._structure_to_doc` to change the *output dict
     shape* — e.g. attach token ids, surface a passthrough column, yield
     multiple variants per structure.

### Running it locally

```bash
cd experiments/exp42_data_shared_protein_data_substrate_csr_parquet
uv sync --extra test
uv run pytest tests/ -v
```

### Precompute on Iris (one-time)

In a dedicated terminal:

```bash
uv run iris --cluster=marin cluster dashboard
```

Smoke test — single input shard, capped to 100 structures, single output file:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py parse-to-csr --input "hf://datasets/timodonnell/afdb-1.6M/shard_000-999/shard_000000.parquet" --num-structures 100 --out "gs://marin-tmp-us-central1/marin-fold-tests/exp42-smoke.parquet" --worker-cpu 1 --worker-memory 4g --fetch-concurrency 32
```

Full run — all 1.6M structures, one CSR parquet per input shard:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py parse-to-csr --input "hf://datasets/timodonnell/afdb-1.6M/**/*.parquet" --out "gs://marin-us-central1/protein-structure/MarinFold/csr-v1/csr-{shard:05d}-of-{total:05d}.parquet" --worker-cpu 1 --worker-memory 4g --worker-disk 64g --max-workers 512 --fetch-concurrency 32
```

Keep each command on **one line** — a trailing space after a backslash
silently truncates the command and the rest leaks to your shell.

### Reading from a downstream experiment

Any experiment can compose the substrate with its own doc format in
three lines (the example uses exp5's v2 generator):

```python
# In a doc-format experiment (e.g. exp5):
from generate import generate_one, GenerationConfig

def v2_generator(cfg=None, context_length=8192):
    cfg = cfg or GenerationConfig()
    def _gen(structure):
        return generate_one(structure, context_length=context_length, cfg=cfg)
    return _gen

# At the training-job call site:
from dataset import CSRDocumentDataset  # exp42's dataloader
ds = CSRDocumentDataset(
    dataset_path="gs://marin-us-central1/protein-structure/MarinFold/csr-v1/",
    generator=v2_generator(),
    epoch=current_epoch,                  # bump between epochs for fresh draws
)
```

## Success criteria

1. **Column-level roundtrip holds.** `test_csr_roundtrip_preserves_shapes_and_values`
   passes — every CSR column survives parquet write + read with bit-equal
   dtype, shape, and values. This is *the* substrate contract: any pure
   function of the columns (including any doc generator) is byte-identical
   through CSR if this passes.
2. **Throughput ≥ 150 docs/sec/CPU** end-to-end on real AFDB structures
   with a representative generator (measured at ~225 docs/sec/CPU on the
   prototype with the v2 generator).
3. **CSR artifact published** for the full 1.6M-row corpus to
   `gs://marin-us-central1/protein-structure/MarinFold/csr-v1/`, size ≤ 50 GB,
   total `parse-to-csr` wall-clock ≤ 30 min on the Iris cluster at 512 workers.
4. **Remixability demonstrated** — at least one non-v2 `DocumentGenerator`
   (even a trivial test fixture) composes cleanly with the same dataloader
   + same CSR shards. Pinned by
   `tests/test_csr_store.py::test_dataset_accepts_custom_generator_callback`.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
