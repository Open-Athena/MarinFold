---
marinfold_experiment:
  issue: 40
  title: "exp: Generate Bio2Token documents on Zephyr"
  kind: document_structures
  branch: main
---

# exp: Generate Bio2Token documents on Zephyr

**Issue:** [#40](https://github.com/Open-Athena/MarinFold/issues/40) · **Kind:** `document_structures` · **Branch:** `main`

## Question

Does a commonly used neural tokenizer, like bio2token, make useful documents for protein prediction? Can we make use of the compressed representation to supplement context to predict other useful information (like secondary structures)? Can we do this at scale on Marin infrastructure (iris, zephyr, via a data loader)?

## Hypothesis

- Documents should be smaller
- They should do about as well as our custom document format
- Zephyr should be equipped to perform neural inference en masse.

## Background

See [parent issue](https://github.com/Open-Athena/MarinFold/issues/2).

## Approach

- Add [bio2token](https://github.com/flagshippioneering/bio2token/tree/main) as a `uv` dep in this experiment's pyproject.
- Based on our model for PDBs, write an adapter for our structure of data to their [expected pdb dict format](https://github.com/flagshippioneering/bio2token/blob/e3139ba655aa71e2afd0904ef46679b2796815d9/src/bio2token/data/utils/utils.py#L300).
- Create an efficient Zephyr pipeline (see the [Zephyr agent skill](../../.agents/skills/zephyr-pipeline-performance/SKILL.md)) that adapts our data source to inference batches (via the previous step), and make it perform inference via the [bio2token encoder](https://github.com/flagshippioneering/bio2token/blob/main/src/bio2token/models/encoder.py). The encoder outputs a 1d tensor of integers (tokens).
- Hydrate the model with the official [bio2token checkpoint](https://github.com/flagshippioneering/bio2token/blob/main/checkpoints/bio2token/bio2token_pretrained/epoch%3D0243-val_loss_epoch%3D0.71-best-checkpoint.ckpt).
- Update the torch backend to [target XLA/TPUs](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html).
- Write the token documents in a similar document format (i.e. parquet) with a similar chunking/shard structure as the [standard token documents](../exp1_document_structures_contacts_and_distances_v1/README.md). 
  - See this parquet store for reference: `gs://marin-us-central1/protein-structure/MarinFold/exp5/corpus_v2-{shard:05d}-of-{total:05d}.parquet`

## Success criteria

- Documents can be generated on GCS that are ready for experiments.
  - "Ready for experiments" means is reviewed and approved by Tim.
- (reach): a Dataloader can efficiently apply bio2token on a source data store to generate documents on the fly.

## Results

**bio2token runs end-to-end on Iris/Zephyr TPUs, emitting the self-describing
`bio2token-v2` documents described below.** The infrastructure was validated by
a full 22-shard `val` run; the document format was then redesigned per review
(see the [PR thread](https://github.com/Open-Athena/MarinFold/pull/114)), so the
published corpus predates the format change and is being **regenerated to v2**.

### Document format (`bio2token-v2`)

Each atom is a **self-describing triple** `<pN> <atom-name> <btC>` (residue
position, atom name, bio2token code), and the triples are **shuffled** into a
random per-document order so the model never has to index into a positional
stream; a `<pN> <RES>` sequence section precedes them:

    <bio2token-v2> <begin_sequence> <p0> <ASP> <p1> <ILE> …
      <begin_statements> <p51> <CB> <bt3891> <p6> <OD1> <bt3983> … <end>

Token strings (`<MET>`, `<CA>`, `<pN>`, the section markers) are **reused
verbatim from the contacts document structures** so bio2token documents share a
token space with them and can be mixed under one tokenizer; only
`<bio2token-v2>` and the 4096 `<bt*>` codes are minted here. The format is
**losslessly invertible** (a document decodes back to the exact per-atom codes
in canonical order), and decoding to coordinates reproduces the original CA
positions at **0.81 Å CA-RMSD** on 1QYS — bio2token's lossy-FSQ reconstruction
floor, i.e. the ceiling for any model trained on these tokens
(`decode.py`, `tests/test_decode.py`). The self-describing layout costs ~3
tokens/atom vs. the initial 1, a deliberate tradeoff for trainability.

Because ~3 tokens/atom means a large protein won't fit an 8k context, a
structure with more atoms than fit has its **atoms randomly sampled** (seeded
by entry id) down to the budget — the full residue sequence is kept, and the
atom budget is `(context_length − 4 − 2·n_residues) / 3`. Rows record
`num_atoms` (emitted), `num_atoms_total`, and a `truncated` flag. Only the rare
chain too long to position-number (> 2700 residues) is dropped outright.

### Infrastructure (measured)

- **Validation run:** 22 shards on **8 × v6e-4 preemptible** workers (us-east5-b),
  ~16 min end-to-end, 41,954 structures, 86.6 M atoms. All compute + manifest +
  output co-located in us-east5 → **zero cross-region egress**.
- **Throughput (1 v6e chip, XLA compile amortized):** ~**0.108 s/doc**,
  ~**20,300 atoms/s** (full 2,000-row shard). A short run is dominated by the
  one-time XLA compile (~7 graphs, one per length bucket): the 100-doc smoke was
  0.74 s/doc, mostly compile.

**How it works (the deployable pipeline).** bio2token's published encoder needs
CUDA-only kernels; exp40 reimplements the all-atom Mamba-1 encoder + FSQ in
**pure, XLA-compilable PyTorch** (`mamba.py`, `model.py`), loads the official
checkpoint exactly (round-trip RMSD 0.94 Å), and serves it on TPU via a
two-phase Zephyr worker (`generate_rows.py`): concurrent fetch+parse, then **one
batched forward pass per length bucket** (`tokenizer.py`). Three properties make
the TPU path work: an **associative (parallel) scan** replacing the sequential
recurrence (compact XLA graph, bit-exact), **static-length bucketing + masking**
(compile once per bucket; padded tokens are bit-for-bit equal to unpadded — see
`tests/test_bucketing.py`), and **token-budgeted batches** (`B·L ≤ 131072`) that
keep HBM flat (a fixed batch count OOMs the largest bucket on v6e). Provisioned
with `ResourceConfig.with_tpu("v6e-4", zone="us-east5-b", regions=["us-east5"])`.

**Open (throughput headroom, not correctness):** a worker uses only 1 of the
v6e-4's 4 chips — multi-chip fan-out (or a shared GCS XLA compile-cache) is the
lever for a cost-efficient full 4.2 M-doc run. See
[`data/timings.csv`](data/timings.csv).

### Reproduce

```bash
# Iris TPU smoke (1 worker, 100 docs) — validates the full path end-to-end.
uv run iris --cluster=marin job run --cpu 1 --memory 2GB --extra tpu -- \
  python cli.py generate \
    --input 'gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/selection_manifest/val/shard_00000.parquet' \
    --out   'gs://marin-us-east5/protein-structure/MarinFold/exp40/smoke/corpus.parquet' \
    --device xla --tpu-type v6e-4 --zone us-east5-b --num-docs 100 --max-workers 1

# Full split (per-shard output; scale --max-workers).
uv run iris --cluster=marin job run --cpu 1 --memory 2GB --extra tpu -- \
  python cli.py generate \
    --input 'gs://.../selection_manifest/val/shard_*.parquet' \
    --out   'gs://marin-us-east5/.../exp40/val/corpus-{shard:05d}-of-{total:05d}.parquet' \
    --device xla --tpu-type v6e-4 --zone us-east5-b --max-workers 8
```

> Region note: output/compute use **us-east5** (co-located with the input
> manifest and the v6e pool) rather than the issue's `us-central1` — the v5p
> pool in us-central1 was capacity-contended, and co-location keeps egress at
> zero. Tim's review needs the corpus, not a specific region.

## Conclusion

The reach question — "can Zephyr do neural inference en masse?" — is answered
**yes**: a checkpoint-faithful, pure-PyTorch bio2token tokenizer runs on TPU
through the existing Zephyr/Iris data-generation harness, producing a clean
`val` corpus with zero malformed documents. Whether the documents are *better*
(smaller, and as good as contacts-v1 for downstream prediction) is the next
question — it needs Tim's review of the published corpus plus a head-to-head on
a downstream eval. The train/test splits and any full-scale run should first add
multi-chip fan-out or a shared XLA compile-cache for cost efficiency.
