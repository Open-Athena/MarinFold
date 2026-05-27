---
marinfold_experiment:
  issue: 34
  title: "exp: generate training dataset that includes pause tokens"
  kind: document_structures
  branch: exp/34-contacts-and-distances-v2
---

# exp: generate training dataset that includes pause tokens

**Issue:** [#34](https://github.com/Open-Athena/MarinFold/issues/34) · **Kind:** `document_structures` · **Branch:** `exp/34-contacts-and-distances-v2`

## Question

Can we improve task-level accuracy by adding `<think>` tokens at
inference time (assuming the model was trained on these)?

## Hypothesis

At inference time we can use `<think>` tokens to allow the model to
spend some more cycles without having to condition on additional
tokens. This might give an accuracy boost.

## Background

[Goyal et al. 2023, "Think before you speak: Training Language
Models With Pause Tokens"](https://arxiv.org/abs/2310.02226).

`contacts-and-distances-v1` is the document structure used to date —
see [`exp1_document_structures_contacts_and_distances_v1`](../exp1_document_structures_contacts_and_distances_v1/README.md)
for the canonical layout, vocabulary, and the published
`timodonnell/protein-docs-tokenizer@83f597d88e9b` tokenizer.

## Approach

This experiment **only implements training-data generation** — the
follow-up experiments will run it at scale (a separate issue) and
then train a model on the result (yet another issue).

### Document structure

Introduce a new doc structure `<contacts-and-distances-v2>`, byte-
identical to v1 except that `<think>` tokens are spliced in between
statements:

1. **Initial run.** With probability `0.75`, immediately after
   `<begin_statements>` emit `k1` think tokens where
   `k1 ~ Geometric(p=0.13)` (support ≥ 1). The other 25% of the
   time, emit none.
2. **Additional runs.** Sample `k2 ~ Uniform([-4, 4])`. Place
   `max(int(k2), 0)` additional `<think>` runs at random
   inter-statement slots (uniformly in `[0, N-1]`, with replacement).
   Each run's length is sampled independently as `Geometric(p=0.25)`.
3. **No splitting statements.** Runs always land *before* a statement
   opener — i.e. before `<distance>`, `<long-range-contact>`,
   `<medium-range-contact>`, or `<short-range-contact>` — never
   inside one.
4. **Context budget.** The 8192-token v1 budget is preserved. Think-
   token overhead is sampled *before* the statement budget is
   allocated and subtracted from the available tokens, so docs still
   end with `<end>` and fit in the context window.

When training on these later, the loss will be masked at `<think>`
positions (out of scope for this experiment).

### Vocabulary

The v2 tokenizer is built by appending two tokens to the v1 vocab:

| id range          | tokens                                                                                |
|-------------------|---------------------------------------------------------------------------------------|
| 0..1              | `<pad>`, `<eos>` (specials)                                                           |
| 2..2839           | the v1 domain vocab, **in the same order** (2838 tokens)                              |
| 2840              | `<contacts-and-distances-v2>` — the doc-structure marker                              |
| 2841              | `<think>` — the pause / scratch token                                                 |

Append-only is load-bearing: keeping all 2838 v1 token ids stable
means a v1-pretrained checkpoint can be warm-started on v2 by
growing its embedding table by 2 rows, instead of retraining the
tokenizer-projection layer from scratch.

### Layout

```
experiments/exp34_document_structures_contacts_and_distances_v2/
├── vocab.py         # NAME='contacts-and-distances-v2', THINK_TOKEN,
│                    # all_domain_tokens() = v1 ++ [<v2>, <think>]
├── parse.py         # gemmi PDB/mmCIF reader (copied from v1); fsspec/cloud
│                    # paths + in-memory mmCIF (parquet cif_content) support
├── generate.py      # v2 algorithm: think-overhead pre-sampling + slot placement
├── cli.py           # `generate` (Zephyr pipeline) + `tokenizer` subcommands
├── pyproject.toml   # marinfold + gemmi/numpy/pyarrow + marin-zephyr/fsspec/gcsfs/hf
└── tests/test_structure.py
```

`vocab.py` imports `all_domain_tokens` from the marinfold v1
subpackage and appends, so the v1/v2 prefix can never silently drift.
The CLI exposes `generate` and `tokenizer` only — inference and
evaluation surfaces will be added once a v2-trained model exists.

### Running it locally

```bash
uv sync --extra test

# Build / save / push the v2 tokenizer (includes <think>).
uv run python cli.py tokenizer --save-local /tmp/tok-v2/
uv run python cli.py tokenizer --push open-athena/contacts-and-distances-v2-tokenizer

# Generate v2 docs from a directory of mmCIFs / PDBs.
uv run python cli.py generate \
    --input /path/to/afdb-cifs/ --num-docs 100 \
    --out /tmp/sample-v2-docs.parquet

# Run tests.
uv run pytest tests/ -v
```

`generate` runs on Zephyr and auto-detects its backend: a local thread pool
off-cluster (as above), or Iris worker tasks when launched on the cluster.

### Generating documents with Zephyr on Iris

In a dedicated terminal, connect to the cluster:

```bash
uv run iris --cluster=marin cluster dashboard
```

The input is the [`timodonnell/afdb-1.6M`](https://huggingface.co/datasets/timodonnell/afdb-1.6M)
dataset: ~1,000 parquet shards (2,000 rows each, under `shard_000-999/`) whose
`cif_content` column holds the raw mmCIF text of one AlphaFold structure per row.

Smoke test first — a single shard (2,000 structures), capped to 100 docs in one file:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py generate --input "hf://datasets/timodonnell/afdb-1.6M/shard_000-999/shard_000000.parquet" --num-docs 100 --out "gs://marin-tmp-us-central1/marin-fold-tests/v2-corpus100.parquet" --worker-cpu 1 --worker-memory 4g
```

Then the full run — all 1.6M structures, one output parquet per input shard:

```bash
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- python cli.py generate --input "hf://datasets/timodonnell/afdb-1.6M/**/*.parquet" --out "gs://marin-us-east5/protein-structure/MarinFold/exp34/corpus_v1-{shard:05d}-of-{total:05d}.parquet" --worker-cpu 1 --worker-memory 4g --worker-disk 64g --max-workers 512
```

Keep each on **one line** — a backslash-continuation with a trailing space
silently truncates the command (everything after it leaks to your shell).

What the arguments do:

- **`--input`** accepts a `.parquet` file/glob (rows with mmCIF in `--cif-column`,
  default `cif_content`) or a `.cif`/`.pdb` file/dir/glob. `hf://`, `gs://`, `s3://`
  and local paths all work.
- **`--num-docs N`** caps the run to N documents. For parquet it reads every
  matched shard before truncating, so pair it with a single-shard `--input` (as
  above) for a cheap sample — not the full `**/*.parquet` glob.
- **`--out` with a `{shard}`** placeholder writes one parquet per input shard;
  drop it (or set `--num-docs`) for a single merged file.
- **Scale with `--max-workers`, not `--worker-cpu`.** Each shard's rows are
  parsed + generated single-threaded, so 1 CPU/worker is the efficient default;
  raising the CPU count just leaves cores idle. `--cpu`/`--memory` before `--`
  size the launcher; `--worker-*` after `--` size the workers.

Cancel a running job with `uv run iris --cluster=marin job stop <JOB_ID>` (find
the id via `iris --cluster=marin job list`).

Knobs that override the issue's pinned defaults are exposed as flags
(`--think-initial-prob`, `--think-initial-geom-p`,
`--think-additional-count-range LO HI`, `--think-run-length-geom-p`)
for ablations / sweeps in the follow-up at-scale experiment.

## Success criteria

We have a new document structure `contacts-and-distances-v2`:

1. New tokens `<contacts-and-distances-v2>` and `<think>` are appended
   to the v1 vocab (every v1 token id is preserved) — covered by
   `tests/test_structure.py::test_v1_ids_unchanged_under_v2_tokenizer`.
2. Generated docs fit in 8192, start with `<contacts-and-distances-v2>`,
   end with `<end>`, and tokenize without `<UNK>` — covered by
   `test_doc_fits_in_context_window` /
   `test_doc_starts_with_v2_marker_and_ends_with_end` /
   `test_doc_tokenizes_without_unk`.
3. `<think>` tokens never split a statement — they always land before
   a statement opener (or pLDDT bin) — covered by
   `test_think_never_breaks_within_a_statement`.
4. The sampling matches the issue spec: 0.75 gate on the initial run,
   `Geom(0.13)` for `k1`, `Uniform(-4, 4)` for `k2`, `Geom(0.25)` for
   each additional run — covered by the
   `test_sample_think_overhead_*` suite.
5. Generation is deterministic per `entry_id` (same RNG-stream order
   as v1, with the think-overhead draw inserted at the top) — covered
   by `test_doc_is_deterministic`.

## Results

`uv run pytest tests/ -v -m 'not network'` → 44 passed.

End-to-end smoke generating a doc from a 60-residue synthetic ALA
chain: produced a `<contacts-and-distances-v2>` doc that ended with
`<end>`, packed exactly 8192 tokens, and contained 15 `<think>`
tokens — including a 9-token run immediately after
`<begin_statements>` and before the first `<distance>` statement,
matching the issue's example layout.

## Conclusion

Training-data generation for `contacts-and-distances-v2` is
implemented and tested. Follow-up issues are needed to (a) run this
generator at scale over the AFDB-24M / `protein-docs-5x` subset, and
(b) train a model on the resulting dataset (with loss masking at
`<think>` positions).
