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
├── parse.py         # gemmi PDB/mmCIF reader (copied from v1; same atom vocab)
├── generate.py      # v2 algorithm: think-overhead pre-sampling + slot placement
├── cli.py           # `generate` + `tokenizer` subcommands
├── pyproject.toml   # marinfold path dep, gemmi, numpy, pyarrow
└── tests/test_structure.py
```

`vocab.py` imports `all_domain_tokens` from the marinfold v1
subpackage and appends, so the v1/v2 prefix can never silently drift.
The CLI exposes `generate` and `tokenizer` only — inference and
evaluation surfaces will be added once a v2-trained model exists.

### Running it

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
