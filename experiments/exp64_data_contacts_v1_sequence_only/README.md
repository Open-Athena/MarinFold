---
marinfold_experiment:
  issue: 64
  title: "exp: generate sequence-only contacts-v1 dataset"
  kind: data
  branch: exp/64-contacts-v1-sequence-only
---

# exp: generate sequence-only contacts-v1 dataset

**Issue:** [#64](https://github.com/Open-Athena/MarinFold/issues/64) · **Kind:** `data` · **Branch:** `exp/64-contacts-v1-sequence-only`

## Question

Can we improve performance on the contacts-v1 eval set by including sequence-only data?

For this experiment, we just want to generate the dataset.

## Hypothesis

N/A - this experiment is just data generation

## Background

We recently added contacts-v1, which has an initial section describing the sequence and a later section that defines the contacts:

```
<contacts-v1>
<begin_sequence> 
<p1320> <LEU> 
<p1230> <TYR>
<p1045> ...
<begin_statements>
<contact> <p1262> <p1196> 
<contact> <p1016>
...
<end>
```

Take a large sequence database (UniRef50) and emit one document per sequence:

```
<contacts-v1.sequence_only>
<begin_sequence>
<p1320> <LEU>
<p1230> <TYR>
<p1045> ...
<n-term> <p1318>
<c-term> <p1102>
<end>
```

i.e. **exactly the contacts-v1 sequence section** — same random wrap-around
indexing, `<n-term>`/`<c-term>` markers, and shuffled `<pX> <AA>` statements —
just under a new doc type and with **no structure section** (no contacts).
That shared representation is the point: the sequence-only corpus can be mixed
with the contacts-v1 corpus under one tokenizer, to (hypothesis, tested later)
improve the contacts-v1 eval.

## Approach

Two halves: a tiny **library change** (the reusable "mode") and a local
**generation driver** (this experiment).

### Library — `marinfold/.../document_structures/contacts_v1/` (minimal)

- `vocab.py`: mint `<contacts-v1.sequence_only>`, **appended last** in
  `all_domain_tokens()` (its own `sequence_only_tokens()` group) so every
  pre-existing token id is unchanged — the unified tokenizer is the
  contacts-v1 one + 1 token (2844 domain / 2846 with specials).
- `generate.py`: `GenerationConfig(sequence_only=True)` switches
  `build_document` to emit the sequence section only; the structure-free
  entry point `generate_sequence_only_document(sequence, entry_id=…)` maps a
  one-letter sequence → residues and calls the builder. **No pyconfind.**
- `parse.py`: `residues_from_sequence` (one-letter → 3-letter; non-standard →
  `<UNK>`).
- The sequence section is **byte-identical** to contacts-v1's for the same
  `entry_id` + residues (pinned in `tests/.../test_sequence_only.py`); see
  the contacts-v1 `SPEC.md` "Sequence-only variant" note.

### Generation — this dir (local, no pyconfind, no Iris)

- `generate_rows.py`: per-shard core — stream a `*.fasta.zst`, drop sequences
  outside `[2, 2000]` residues (contacts-v1's serializable range), generate a
  document per surviving record, bucket into an **arbitrary** `train`/`val`/
  `test` split by `sha1(entry_id) % 1000` (~99 / 0.5 / 0.5), and write typed,
  size-bounded sharded parquet per split.
- `cli.py`: `generate` (one worker process per UniRef50 shard, shards
  downloaded on demand, resumable via `_done/` markers), `tokenizer` (save the
  unified tokenizer), `inspect` (print docs + corpus stats).

```bash
uv sync --extra test && uv run python -m pytest tests/ -q   # unit tests

# Full run (61 shards, ~all of UniRef50): downloads + generates locally.
uv run python cli.py generate --out ~/exp64_out --workers 32 --cleanup-input
uv run python cli.py tokenizer --save-local ~/exp64_out/tokenizer
uv run python cli.py inspect  --out ~/exp64_out --num 5
```

### Source: `LiteFold/UniRef50`

60,315,044 sequences / 17.28 B residues across 61 `*.fasta.zst` shards
(~11.5 GB compressed). **The shards are globally sorted by length, longest
first**: shard 0 holds the giant sequences (15 k–49 k residues), and from
shard 2 on essentially everything is ≤ 1000 residues (shard 60 ≈ 32). So the
`> 2000`-residue sequences we must drop (they can't be indexed under the 2000
position-token wrap-around) are concentrated in shard 0 / early shard 1 — a
small fraction of the 60 M. Hashing the split on `entry_id` (not shard index)
keeps every split length-balanced despite the sort.

## Success criteria

We have the data as described above and uploaded to the MarinFold HuggingFace
bucket (`data/document_structures/contacts-v1.sequence_only/<split>/`) with the
unified tokenizer.

## Results

_In progress._ Library + driver implemented and unit-tested (lib: pure
`contacts_v1` suite green; driver: `tests/test_generate_rows.py` green,
byte-identical to `generate_sequence_only_document`). A streamed real-data
sample (shard 0 front) validated parsing/generation (0 % `<UNK>` residues,
well-formed docs). Full local generation + HF upload pending go-ahead (issue:
inspect locally before upload).

Open decisions before the full run: physical output order (length-sorted as
written vs. an added shuffle pass) and confirmation that `> 2000`-residue
sequences are dropped (not truncated).

## Conclusion

_(Fill in after results are in.)_
