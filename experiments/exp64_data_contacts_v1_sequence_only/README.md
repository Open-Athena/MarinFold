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

Full corpus generated locally — 60 workers (one per UniRef50 shard), **30.5 min**
(`wall_seconds` 1827.6), **0 errors / 0 warnings**:

- **60,004,535 documents — 32,983,420,021 tokens** (~32.98 B) from all
  60,315,044 UniRef50 sequences.
- **310,509 (0.51%) dropped** for falling outside `[2, 2000]` residues (the
  `>2000`-residue giants, concentrated in source shards 0–1); **0**
  unserializable. Accounting is exact: `written + skipped_length =
  60,315,044` = every sequence.
- Splits (arbitrary, `sha1(entry_id) % 1000`): train **59,403,434** (32.65 B
  tok) / val **300,982** (165 M) / test **300,119** (165 M) ≈ 99.0 / 0.50 / 0.50 %.
- Mean ~550 `num_tokens`/doc (≈271 residues). Integrity re-verified: parquet
  rows sum to exactly **60,004,535** (301 / 61 / 61 files, 58 GB zstd).
- Format confirmed on real data: every doc is `<contacts-v1.sequence_only>
  <begin_sequence> … <n-term>/<c-term> … <end>`, **no** structure section,
  `num_tokens == 2*seq_len + 7`, 0 `<UNK>` leakage.

Counts: [`data/generation_counts.csv`](data/generation_counts.csv) (per shard)
and [`data/summary.json`](data/summary.json). Dataset card:
[`DATASET_README.md`](DATASET_README.md).

**Resolved open decisions:** (1) **physical order** — shipped as generated
(length-banded, since UniRef50 is globally length-sorted), documented in the
dataset card with a recommendation to shuffle at training time, rather than
risk a 58 GB shuffle pass on the unattended run; (2) `>2000`-residue sequences
are **dropped, not truncated** (310,509 `skipped_length`, 0 truncations).

**Upload status — pending an org-scoped token.** The dataset (train/val/test +
`tokenizer/` + `README.md`, 58 GB) is staged at `~/exp64_out`. Publishing to the
HF bucket `open-athena/MarinFold` → `data/document_structures/contacts-v1.sequence_only/`
is blocked because the workstation's active token (`write2`) is fine-grained to
`timodonnell` only and 403s on the open-athena Xet write endpoint (the other
stored token, `boltzgen-write`, is scoped to the `boltzgen` org). Finish with an
open-athena-write token active:

```bash
hf auth switch        # to a token with open-athena repo.write (or: hf auth login)
hf buckets sync ~/exp64_out \
  hf://buckets/open-athena/MarinFold/data/document_structures/contacts-v1.sequence_only \
  --exclude "_done/*"
```

(A `--dry-run` of exactly this sync is verified: 427 files / 58 GB, no `_done/`
markers.)

## Conclusion

The sequence-only contacts-v1 corpus is **generated and verified**:
**60,004,535 documents / ~32.98 B tokens** from UniRef50, byte-faithful to
`generate_sequence_only_document` (sequence section identical to contacts-v1),
under the appended `<contacts-v1.sequence_only>` token (unified 2846-token
tokenizer, every pre-existing id preserved). The issue's data-generation work is
done; the final publish to the `open-athena/MarinFold` bucket is staged and a
single `hf buckets sync` away once an open-athena-write token is active (the
available workstation token is scoped to `timodonnell` only).
