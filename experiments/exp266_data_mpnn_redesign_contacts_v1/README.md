---
marinfold_experiment:
  issue: 266
  title: 'exp: ProteinMPNN-redesign the decontaminated contacts-v1 training set — 8 sequences per backbone'
  kind: data
  branch: exp/266-mpnn-redesign
---

# exp: ProteinMPNN-redesign the decontaminated contacts-v1 training set — 8 sequences per backbone

**Issue:** [#266](https://github.com/Open-Athena/MarinFold/issues/266) · **Kind:** `data` · **Branch:** `exp/266-mpnn-redesign`

## Question

Does adding ProteinMPNN-redesigned sequences for backbones we already have
improve contacts-v1 — and is inverse-folding augmentation a cheaper substitute
for generating novel structures?

Concretely: take all **3,963,003** documents of the decontaminated AFDB corpus
([`contacts_v1_decontam`](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/document_structures/contacts_v1_decontam),
[#225](https://github.com/Open-Athena/MarinFold/issues/225)), redesign each
backbone into **8 new sequences**, re-run pyconfind on each, and publish the
resulting **~31.7 M-document** corpus.

This issue ships the corpus. The accuracy question is a follow-up
`kind/models` issue against the [#232](https://github.com/Open-Athena/MarinFold/issues/232)
decontaminated recipe.

## Hypothesis

Redesign augmentation supplies a contrast the current corpus cannot: today
every fold appears with exactly one sequence, so the model has no way to
separate "this contact is geometry" from "this contact is this rotamer". Eight
sequences per backbone provides exactly that contrast, at essentially the cost
of the pyconfind pass alone.

## Background: why this and not a structure generator

This is the cheap null version of a larger idea — an unconditional backbone
generator (BoltzGen / Proteina / RFdiffusion3) plus ProteinMPNN to manufacture
millions of *novel* training structures. That version costs GPU-weeks and
changes two things at once (novel folds **and** MPNN-flavoured sequences).
This one changes only the second, so a flat or negative result here is strong
evidence against the expensive version — and a positive one validates the whole
pyconfind-on-designed-sequences path before any generator is bought.

Cost of the generator version for reference (3 M monomers, L≈256, H100-hours,
generation only): distilled Proteina 75, Proteina-400-step 1,750,
RFdiffusion3 ~2,500, **BoltzGen ~15,000**. BoltzGen is the wrong tool — a
conditional *binder-design* model at ~24 s/design on an H100, most of whose
machinery we would not use.

**The prior in this repo is not encouraging.**
[#120](https://github.com/Open-Athena/MarinFold/issues/120) is the closest
thing we have run and synthetic documents lost: regenerated (even
only-correct-filtered) docs vs re-epoching the original at matched budget
scored R-precision **0.330 vs 0.350** overall and **0.262 vs 0.279** on long
proteins, two orders of magnitude above
[#204](https://github.com/Open-Athena/MarinFold/issues/204)'s 0.0023 noise
floor. The difference here is that #120 regenerated the *documents* by
self-distillation over the same structures; this regenerates the *sequences*
with an independent, structure-aware model MarinFold has never seen.

**The headroom argument for doing it at all.**
[#139](https://github.com/Open-Athena/MarinFold/issues/139)'s 66.8 M ESM-Atlas
structures are already the 40 %-identity linclust representatives, so we are
near the end of readily available *non-redundant* real predicted structures.
Inverse-folding augmentation is the only axis that multiplies the corpus
without either new structures or new redundancy.

## What was verified before writing the pipeline

### 1. pyconfind ignores input side chains — bit-identically

confind's contact degree is a rotamer-ensemble quantity: it rebuilds side
chains from the Dunbrack library rather than reading the ones in the file.
Stripping to backbone atoms (`N/CA/C/O`, no `CB`) and re-running
`contacts_v1.parse.analyze_structure` (`probe_pyconfind.py`):

| structure | residues | contacts (all-atom) | backbone-only | identical pair set | max Δdegree |
|---|---|---|---|---|---|
| 1crn | 46 | 57 | 57 | ✅ | 0.000 |
| 1ubq | 76 | 162 | 162 | ✅ | 0.000 |
| 101m | 154 | 352 | 352 | ✅ | 0.000 |
| 1mbn | 153 | 327 | 327 | ✅ | 0.000 |

**A backbone plus a residue-name assignment is a complete pyconfind input.**
So the redesigned corpus is computed under exactly the same contact operator as
`contacts_v1` — no silent train-distribution mismatch — and all-atom generators
would buy us nothing for this document format.

Pinned by `tests/test_backbone.py`, which asserts the stronger property the
pipeline actually depends on: a document built from *stripped backbone + native
sequence written back on* is **byte-identical** to the document built from the
untouched structure.

### 2. The contact label is strongly sequence-dependent at fixed geometry

Holding the backbone fixed, changing only residue names, applying the
contacts-v1 selection rule (`degree ≥ 0.001`, `|i−j| ≥ 6`)
(`probe_seq_sensitivity.py` → `data/sequence_sensitivity_probe.csv`):

| backbone | native contacts | shuffled native | random uniform | poly-LEU | poly-ALA | poly-GLY |
|---|---|---|---|---|---|---|
| 1crn (46 res) | 28 | J=0.364 | J=0.341 | J=0.463 | 0 contacts | 0 contacts |
| 1ubq (76 res) | 67 | J=0.409 | J=0.426 | J=0.586 | 0 contacts | 0 contacts |
| 101m (154 res) | 130 | J=0.353 | J=0.313 | J=0.492 | 1 contact | 0 contacts |

Only ~43–54 % of native contacts survive a sequence shuffle on an identical
backbone. **The 8 redesigns are not 8 near-duplicates** — each is a genuinely
different document, and pyconfind genuinely has to be re-run per
(backbone, sequence) pair rather than copied from the parent.

It also flags the risk: contact count collapses for small side chains, so an
Ala-rich design distribution would systematically shorten the documents. That
is measured below.

## Approach

Two stages, following [#53](https://github.com/Open-Athena/MarinFold/issues/53)'s
pipeline shape.

**Stage A — keep-list** (`select_backbones.py`, local, minutes). Read
`entry_id` + provenance from the published `contacts_v1_decontam/train` corpus
(2,067 shards; schema verified) and emit a length-sorted Stage-B manifest.
`gcs_uri` is rebuilt from `entry_id` rather than joined back to the
12,005-shard afdb-24M manifest, since AFDB's layout makes it a pure function of
the accession.

Decontamination is **inherited, not re-derived**: the keep-list is read out of
the published decontaminated corpus, so we only ever redesign a backbone that
survived #225, and a redesigned sequence cannot reintroduce an eval *sequence*
because it is a new sequence.

**Stage B — redesign + documents** (`cli.py` → `generate_rows.py`, one Iris
job). Per row: fetch the cif once (gzip-safe `read_object_bytes`), strip to
backbone, run ProteinMPNN `v_48_020` for 8 designs, relabel and call
`generate_document` 8 times. One fetch, one parse, 8 documents.

8 designs per backbone on a **temperature ladder**
`{0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5}`, recorded per row alongside
`mpnn_score` and `identity_to_native`, so a training experiment can subset the
near-native ↔ diverse axis without regenerating anything.

### Three engineering decisions the measurements forced

**One job on CPU, not a GPU job plus a CPU job.** ProteinMPNN is a 1.7 M-param
GNN whose decode is `L` sequential steps, so the GPU path is launch-bound, not
compute-bound: a single CPU core is only ~18× slower per sequence than an RTX
A5000. Splitting into "redesign on CoreWeave, documents on Iris" would stage
4 M AFDB cifs cross-cloud from GCS to S3, fetch every structure twice, and run
two jobs — to save CPU hours the Iris pool has (#139 scaled to 512 workers in
~10 minutes). `--device cuda` remains available if the pilot disagrees.

**Fold the 8 design slots into the batch dimension.** ProteinMPNN uses
`temperature` only in broadcast-compatible divisions, so a `[B, 1]` tensor
works where a scalar is expected — one `sample()` call produces all 8 designs
at 8 different temperatures. Measured on an A5000 at L=154: 10.38 s → 1.30 s
for 8 designs. `tests/test_redesign.py::test_per_item_temperature_matches_scalar`
asserts bit-equality with the scalar path, because a silent failure here would
draw every sequence at the wrong temperature.

**Exact-length batches** (`redesign.batch_by_exact_length`). Not an
optimization but a correctness requirement: `tied_featurize` pads
`omit_AA_mask` (an `[L, 21]` array) with the 1-D pad spec `[[0, L_max - l]]`,
widening the *alphabet* axis, and raises on any batch of mixed lengths
(`could not broadcast input array from shape (476,426) into shape (476,21)`).
Upstream never hits it because the stock CLI batches N copies of one protein.
Rather than monkey-patch a dependency we feed it only the shape it handles,
which also means zero padding waste. Stage A's length sort is what keeps the
equal-length groups full.

## Measurements so far (local, RTX A5000 + 1 CPU core)

`data/local_smoke_throughput.csv`, from `smoke_local.py --limit 48` and the
timing scripts.

| path | ms/sequence | note |
|---|---|---|
| GPU, one `sample()` per design, B=1 | 1298 | what the stock CLI does |
| GPU, designs folded in, B=1 | 162 | 8× from the batch-dimension fold |
| GPU, designs folded in, B=8 (L=154) | 33 | +another 5× from batching |
| GPU, designs folded in, B=32 (L=46) | 6.3 | |
| **CPU, 1 thread, designs folded in (L=154)** | **602** | 4.82 s per backbone for 8 designs |
| pyconfind → document, 1 thread | 463 ms/**document** | at L≈250 |

Device memory scales cleanly at **0.52 MB per padded residue** in the effective
batch, independent of L and B — that is what `--max-batch-residues` bounds.

### Composition drift — the risk flagged in the issue

48 monomers, 384 designs (`data/local_smoke_composition.csv`). ProteinMPNN's
known biases are present and in the expected direction, but modest — the
largest single shift is 2.2 percentage points:

| over-used | Δpp | under-used | Δpp |
|---|---|---|---|
| A | +2.06 | Q | −2.19 |
| P | +1.70 | M | −1.51 |
| E | +1.54 | S | −1.35 |
| L | +1.29 | H | −1.21 |
| K | +1.09 | C | −0.58 |

And the number that actually matters, given that contact count collapses for
small side chains:

**contacts per residue: native 0.930, designed 0.900 — ratio 0.968.**

So the feared artifact (Ala-rich designs producing systematically shorter
documents) is real but small at ~3 %. To be re-measured on the pilot at
corpus scale and on AFDB rather than PDB structures.

Temperature behaves as intended, though the ladder spreads gently
(`data/local_smoke_temperature.csv`):

| T | mean identity to native | mean MPNN score |
|---|---|---|
| 0.1 | 0.467 | 0.845 |
| 0.2 | 0.457 | 0.876 |
| 0.3 | 0.451 | 0.919 |
| 0.5 | 0.434 | 1.088 |

0.467 recovery at T=0.1 matches ProteinMPNN's published ~50 %.

## Projected full-run cost — **the gate**

Corpus mean `seq_len` ≈ 280 (verified against the published corpus: 2,067
shards, shard 0 mean 291.4 / median 221). Scaling the measured per-core rates:

| stage | core-hours |
|---|---|
| ProteinMPNN, 8 designs × 3.96 M backbones | ~9,700 |
| pyconfind → 31.7 M documents | ~4,600 |
| **total** | **~14,300 CPU core-hours** |

≈ **28 h on 512 workers**, ≈ **14 h on 1,000**. For scale, #139 was ~2,850
core-hours, so this is ~5× the largest data job we have run.

The alternative (`--device cuda` for the redesign on CoreWeave) cuts the
ProteinMPNN term to roughly 100 H100-hours but adds a cross-cloud staging of
4 M cifs, a second fetch of every structure, and a second job, while still
needing the ~4,600 core-hours of pyconfind on Iris.

Expected output: **31,704,024 documents, ~39 B tokens, ~95–110 GB** ZSTD
parquet.

**Nothing has been launched.** Per the `zephyr-pipeline-performance` skill the
next step is a 20 k-backbone Iris smoke to replace these workstation numbers
with cluster ones, then a user go-ahead before the full run.

## Success criteria

1. **Fidelity.** Native-sequence relabelling reproduces the parent document
   byte-for-byte (`tests/test_backbone.py`) — ✅ on 5 structures.
2. **Completeness.** 3,963,003 × 8 = 31,704,024 documents, drops counted and
   reported by reason, fail-loud per the pipeline skill.
3. **Composition check.** AA frequency and contacts-per-residue vs native, per
   temperature, reported whatever it shows — ✅ locally (ratio 0.968), to be
   repeated at pilot scale.
4. Published to the public HF bucket with a `DATASET_README.md` and a
   reproducible `publish_to_hf.py`.

## Risks

- **The labels assume the design folds.** MPNN sequences are not refolded.
  Unlike the generator version the backbone is a real AFDB fold and the
  sequence is a high-likelihood design for it, so this is the mildest form of
  that assumption — but a refold-and-check on a 20 k subsample (~5 GPU-h with
  ESMFold) would measure the per-sequence self-consistency rate and is worth
  folding into the pilot.
- **`global_plddt` is inherited from the parent AFDB structure.** Stripping to
  backbone keeps CA, so contacts-v1 recomputes the same value. It correctly
  describes confidence in the *backbone* we reused, and says nothing about
  whether the designed sequence folds there.
- **T=0.5 may be past the designability cliff.** That is what the recorded
  `mpnn_temperature` column is for.
- **#120 says synthetic documents start at a deficit here.**

## Files

| file | role |
|---|---|
| `backbone.py` | strip-to-backbone / relabel / coordinate extraction — the verified core |
| `redesign.py` | ProteinMPNN wrapper: exact-length batches, folded design dimension |
| `select_backbones.py` | Stage A — keep-list from the decontaminated corpus |
| `generate_rows.py` | Stage B per-row worker (no zephyr import) |
| `cli.py` | Stage B driver — Zephyr `map_shard` on Iris |
| `smoke_local.py` | local end-to-end: throughput, composition, contact density |
| `probe_pyconfind.py` | does confind need side chains? (no) |
| `probe_seq_sensitivity.py` | how much does the label move with sequence? (a lot) |
| `tests/` | fidelity regression + Stage-B tests (17 passing) |

## Results

Pending — nothing has run on the cluster yet.

## Conclusion

Pending.
