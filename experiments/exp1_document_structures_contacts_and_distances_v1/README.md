---
marinfold_experiment:
  issue: 1
  title: "exp: contacts-and-distances-v1 document-structure impl"
  kind: document_structures
  branch: main
---

# exp: contacts-and-distances-v1 document-structure impl

**Issue:** [#1](https://github.com/Open-Athena/MarinFold/issues/1) · **Kind:** `document_structures` · **Branch:** `main`

## Question

How do we implement `contacts-and-distances-v1` as a MarinFold
first-class document structure, conforming to the `Generator` +
`Inference` Protocols in
`document_structures/marinfold_document_structures/interface.py`?

## Hypothesis

The format is already well-defined and battle-tested (used by every
model in `experiments/exp0_models_protein_docs_initial_port/`, and
backing the `timodonnell/protein-docs` HF dataset's
`contacts-and-distances-v1-5x` subset). Porting it into a
document-structure impl is a mostly mechanical translation: the
doc-generation logic moves into `generate.py:V1Generator.run`, the
in-training distogram-benchmark logic from
`protein_distogram_eval.py` moves into
`inference.py:V1Inference.evaluate`, and tokenizer construction
(currently `create_protein_tokenizer.py`) is replaced by the shared
`build_tokenizer(component)` helper sourcing `tokens()` from the
canonical vocab in `_vocab.py`.

## Background

The format originated in
[`timodonnell/LlamaFold-experiments/experiments/exp6_contact_prediction/src/data.py`](https://github.com/timodonnell/LlamaFold-experiments/blob/main/experiments/exp6_contact_prediction/src/data.py).
Document layout (from the [HF dataset README](https://huggingface.co/datasets/timodonnell/protein-docs)):

```
<contacts-and-distances-v1>
<begin_sequence> <AA_1> ... <AA_n>
<begin_statements>
<long-range-contact> <p_i> <p_j>
<medium-range-contact> <p_i> <p_j>
<distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>
<short-range-contact> <p_i> <p_j>
<plddt_80_85>
<end>
```

Key constants:
- 64 distance bins at 0.5 Å resolution (`<d0.5>` … `<d32.0>`)
- 3 contact modes by sequence separation: long ≥ 24, medium 12–24, short 6–12
- Contact definition: CB–CB ≤ 8 Å (CA for GLY / missing CB)
- 7 pLDDT bins
- Position tokens `<p0>` … `<p2700>`
- Vocab: 2840 tokens

Pinned artifacts:
- Dataset: `huggingface.co/datasets/timodonnell/protein-docs`, subset
  `contacts-and-distances-v1-5x` (~5.39M docs from AFDB-24M, 5 per
  structural cluster).
- Tokenizer: `huggingface.co/timodonnell/protein-docs-tokenizer@83f597d88e9b`.
- Consumed by every model in `experiments/exp0_models_protein_docs_initial_port/`.
- In-training distogram benchmark already implemented at
  `experiments/exp0_models_protein_docs_initial_port/protein_distogram_eval.py`.

Now that MarinFold has `Generator` + `Inference` Protocols, this
format should be the first concrete impl. It becomes the reference
for any v2/v3 successors and the contract for downstream data-gen +
eval experiments.

## Approach

Layout (per the document-structures Generator + Inference split):

```
experiments/exp1_document_structures_contacts_and_distances_v1/
├── _vocab.py        # canonical 2838-token vocab + NAME + CONTEXT_LENGTH
├── _parse.py        # gemmi-backed PDB / mmCIF reader; ParsedStructure dataclass
├── generate.py      # V1Generator + get_generator()
├── inference.py     # V1Inference + get_inference() (predict + evaluate)
└── pyproject.toml
```

`_vocab.py` is the canonical vocab list, ported from
`exp0_models_protein_docs_initial_port/create_protein_tokenizer.py`.
Both `generate.py` and `inference.py` source `tokens()` from it —
test pins them equal.

`generate.py:V1Generator.run(args)` ports the document algorithm
from `timodonnell/contactdoc/contactdoc/generators/contacts_and_distances_v1.py`:
deterministic seed per `entry_id`, CB-CB contact eligibility
(CA for GLY), per-mode fraction sampling, distance statements over
uniform residue+atom pairs, rank-ordered statements, 50/50 pLDDT
placement. One doc per input. CLI flags: `--input`, `--num-docs`,
`--context-length`, plus the algorithm knobs
(`--contact-cutoff-angstrom`, `--residue-plddt-min`,
`--contact-f-range`, …). The `-5x` HF subset suffix is a
*data-pipeline* concern ("up to 5 AFDB entries per Foldseek
structural cluster"), not augmentation here.

`inference.py:V1Inference` powers both the `infer` and `evaluate`
CLI subcommands. Same per-pair query loop in both cases: walk pairs
(i, j) with i<j on each input structure, prompt with the v1 base
+ sequence + `<begin_statements>` + N seeded GT long-range contacts
+ `<distance> <p_i> <p_j> <atom_i> <atom_j>`, renormalize the
next-token distribution over the 64 `<d_X.X>` bins to an expected
distance.

- `predict(args)` yields per-(structure, n_seeded) records with
  `expected_distances` (and optionally the full `bin_probs` via
  `--keep-bin-probs`). No GT consulted.
- `evaluate(args)` runs the same loop, then scores against the
  GT distance matrix at the same atom pair (the input file IS the
  ground truth — its coordinates are the comparison target).
  Returns `mae_at_n<N>_angstrom` per seeded-contact count.

`--seed-n-values 0,5,20,50` sweeps over seeded counts in a single
run — reproduces the Phase 7c trace from the
LlamaFold-experiments notebook ("a few GT contacts as hints").
`--query-atom` (default `CA`) picks the atom on both i and j.

Verify locally via the CLI:

```bash
marinfold-document-structure generate \
    experiments/exp1_document_structures_contacts_and_distances_v1/ \
    --input /path/to/afdb-cifs/ --num-docs 100 \
    --out /tmp/sample-docs.parquet

marinfold-document-structure infer \
    experiments/exp1_document_structures_contacts_and_distances_v1/ \
    --model open-athena/<model> --input /path/to/seqs/ \
    --out /tmp/predictions.parquet

marinfold-document-structure evaluate \
    experiments/exp1_document_structures_contacts_and_distances_v1/ \
    --model open-athena/<model> --input /path/to/pdbs/ \
    --seed-n-values 0,5,20,50 \
    --out /tmp/metrics.json
```

## Success criteria

1. `generate.py` and `inference.py` both satisfy their respective
   `@runtime_checkable` Protocols (`Generator` / `Inference`) — i.e.
   `marinfold-document-structure` loads them without a Protocol
   error.
2. `gen.tokens() == inf.tokens()` (pinned in the test suite — both
   read from `_vocab.py`), and the tokenizer built via
   `build_tokenizer(gen)` is byte-identical to the published
   `timodonnell/protein-docs-tokenizer@83f597d88e9b`.
3. Tokenizing a generated doc for 1QYS (top7) matches the token-ID
   sequence that `protein_distogram_eval.py` builds for the same
   structure today.
4. `V1Inference.evaluate(args)` on
   `protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-31337`
   reproduces the Phase 7c trend from the LlamaFold-experiments
   notebook: zero-shot CA-CA MAE around 3.6 Å on the 12-target set,
   dropping to ~1.4 Å with N=50 seeded GT contacts. We'll consider
   the eval validated once `mae_at_n0_angstrom` and
   `mae_at_n50_angstrom` agree (within run-to-run noise) with the
   notebook's `r_d` rows.
5. Graduated as `document_structures/contacts_and_distances_v1/ →
   ../experiments/exp1_document_structures_contacts_and_distances_v1/`
   once results land.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
