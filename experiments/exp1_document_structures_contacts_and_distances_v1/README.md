---
marinfold_experiment:
  issue: 1
  title: "exp: contacts-and-distances-v1 DocumentStructure impl"
  kind: document_structures
  branch: main
---

# exp: contacts-and-distances-v1 DocumentStructure impl

**Issue:** [#1](https://github.com/Open-Athena/MarinFold/issues/1) · **Kind:** `document_structures` · **Branch:** `main`

## Question

How do we implement `contacts-and-distances-v1` as a MarinFold
first-class `DocumentStructure` impl, conforming to the Protocol in
`document_structures/marinfold_document_structures/interface.py`?

## Hypothesis

The format is already well-defined and battle-tested (used by every
model in `experiments/exp0_models_protein_docs_initial_port/`, and
backing the `timodonnell/protein-docs` HF dataset's
`contacts-and-distances-v1-5x` subset). Porting it into a
`DocumentStructure` impl is a mostly mechanical translation: the
doc-generation logic moves into `generate_documents`, the in-training
distogram-benchmark logic from `protein_distogram_eval.py` moves into
`evaluate`, and tokenizer construction (currently
`create_protein_tokenizer.py`) is auxiliary support.

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

Now that MarinFold has a `DocumentStructure` Protocol, this format
should be the first concrete impl. It becomes the reference for any
v2/v3 successors and the contract for downstream data-gen + eval
experiments.

## Approach

- Create `experiments/exp<N>_document_structures_contacts_and_distances_v1/`.
- Implement `structure.py` exposing `get_structure() -> DocumentStructure`. Attributes / methods:
  - `name = "contacts-and-distances-v1"`
  - `context_length = 8192`
  - `tokens()` — return the canonical 2840-token domain list (ported
    from `exp0_models_protein_docs_initial_port/create_protein_tokenizer.py`).
    The tokenizer is then built by
    `marinfold_document_structures.build_tokenizer(structure)`; no
    hardcoded tokenizer URL.
  - `iter_inputs(path)` — accept PDB / mmCIF / `.gz` / directory.
    AFDB inputs are mmCIF (gemmi-parsed); PDB targets are
    hand-rolled (port from `protein_distogram_eval.py`).
  - `iter_ground_truth(path)` — same parser; eval-time records may
    carry a precomputed GT contact map.
  - `generate_documents(input_records, *, context_length, num_docs)` —
    one doc per input. Layout per the HF dataset README: sequence,
    contact statements (long / medium / short, rank-ordered),
    distance statements (0.5 Å bins, sampled atom pairs), pLDDT bin
    (from AFDB B-factor). The `-5x` HF subset means "up to 5 AFDB
    entries per Foldseek structural cluster", a *data-pipeline*
    selection — not augmentation here.
  - `evaluate(model_path, ground_truth_records)` — vllm-backed
    rollout eval. Phase 0/1-style for v1 (forced-scaffold contact
    statements, consensus vs GT CB-CB ≤ 8 Å). Phase 6 SMC / Phase 11
    routing are follow-ups in separate eval experiments.
- Verify via the local CLI in `document_structures/`:
  - `marinfold-document-structure generate structure.py <sample-pdb-dir>
    --num-docs 100 --context-length 8192 --out /tmp/sample.parquet`
    — spot-check that generated docs match rows in the HF dataset.
  - `marinfold-document-structure evaluate structure.py
    gs://marin-us-east5/checkpoints/.../hf/step-31337
    <sample-pdb-dir> --out /tmp/eval.json`
    — confirm metrics align with what `protein_distogram_eval.py`
    produces for the same (target, N) pairs.

## Success criteria

1. `structure.py` defines a class satisfying the `DocumentStructure`
   Protocol (passes `isinstance(structure, DocumentStructure)` at
   load time via the `marinfold-document-structure` CLI).
2. The tokenizer built via
   `marinfold_document_structures.build_tokenizer(structure)` (from
   `structure.tokens()`) is byte-identical to the published
   `timodonnell/protein-docs-tokenizer@83f597d88e9b`.
3. Tokenizing a generated doc for 1QYS (top7) matches the token-ID
   sequence that `protein_distogram_eval.py` builds for the same
   structure today.
4. `evaluate()` on
   `protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-31337`
   reproduces the `eval/protein_dist/macro_loss` from W&B for a small
   subset of targets (within sampling tolerance).
5. Graduated as `document_structures/contacts_and_distances_v1/ →
   ../experiments/exp<N>_document_structures_contacts_and_distances_v1/`
   once results land.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
