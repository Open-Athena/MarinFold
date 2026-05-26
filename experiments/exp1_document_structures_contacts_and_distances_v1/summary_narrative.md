# Summary slides — contacts-and-distances-v1 document-structure impl

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Implementing `contacts-and-distances-v1` as a MarinFold first-class
document structure: a set of plain modules + a `cli.py` driver that
uses the shared toolkit in `marinfold_document_structures`. Layout:
`vocab.py`, `parse.py` (gemmi-backed PDB/mmCIF reader),
`generate.py` (doc generation), `inference.py` (predict + evaluate),
`cli.py` (argparse driver).

## Why

The format is already well-defined and battle-tested — used by
every model in `exp0_models_protein_docs_initial_port/`, backing
the `timodonnell/protein-docs` HF dataset's
`contacts-and-distances-v1-5x` subset. Porting it is a mostly
mechanical translation: doc-generation logic → `generate.generate_documents`,
in-training distogram-benchmark logic → `inference.evaluate`,
tokenizer construction → `build_tokenizer(all_domain_tokens())`.

This is the first MarinFold document-structure impl. It's the
reference for any v2/v3 successors and the contract for downstream
data-gen + eval experiments.

## Document layout (reference)

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

64 distance bins at 0.5 Å resolution; 3 contact modes by sequence
separation; 7 pLDDT bins; position tokens `<p0>`–`<p2700>`; vocab
2840 tokens.

## Success criteria

1. CLI subcommands `generate / infer / evaluate / tokenizer` parse
   and dispatch correctly (covered by `tests/test_structure.py`).
2. Tokenizer byte-identical to the published
   `timodonnell/protein-docs-tokenizer@83f597d88e9b`.
3. Tokenizing a generated 1QYS doc matches the token-ID sequence
   built by `protein_distogram_eval.py` today.
4. `evaluate(cfg)` on the 1B checkpoint reproduces the Phase 7c
   trend: zero-shot CA-CA MAE ~3.6 Å, dropping to ~1.4 Å with N=50
   seeded GT contacts.
5. Graduates as `document_structures/contacts_and_distances_v1/`.

## Results so far

_(Fill in as results come in.)_

## Conclusion

_(Fill in after results are in.)_
