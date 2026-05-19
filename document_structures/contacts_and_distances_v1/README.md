# contacts-and-distances-v1

The first MarinFold document structure. Serializes each protein as
a sequence of amino-acid tokens followed by a stream of distance and
contact statements over residue pairs. Distance bins quantize to
0.5 Å resolution (64 bins covering 0.5–32.0 Å).

The historical experiment that defined this format lives at
[`../../experiments/exp1_document_structures_contacts_and_distances_v1/`](../../experiments/exp1_document_structures_contacts_and_distances_v1/)
and stays frozen as the historical record. This directory is the
working version that future fixes and feature work land in.

## Document layout

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

- 64 distance bins at 0.5 Å (`<d0.5>` … `<d32.0>`)
- 3 contact modes by sequence separation:
  long ≥ 24, medium 12–24, short 6–12
- Contact definition: Cβ–Cβ ≤ 8 Å (Cα for GLY / missing Cβ)
- 7 pLDDT bins
- Position tokens `<p0>` … `<p2700>`
- Total domain vocab: 2838 tokens

## Layout

```
document_structures/contacts_and_distances_v1/
├── pyproject.toml
├── README.md
├── src/
│   └── contacts_and_distances_v1/
│       ├── __init__.py
│       ├── cli.py           # local CLI: generate / infer / evaluate / tokenizer
│       ├── vocab.py         # canonical token list + NAME + CONTEXT_LENGTH
│       ├── parse.py         # gemmi-backed PDB / mmCIF reader
│       ├── generate.py      # generate_documents() + GenerationConfig
│       └── inference.py     # predict() + evaluate() + InferenceConfig
└── tests/
```

## Usage

There are two ways to run inference and evaluation against this
document structure:

### High-level (recommended): the top-level `marinfold` CLI

```bash
cd marinfold
uv sync --extra mlx        # or --extra vllm, or --extra transformers

uv run marinfold infer \
    --backend mlx \
    --input-sequence SIINFEKLLLSKP \
    --out /tmp/preds.json

uv run marinfold evaluate \
    --backend mlx \
    --input-dir /path/to/pdbs/ \
    --out /tmp/preds.json \
    --metrics-out /tmp/metrics.json
```

`marinfold` looks up the default model from `MODELS.yaml`, picks
the first document structure that model supports
(contacts-and-distances-v1), and dispatches to this package's
`predict` / `evaluate`.

### Low-level: this package's own `cli.py`

For direct access to all impl-specific flags (seeded-contact sweeps,
top-k logprobs, distance cap):

```bash
cd document_structures/contacts_and_distances_v1
uv sync --extra eval-mlx

uv run python -m contacts_and_distances_v1.cli evaluate \
    --backend mlx --model 1B \
    --input /path/to/pdbs/ --seed-n-values 0,5,20,50 \
    --out /tmp/metrics.json
```

(`--backend` selects vLLM / transformers / MLX; see
[`../../marinfold/README.md`](../../marinfold/README.md) for the
backend matrix.)

## Public API

```python
from contacts_and_distances_v1 import (
    NAME, CONTEXT_LENGTH, all_domain_tokens,
    GenerationConfig, generate_documents,
    InferenceConfig, predict, evaluate,
)
```
