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
first-class document structure — a set of plain modules + a `cli.py`
driver that uses the shared toolkit in `marinfold_document_structures`?

## Hypothesis

The format is already well-defined and battle-tested (used by every
model in `experiments/exp0_models_protein_docs_initial_port/`, and
backing the `timodonnell/protein-docs` HF dataset's
`contacts-and-distances-v1-5x` subset). Porting it is a mostly
mechanical translation: the doc-generation logic becomes
`generate.generate_documents`, the in-training distogram-benchmark
logic from `protein_distogram_eval.py` becomes `inference.evaluate`,
and tokenizer construction (currently `create_protein_tokenizer.py`)
is replaced by `build_tokenizer(all_domain_tokens())` from the
canonical vocab in `vocab.py`.

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

This is the first MarinFold document-structure impl. It's the
reference for any v2/v3 successors and the contract for downstream
data-gen + eval experiments.

## Approach

Layout:

```
experiments/exp1_document_structures_contacts_and_distances_v1/
├── vocab.py         # canonical 2838-token vocab + NAME + CONTEXT_LENGTH
├── parse.py         # gemmi-backed PDB / mmCIF reader; ParsedStructure dataclass
├── generate.py      # generate_documents() + GenerationConfig
├── inference.py     # predict() + evaluate() + InferenceConfig
├── cli.py           # argparse driver: generate / infer / evaluate / tokenizer
└── pyproject.toml
```

`vocab.py` is the canonical token list, ported from
`exp0_models_protein_docs_initial_port/create_protein_tokenizer.py`.
`parse.py` reads PDB / mmCIF (+ `.gz`) via gemmi.

`generate.py:generate_documents(...)` ports the algorithm from
`timodonnell/contactdoc/contactdoc/generators/contacts_and_distances_v1.py`:
deterministic seed per `entry_id`, CB-CB contact eligibility
(CA for GLY), per-mode fraction sampling, distance statements over
uniform residue+atom pairs, rank-ordered statements, 50/50 pLDDT
placement. One doc per input. The `-5x` HF subset suffix is a
*data-pipeline* concern ("up to 5 AFDB entries per Foldseek
structural cluster"), not augmentation here.

`inference.py` exposes two plain functions sharing the per-pair
query loop: walk pairs (i, j) with i<j on each input structure,
prompt with the v1 base + sequence + `<begin_statements>` + N
seeded GT long-range contacts + `<distance> <p_i> <p_j> <atom_i>
<atom_j>`, renormalize the next-token distribution over the 64
`<d_X.X>` bins to an expected distance.

- `predict(cfg)` yields per-(structure, n_seeded) records with
  `expected_distances` (and optionally the full `bin_probs` via
  `--keep-bin-probs`). No GT consulted.
- `evaluate(cfg)` runs the same loop, then scores against the GT
  distance matrix at the same atom pair (the input file IS the
  ground truth — its coordinates are the comparison target).
  Returns `mae_at_n<N>_angstrom` per seeded-contact count. Pairs
  with non-finite GT or GT > `distance_cap_angstrom` are filtered
  *before* the LLM forward pass.

`--seed-n-values 0,5,20,50` sweeps over seeded counts in a single
run — reproduces the Phase 7c trace from the
LlamaFold-experiments notebook ("a few GT contacts as hints").
`--query-atom` (default `CA`) picks the atom on both i and j.

Run from this directory with the venv active.

### Backend-independent commands

`generate` and `tokenizer` don't load a trained model, so they don't
need a backend extra installed:

```bash
uv sync   # base install only

uv run python cli.py generate \
    --input /path/to/afdb-cifs/ --num-docs 100 \
    --out /tmp/sample-docs.parquet

uv run python cli.py tokenizer --save-local /tmp/tok/
uv run python cli.py tokenizer --push open-athena/contacts-and-distances-v1-tokenizer
```

### Generating documents with Zephyr on Iris

First, follow the Marin's internal guidelines to get set up on our internal GCP project: https://marin.readthedocs.io/en/latest/dev-guide/guidelines-internal/

Next, in a dedicated terminal window, please connect to the Iris cluster:
```bash
uv run iris --cluster=marin cluster dashboard
```

Then, in another dedicated terminal window, run the document generation job:

```bash
uv run iris --cluster=marin-dev job run -- python cli.py generate --input "gs://public-datasets-deepmind-alphafold-v4/" --num-docs 100 --out "gs://marin-tmp-us-central1/marin-fold-tests/*.parquet"
```

### Running inference and evaluation

`--backend` selects the runtime; `--model` accepts a nickname listed
in repo-root [`MODELS.yaml`](../../MODELS.yaml) (the matching HF
subfolder is downloaded on first use) or a local directory holding
the model + tokenizer. Bare HF repo ids are not accepted —
register the model in `MODELS.yaml` first.

**On a GPU machine (Linux + NVIDIA):** vLLM. This is the production
path; fastest for sweeps over many structures and seed counts.

```bash
uv sync --extra eval
uv run python cli.py evaluate \
    --backend vllm --model 1B \
    --input /path/to/pdbs/ --seed-n-values 0,5,20,50 \
    --out /tmp/metrics.json
```

**On an Apple Silicon Mac (M1–M4):** MLX is the fastest local path.

```bash
uv sync --extra eval-mlx
uv run python cli.py evaluate \
    --backend mlx --model 1B \
    --input /path/to/pdbs/ --seed-n-values 0,5,20,50 \
    --out /tmp/metrics.json
```

**Generic torch (Apple Silicon MPS, CPU, or CUDA):** lowest-effort
fallback. Slower than MLX on the same Mac and slower than vLLM on
the same GPU, but works anywhere torch installs.

```bash
uv sync --extra eval-local
uv run python cli.py evaluate \
    --backend transformers --model 1B --dtype bfloat16 \
    --input /path/to/pdbs/ --seed-n-values 0,5,20,50 \
    --out /tmp/metrics.json
```

On MPS, prefer `--dtype bfloat16` (default) or `--dtype float32`. Do
not use `float16` on MPS for these bf16-trained checkpoints — the
narrower dynamic range overflows the residual stream and produces
NaN logits.

`infer` (no ground truth) takes the same `--backend` / `--model`
flags:

```bash
uv run python cli.py infer \
    --backend mlx --model 1B \
    --input /path/to/seqs/ --out /tmp/predictions.parquet
```

## Success criteria

1. `cli.py generate / infer / evaluate / tokenizer` all parse and
   dispatch correctly (covered by `tests/test_structure.py`).
2. The tokenizer built via `build_tokenizer(all_domain_tokens())`
   is byte-identical to the published
   `timodonnell/protein-docs-tokenizer@83f597d88e9b` (network-marked
   test).
3. Tokenizing a generated doc for 1QYS (top7) matches the token-ID
   sequence that `protein_distogram_eval.py` builds for the same
   structure today.
4. `inference.evaluate(cfg)` on
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
