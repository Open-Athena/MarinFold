# marinfold/

The top-level MarinFold Python package: backend abstraction, the shared
document-structure toolkit, every document-structure impl, and the
`marinfold` CLI.

```
marinfold/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── marinfold/
│   ├── __init__.py                       # public API re-exports
│   ├── cli.py                            # `marinfold infer / evaluate`
│   ├── registry.py                       # MODELS.yaml → a local model dir
│   ├── MODELS.yaml                       # model registry: nickname → HF URL
│   ├── inference/                        # Backend protocol + three backends
│   │   ├── core.py
│   │   ├── _vllm.py
│   │   ├── _transformers.py
│   │   └── _mlx.py
│   └── document_structures/              # shared toolkit + one subpackage per impl
│       ├── core.py                       # EvalResult, build_tokenizer
│       ├── writers.py                    # write_docs / write_predictions / write_eval
│       ├── io.py                         # read_object_bytes / thread_per_row_in_shard
│       ├── contacts_v1/                  # current format (README.md + SPEC.md)
│       ├── contacts_and_coordinates_v1/  # contacts-v1 + a 3D coordinate section
│       └── contacts_and_distances_v1/    # previous generation: CB-CB distances
└── tests/                                # mirrors the package layout
```

## Public API

```python
# Run a trained model:
from marinfold import Backend, load_backend, resolve_model

backend = load_backend("mlx", model=None)   # None → the MODELS.yaml default
probs = backend.next_token_probs(prompts, target_token_ids)

# Inspect the model registry:
from marinfold import (
    ModelEntry, default_model_nickname, list_model_entries, resolve_model_entry,
)

# Build a doc-structure impl:
from marinfold import EvalResult, build_tokenizer
from marinfold import write_docs, write_predictions, write_eval

# Per-row I/O for data-generation pipelines (not re-exported at the top level):
from marinfold.document_structures import read_object_bytes, thread_per_row_in_shard

# Use an impl directly:
from marinfold.document_structures.contacts_v1 import (
    predict, evaluate, InferenceConfig, generate_document,
)
```

## Backends

| Backend | When to use | Extra |
|---|---|---|
| `vllm` | Linux + NVIDIA GPU, production / scaled eval | `marinfold[vllm]` |
| `transformers` | Generic torch (Apple Silicon MPS, CPU, CUDA). Lowest-effort local eval. | `marinfold[transformers]` |
| `mlx` | Apple Silicon native, fastest local. | `marinfold[mlx]` |

All three load HF safetensors directly — no GGUF / MLX conversion
step is required.

## Document-structure impls

Every document structure is implemented **here**, as a subpackage of
`marinfold.document_structures`, from its first commit. There is no
separate in-flight location and no graduation step: an experiment that
needs a new format adds the subpackage in this package and imports it,
so an experiment dir under `../experiments/` holds only that
experiment's own driver code and results. (Two early dirs —
`exp1_document_structures_contacts_and_distances_v1` and
`exp34_document_structures_contacts_and_distances_v2` — predate this and
still carry their own copies of an impl. They are frozen historical
records, not the working code.)

| Impl | Format | `generate` | `infer` / `evaluate` |
|---|---|---|---|
| `contacts-v1` | Sequence section + the strongest pyconfind side-chain contacts. The current format — the default model speaks it. | yes | yes |
| `contacts-and-coordinates-v1` | contacts-v1 plus a coordinate section revealing 3D atom positions coarse-to-fine. | yes | no — generation-only, no trained model yet |
| `contacts-and-distances-v1` | Previous generation: `<d_X.X>` CB-CB distance bins over residue pairs. | yes | yes |

An impl owns everything format-specific: vocabulary, parsing, document
generation, and — where it has one — the inference readout. `contacts_v1`
and `contacts_and_coordinates_v1` each carry a `README.md` (orientation)
and a `SPEC.md` (the authoritative format definition, plus a maintained
"Implementation notes & discrepancies" section).

The high-level `marinfold infer` / `marinfold evaluate` CLI picks the
impl automatically from the model's `document_structures` list in
`MODELS.yaml`. Each impl also has its own lower-level CLI for power-user
flag access (`generate`, `view`, `tokenizer`, seed-N sweeps, …). The two
older impls expose theirs as console scripts (`contacts-v1`,
`contacts-and-distances-v1`); `contacts-and-coordinates-v1` has no
console script — run it with
`python -m marinfold.document_structures.contacts_and_coordinates_v1.cli`.

Dependencies: `gemmi` and `matplotlib` come with the base install, and
are imported lazily. Generating documents for `contacts-v1` or
`contacts-and-coordinates-v1` additionally needs `pyconfind` — install
`marinfold[contacts-v1]`. (Their `tokenizer` subcommands work without
it.)

## Model resolution

`--model` (or the `model=` arg to `load_backend`) accepts:

1. A local directory (path that exists on disk). Used as-is.
2. A nickname listed in `marinfold/MODELS.yaml`.
3. `None` / omitted — the entry marked `default: true` is used.

Bare HF repo ids are not accepted — register the model in
`MODELS.yaml` to use it. This is deliberate: it keeps the set of
known models small, named, and discoverable.

A nickname's `url` is fetched according to its shape:

- A regular repo (`.../<org>/<repo>/tree/<rev>/<subfolder>`) — the
  matching subfolder is downloaded via `huggingface_hub.snapshot_download`.
- A storage bucket (`.../buckets/<org>/<bucket>/tree/<prefix>`) — buckets
  are flat and have no revisions, so the prefix is mirrored into the HF
  cache via the bucket HTTP API instead.

`MODELS.yaml` is located in this order:

1. The path named by `MARINFOLD_MODELS_YAML`.
2. Walking up from `os.getcwd()`.
3. Walking up from this package's location (covers the normal
   installed-package / editable-install case even when cwd is elsewhere).

## See also

- [`marinfold/MODELS.yaml`](./marinfold/MODELS.yaml) — registered trained models.
- [`contacts_v1/`](./marinfold/document_structures/contacts_v1/) — the current
  document structure: `README.md` + `SPEC.md`.
- [`../experiments/`](../experiments/) — in-flight experiments, which
  import the impls from this package.
