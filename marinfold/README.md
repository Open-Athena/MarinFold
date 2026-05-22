# marinfold/

The top-level MarinFold Python package: backend abstraction, shared
document-structure toolkit, graduated document-structure impls, and
the `marinfold` CLI.

```
marinfold/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── marinfold/
│   ├── __init__.py                       # public API re-exports
│   ├── cli.py                            # `marinfold infer / evaluate`
│   ├── registry.py                       # MODELS.yaml: nickname → local path, default model
│   ├── MODELS.yaml                       # model registry: nickname -> model path
│   ├── inference/                        # Backend protocol + three backends
│   │   ├── core.py
│   │   ├── _vllm.py
│   │   ├── _transformers.py
│   │   └── _mlx.py
│   └── document_structures/              # shared toolkit + graduated impl subpackages
│       ├── core.py                       # EvalResult, build_tokenizer
│       ├── writers.py                    # write_docs / write_predictions / write_eval
│       └── contacts_and_distances_v1/    # graduated impl (subpackage)
│           ├── cli.py
│           ├── vocab.py / parse.py / generate.py / inference.py
│           └── ...
└── tests/
    ├── test_registry.py / test_cli.py / test_transformers.py
    └── document_structures/contacts_and_distances_v1/test_structure.py
```

## Public API

```python
# Run a trained model:
from marinfold import Backend, load_backend, resolve_model

backend = load_backend("mlx", model="1B")          # MODELS.yaml nickname
probs = backend.next_token_probs(prompts, target_token_ids)

# Build a doc-structure impl:
from marinfold import EvalResult, build_tokenizer
from marinfold import write_docs, write_predictions, write_eval

# Use a graduated impl directly:
from marinfold.document_structures.contacts_and_distances_v1 import (
    predict, evaluate, InferenceConfig,
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

Graduated impls live as subpackages of `marinfold.document_structures`.
The current impl (`contacts-and-distances-v1`) and its parser /
plotting deps (`gemmi`, `matplotlib`) are pulled in by the base
install.

The high-level `marinfold infer` / `marinfold evaluate` CLI picks
the impl automatically based on the model's `document_structures`
list in `MODELS.yaml`. Each impl also ships its own lower-level
console script (e.g. `contacts-and-distances-v1`) for power-user
flag access.

## Model resolution

`--model` (or the `model=` arg to `load_backend`) accepts:

1. A local directory (path that exists on disk). Used as-is.
2. A nickname listed in `marinfold/MODELS.yaml`. The matching HF
   subfolder is downloaded via `huggingface_hub.snapshot_download`.

Bare HF repo ids are not accepted — register the model in
`MODELS.yaml` to use it. This is deliberate: it keeps the set of
known models small, named, and discoverable.

`MODELS.yaml` is located in this order:

1. The path named by `MARINFOLD_MODELS_YAML`.
2. Walking up from `os.getcwd()`.
3. Walking up from this package's location (covers the normal
   installed-package / editable-install case even when cwd is elsewhere).

## See also

- [`../experiments/`](../experiments/) — in-flight experiments,
  including pre-graduation document-structure impls.
- [`marinfold/MODELS.yaml`](./marinfold/MODELS.yaml) — registered trained models.
