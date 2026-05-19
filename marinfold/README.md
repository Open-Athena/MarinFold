# marinfold/

The top-level MarinFold Python package: backend abstraction, shared
document-structure toolkit, and the `marinfold` CLI.

This consolidates what used to be three things — `inference/`,
`document_structures/` (the library half), and a hypothetical
top-level CLI — into one importable package with three submodules:

```
marinfold/
├── pyproject.toml
├── README.md
├── AGENTS.md
└── marinfold/
    ├── __init__.py               # public API re-exports
    ├── cli.py                    # `marinfold infer / evaluate`
    ├── registry.py               # MODELS.yaml: nickname → local path, default model
    ├── inference/                # Backend protocol + three backends
    │   ├── core.py
    │   ├── _vllm.py
    │   ├── _transformers.py
    │   └── _mlx.py
    └── document_structures/      # EvalResult, build_tokenizer, output writers
        ├── core.py
        └── writers.py
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
```

## Backends

| Backend | When to use | Extra |
|---|---|---|
| `vllm` | Linux + NVIDIA GPU, production / scaled eval | `marinfold[vllm]` |
| `transformers` | Generic torch (Apple Silicon MPS, CPU, CUDA). Lowest-effort local eval. | `marinfold[transformers]` |
| `mlx` | Apple Silicon native, fastest local. | `marinfold[mlx]` |

All three load HF safetensors directly — no GGUF / MLX conversion
step is required.

## Model resolution

`--model` (or the `model=` arg to `load_backend`) accepts:

1. A local directory (path that exists on disk). Used as-is.
2. A nickname listed in repo-root `MODELS.yaml`. The matching HF
   subfolder is downloaded via `huggingface_hub.snapshot_download`.

Bare HF repo ids are not accepted — register the model in
`MODELS.yaml` to use it. This is deliberate: it keeps the set of
known models small, named, and discoverable.

`MODELS.yaml` is located in this order:

1. The path named by `MARINFOLD_MODELS_YAML`.
2. Walking up from `os.getcwd()`.
3. Walking up from this package's location (covers the normal
   editable-install / repo-checkout case even when cwd is elsewhere).

## See also

- [`../document_structures/`](../document_structures/) — graduated
  document-structure impls (each its own proper package).
- [`../experiments/`](../experiments/) — in-flight experiments,
  including pre-graduation impls.
