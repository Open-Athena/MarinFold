# inference/

Shared library for running MarinFold-trained models through different
inference backends, behind one small protocol.

Document-structure impls (under
`experiments/exp<N>_document_structures_<name>/`) own the format-
specific logic — prompt construction, vocab resolution, ground-truth
comparison — and call into this library for the actual model forward
pass. The library knows nothing about proteins; it knows about
"give me a probability over these token ids at the next position".

## Backends

| Backend | When to use | Extra |
|---|---|---|
| `vllm` | Linux + NVIDIA GPU, production / scaled eval | `marinfold-inference[vllm]` |
| `transformers` | Generic torch (Apple Silicon MPS, CPU). Lowest-effort local eval. | `marinfold-inference[transformers]` |
| `mlx` | Apple Silicon native, fastest local. | `marinfold-inference[mlx]` |

All three load HF safetensors directly — no GGUF / MLX conversion
step is required.

## Public API

```python
from marinfold_inference import Backend, load_backend

backend = load_backend("mlx", model="1B")          # MODELS.yaml nickname
# or load_backend("transformers", model="/abs/path/to/ckpt")

probs = backend.next_token_probs(
    prompts=[[ids...], [ids...]],   # equal length within a call
    target_token_ids=[t1, t2, ...],
)                                   # (B, len(target_token_ids)), un-renormalized
tok = backend.tokenizer
```

`next_token_probs` returns one row per prompt giving the probability
mass on each target token id at position `len(prompt)` (i.e. the
distribution for the next token). The caller is expected to
renormalize over its target set if it wants a renormalized
distribution.

## Model resolution

`--model` (or the `model=` arg to `load_backend`) accepts:

1. A local directory (path that exists on disk). Used as-is.
2. A nickname listed in repo-root `MODELS.yaml`. The matching HF URL
   is parsed into `(repo_id, revision, subfolder)` and the subfolder
   is downloaded via `huggingface_hub.snapshot_download`.

Bare HF repo ids are not accepted — register the model in
`MODELS.yaml` to use it. This is deliberate: it keeps the set of
known models small, named, and discoverable.

`MODELS.yaml` is located by walking up from `os.getcwd()`. The env
var `MARINFOLD_MODELS_YAML` overrides this.

## Layout

```
inference/
├── pyproject.toml
├── README.md
├── AGENTS.md
└── marinfold_inference/
    ├── __init__.py           # public API: Backend, load_backend, resolve_model
    ├── core.py               # Backend protocol
    ├── registry.py           # MODELS.yaml resolution
    ├── _vllm.py              # VllmBackend (gated by [vllm] extra)
    ├── _transformers.py      # TransformersBackend (gated by [transformers] extra)
    └── _mlx.py               # MlxBackend (gated by [mlx] extra)
```

## See also

- [`../document_structures/`](../document_structures/) — shared
  toolkit for document-structure impls.
- [`../experiments/exp1_document_structures_contacts_and_distances_v1/`](../experiments/exp1_document_structures_contacts_and_distances_v1/)
  — reference impl that consumes this library.
