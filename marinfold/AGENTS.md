# marinfold/AGENTS.md

Rules for agents working under `marinfold/`. Layered on top of the
root `AGENTS.md`.

## Scope

`marinfold/` is the top-level user-facing package. Four
responsibilities, nothing more:

1. **Inference backends** (`marinfold.inference`) — the `Backend`
   protocol (`next_token_probs`, `tokenizer`) and the three
   implementations (vLLM, transformers, MLX). Protein-unaware.
2. **Document-structure shared toolkit** (`marinfold.document_structures`
   `core` + `writers`) — `EvalResult`, `build_tokenizer`, output
   writers (`write_docs` / `write_predictions` / `write_eval`).
   Protein-unaware.
3. **Graduated document-structure impls**
   (`marinfold.document_structures.<name>`) — each impl is a
   subpackage. The impl is the only place that knows about residues,
   distances, and one specific protein-document format.
4. **Top-level CLI** (`marinfold.cli`) + model registry
   (`marinfold.registry`) — `marinfold infer` / `marinfold evaluate`
   that look up a model in `MODELS.yaml`, pick a supported document
   structure, and dispatch to the impl subpackage.

Format-specific logic stays inside an impl subpackage. The
sibling modules under `marinfold.inference` and `marinfold.document_structures.{core,writers}`
remain protein-unaware. If a helper in those starts to know about
residues or distances, it's in the wrong place.

## Hard rules

1. **Keep the base install lightweight.** `marinfold` (without
   extras) must work with only `huggingface_hub + numpy + pyarrow +
   pyyaml + tokenizers + transformers`. Each backend's heavy runtime
   (`vllm`, `torch`, `mlx`) lives behind its own extra and is lazy-
   imported inside its module. Importing `marinfold` from a process
   with no backend installed must not raise.

2. **Backends are interchangeable at the `Backend` protocol level.**
   Same inputs → same outputs (up to numerical precision). New
   backends must conform; don't widen the protocol with backend-
   specific knobs. Backend-specific construction options live as
   the concrete backend's `__init__` kwargs.

3. **Model resolution is strict.** `resolve_model` accepts a local
   directory or a `MODELS.yaml` nickname — nothing else. Don't
   silently fall through to "treat as HF repo id"; that turns a
   typo into a download. Add new models to `MODELS.yaml` instead.

4. **No protein-specific code.** No imports of gemmi / biopython, no
   constants like `_CONTACT_CUTOFF_A`, no vocabulary references to
   `<d_X.X>` or `<p_N>`. The caller passes opaque token ids.

5. **`next_token_probs` returns un-renormalized probabilities** over
   the target tokens. Backends with full-vocab logits return real
   softmax mass; vLLM (top-k logprobs) returns 0 for targets that
   fall outside the top-k. The caller decides what to do with that.

6. **The CLI is the high-level entry point, not a one-stop shop.**
   `marinfold infer` and `marinfold evaluate` cover the common
   "run a trained model" cases. Lower-level operations (e.g.
   `generate`, `tokenizer`) stay on the per-impl `cli.py` inside
   each graduated impl. Don't grow the top-level CLI into a
   centralized dispatcher for impl-specific tooling.
