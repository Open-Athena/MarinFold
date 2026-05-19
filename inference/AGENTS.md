# inference/AGENTS.md

Rules for agents working under `inference/`. Layered on top of the
root `AGENTS.md`.

## Scope

`inference/` is a small shared library for **running a trained
model**. Three responsibilities, nothing more:

1. The `Backend` protocol (`next_token_probs`, `tokenizer`).
2. Backend implementations against vLLM, HuggingFace transformers,
   and MLX.
3. Model resolution: local path or `MODELS.yaml` nickname → local
   directory on disk.

Format-specific logic — prompt construction, vocab token resolution,
ground-truth comparison, MAE/accuracy math — belongs in the document-
structure impl under `experiments/exp<N>_document_structures_<name>/`.
The library is intentionally protein-unaware. If a helper here starts
to know about residues or distances, it's in the wrong directory.

## Hard rules

1. **Keep the base install lightweight.** `marinfold_inference`
   (without extras) must work with only `huggingface_hub + numpy +
   pyyaml + tokenizers + transformers`. Each backend's heavy runtime
   (`vllm`, `torch`, `mlx`) lives behind its own extra and is lazy-
   imported inside its module. Importing `marinfold_inference` from
   a process with no backend installed must not raise.

2. **Backends are interchangeable at the `Backend` protocol level.**
   Same inputs → same outputs (up to numerical precision). New
   backends must conform; don't widen the protocol with backend-
   specific knobs. Backend-specific construction options can live in
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
