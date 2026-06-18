# marin/levanter cache-reader bug: `Sharded cache ledger missing input_ids/0 count`

**Package:** `marin-levanter` (`lib/levanter/src/levanter/store/cache.py`)
**Versions:** reader *intolerance* confirmed locally on `0.2.19.dev202606171019` (PyPI, latest); the end-to-end training failure was observed on the iris TPU worker on runs whose captured `requirements.txt` reported both `marin-levanter==0.2.0` (git `e78c54a8`) and `==0.2.19.dev202606171019`. See **Reproduction status** below for the precise confirmed-vs-observed split.

## Symptom
Training reading an existing levanter cache dies with:
```
ValueError: Sharded cache ledger missing input_ids/0 count for shard part-00000-of-00133
```
The on-disk ledger is correct — its `field_counts` key is **`input_ids`**, not `input_ids/0`.

## Root cause
The ledger field key is derived from the **exemplar's** pytree paths:
`field = "/".join(_render_path_elem(p) for p in path)` over `jtu.tree_flatten_with_path(exemplar)`.

- An **array** exemplar `{"input_ids": np.ndarray}` → path `['input_ids']` → field **`input_ids`** (what the writer recorded).
- A **python-list** exemplar `{"input_ids": [0]}` (what `BatchTokenizer.output_exemplar` returns) → path `['input_ids'][0]` → field **`input_ids/0`**.

When the reader derives `input_ids/0` but the ledger recorded `input_ids`, the count lookups do a **direct `.get(field)` with no fallback** and raise:
- `_ensure_shard_field_offsets` (`cache.py:185-190`)
- `_build_flat_field_offsets_async` (`cache.py:243-244`)

(#6014 made the *consolidation/writer* exemplar consistent. Whenever the reader
still derives `input_ids/0` — e.g. for a cache written before #6014, or any path
where a python-list `output_exemplar` reaches the path-derivation un-normalized —
the lookup has **no tolerance** for the `input_ids` vs `input_ids/0` mismatch and
raises. Note `0.2.19.dev` normalizes the list in the common `TreeCache.load` path,
so it does not always derive `input_ids/0`; see **Reproduction status**.)

## Minimal repro — reader intolerance (verified on `0.2.19.dev`)
This shows the lookup has no fallback: the ledger records `input_ids`, and asking
for the derived path `input_ids/0` raises. (Synthetic trigger — it calls the
lookup directly to isolate the intolerance, independent of how the field is derived.)
```python
import numpy as np
from levanter.store.cache import TreeCache
c = TreeCache.load(CACHE_DIR, {"input_ids": np.zeros((1,), np.int32)}, None)
print(list(c.ledger.field_counts))                       # ['input_ids']
c._ensure_shard_field_offsets("input_ids")               # OK
c._ensure_shard_field_offsets("input_ids/0")             # ValueError: ... missing input_ids/0 count
```

## Reproduction status (confirmed vs observed)
- **Confirmed locally (`0.2.19.dev`):** the per-shard count lookups raise on a
  field key absent from the ledger, with no fallback (snippet above).
- **Confirmed locally:** the field key is exemplar-derived — an **array** leaf →
  `input_ids`, a **python list** leaf → `input_ids/0` (via `jtu.tree_flatten_with_path`).
- **Observed on the iris TPU worker:** the end-to-end training read of this cache
  died with `missing input_ids/0 count`, on runs reporting `marin-levanter==0.2.0`
  **and** `==0.2.19.dev`.
- **NOT reproduced locally on `0.2.19.dev`:** the *end-to-end* path
  `TreeCache.load(CACHE_DIR, processor.output_exemplar)` reads the cache fine —
  on this build the list `output_exemplar` is normalized so the derived field is
  `input_ids`, never `input_ids/0`. So on `0.2.19.dev` the failing path is reached
  on the worker but not in my local read; **the exact reason the worker derives
  `input_ids/0` while local `0.2.19.dev` does not is unresolved** (candidates: the
  pre-#6014 cache has no stored exemplar, an effective build/exemplar difference on
  the pod, or a tree-flattening/transitive-dep difference). The proposed fix below
  makes the reader correct regardless of which side derives the mismatch.

## Proposed fix (verified to resolve all shards of the affected cache)
Tolerate the naming mismatch by falling back to ancestor paths when the exact field key is absent:

```python
def _resolve_ledger_field(shard_counts: Mapping[str, int], field: str) -> str | None:
    """Reader-derived field paths (e.g. 'input_ids/0' from a python-list exemplar)
    may not match the key the writer recorded (e.g. 'input_ids' for an array leaf).
    Fall back to ancestor paths so existing caches stay readable."""
    if field in shard_counts:
        return field
    parts = field.split("/")
    while len(parts) > 1:
        parts = parts[:-1]
        cand = "/".join(parts)
        if cand in shard_counts:
            return cand
    return None
```
Use it in both `_ensure_shard_field_offsets` and `_build_flat_field_offsets_async` instead of the bare `.get(field)` / `field not in ...` checks.

**Stronger root fix (optional, additive):** normalize sequence leaves in `output_exemplar` to numpy arrays (true leaves) so the derived field path is always `input_ids`, never `input_ids/0`.

## Secondary packaging issues (separate from the code bug)
1. The GitHub `…/releases/expanded_assets/marin-*-latest` find-links are **frozen at `0.99.dev20260529`** (umbrella name `marin`, 404 on PyPI), while PyPI carries current `marin-core`/`marin-levanter`/… `0.2.x.dev<date>`.
2. **Version-ordering trap:** the frozen `0.99.dev` sorts *above* the fixed `0.2.x.dev`, so `prerelease=allow` consumers prefer the broken build unless they add an upper bound (`<0.3`).
3. On the iris **TPU worker**, an external experiment that pins `marin-*` does not reliably land on its pinned version (observed both `0.2.0` and `0.2.19.dev` across otherwise-identical launches) — worth checking how the worker `uv sync --all-packages` resolves marin vs. anything pre-present in the pod.
