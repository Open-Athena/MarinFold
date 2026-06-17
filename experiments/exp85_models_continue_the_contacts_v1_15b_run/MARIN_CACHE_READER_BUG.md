# marin/levanter cache-reader bug: `Sharded cache ledger missing input_ids/0 count`

**Package:** `marin-levanter` (`lib/levanter/src/levanter/store/cache.py`)
**Repro'd against:** `marin-levanter==0.2.0` (git `e78c54a8`) **and** `0.2.19.dev202606171019` (PyPI).

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

(#6014 made the *consolidation/writer* exemplar consistent, but caches written before it — and any reader handed a list `output_exemplar` — still hit this. The reader has no tolerance for the `input_ids` vs `input_ids/0` naming.)

## Minimal repro (verified)
```python
import numpy as np
from levanter.store.cache import TreeCache
c = TreeCache.load(CACHE_DIR, {"input_ids": np.zeros((1,), np.int32)}, None)
print(list(c.ledger.field_counts))                       # ['input_ids']
c._ensure_shard_field_offsets("input_ids")               # OK
c._ensure_shard_field_offsets("input_ids/0")             # ValueError: ... missing input_ids/0 count
```

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
