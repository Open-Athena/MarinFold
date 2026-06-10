# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Auto-loaded startup hook (Python imports ``sitecustomize`` at interpreter
startup from any directory on ``sys.path``; the iris worker unpacks this
experiment's bundle to ``/app`` and runs from there, so this file runs in every
sub-job — driver, tokenize, and training pod).

It works around a marin-latest (``0.99.dev20260529``) bug where a *fresh*
tokenize→train cannot read its own token cache:

    ValueError: Sharded cache ledger missing input_ids/0 count for shard ...

Root cause: ``levanter.data.text._batch_tokenizer.BatchTokenizer.output_exemplar``
returns ``{"input_ids": <python list>}``. The cache WRITER flattens the exemplar
with ``is_leaf=heuristic_is_leaf`` (a list of ints is a leaf → one flat
``input_ids`` field, which is what lands on disk and in the ledger), but the
cache READER's ``jagged_array_tree`` walks the same exemplar WITHOUT
``is_leaf``, so the list is treated as a pytree node and the field becomes the
leaf-path ``input_ids/0`` — which doesn't exist in the (flat) ledger. The
already-written caches are correct and flat; only the reader's in-memory
exemplar is wrong. Making ``output_exemplar`` return numpy arrays (true leaves)
flattens the reader's view to match the cache — verified locally by reading the
real val cache (41,954 rows) end-to-end.

This is fixed upstream by marin PR #6014 (closes #6008, merged 2026-06-02), which
isn't in any published wheel yet (the marin-latest mirror is frozen at
2026-05-29). Remove this shim once the experiment is on a marin build that
includes #6014. See README "Current blocker".
"""

import os


def _install_exemplar_patch() -> None:
    try:
        import numpy as np

        from levanter.data.text import _batch_tokenizer as _bt
    except Exception:
        # levanter not importable in this process (e.g. a pure-CPU helper) —
        # nothing to patch, stay silent.
        return

    cls = getattr(_bt, "BatchTokenizer", None)
    prop = getattr(cls, "output_exemplar", None) if cls is not None else None
    if cls is None or not isinstance(prop, property) or getattr(cls, "_marinfold_exemplar_patched", False):
        return

    _orig_fget = prop.fget

    def _patched_output_exemplar(self):
        exemplar = _orig_fget(self)
        if isinstance(exemplar, dict):
            return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in exemplar.items()}
        return exemplar

    cls.output_exemplar = property(_patched_output_exemplar)
    cls._marinfold_exemplar_patched = True
    # Loud marker so we can confirm the shim loaded in the iris job logs.
    print("[exp67 sitecustomize] patched BatchTokenizer.output_exemplar (marin #6008/#6014 workaround)", flush=True)


# Allow disabling via env in case it ever needs to be bypassed.
if os.environ.get("EXP67_DISABLE_EXEMPLAR_PATCH") != "1":
    _install_exemplar_patch()
