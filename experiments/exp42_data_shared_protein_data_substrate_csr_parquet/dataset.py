# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""On-the-fly training-time document generator over a CSR parquet store.

Reads parquet shards produced by ``cli.py parse-to-csr`` and yields freshly-
generated documents one at a time. Per-epoch reseeding gives the trainer
a different doc per (entry_id, epoch) without storing N variants offline.

**Doc-format-agnostic by design.** The dataloader knows nothing about any
specific document format — every instance takes a ``generator`` callback.
Each doc-format experiment defines its own ``generate_one(structure) -> str``
and wires it in. See :data:`DocumentGenerator` below for the contract.

Designed to compose cleanly into a PyTorch-style dataloader (subclasses
``torch.utils.data.IterableDataset`` when torch is importable, otherwise falls
back to a plain Python iterable so the substrate's tests + dataloader smoke
runs in environments without torch installed).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

import pyarrow.dataset as pa_ds
import pyarrow.fs as pa_fs

import csr_store
from parse import ParsedStructure


# Doc-generation callback signature. Takes a fully-reconstructed
# :class:`parse.ParsedStructure` (with the per-(entry_id, epoch) reseed
# already baked into ``structure.entry_id`` — generators that seed off
# ``entry_id`` get fresh draws each epoch for free) and returns either the
# generated doc string or ``None`` to skip this structure.
#
# Any callable matching this signature plugs into :class:`CSRDocumentDataset`.
# A doc-format experiment typically exposes a factory that captures its
# config and returns a ``DocumentGenerator`` — e.g. exp5's v2 generator:
#
#     # in exp5:
#     def v2_generator(cfg=None, context_length=8192) -> DocumentGenerator:
#         cfg = cfg or GenerationConfig()
#         def _gen(structure):
#             return generate.generate_one(structure, context_length=context_length, cfg=cfg)
#         return _gen
#
#     # at the training-job call site:
#     ds = CSRDocumentDataset(dataset_path="gs://.../csr/", generator=v2_generator())
DocumentGenerator = Callable[[ParsedStructure], Optional[str]]

# Optional torch base class — keeps the POC dependency-free while letting it
# slot into a real torch DataLoader when used in training.
try:
    from torch.utils.data import IterableDataset as _IterableDatasetBase  # type: ignore
    from torch.utils.data import get_worker_info as _torch_get_worker_info  # type: ignore
except ImportError:
    class _IterableDatasetBase:  # type: ignore
        """Placeholder for environments without torch installed."""
        pass

    def _torch_get_worker_info():  # type: ignore
        return None


@dataclass
class CSRDocumentDataset(_IterableDatasetBase):
    """Stream generated documents from a CSR parquet dataset.

    Yields ``{"entry_id": str, "document": str}`` dicts. The dataloader is
    **doc-format-agnostic**: every instance takes a ``generator`` callback
    (matching :data:`DocumentGenerator`) which is where the doc-format
    logic lives. Two extension points are supported:

    * **callback** — pass any ``DocumentGenerator`` to ``generator=`` to
      vary the doc algorithm (different format, different config, scoring
      doc, test fixture, …):

      .. code-block:: python

          from exp5_v2_generator import v2_generator   # or any other format
          ds = CSRDocumentDataset(
              dataset_path="gs://.../csr/",
              generator=v2_generator(),
          )

    * **subclass** — override :meth:`_structure_to_doc` to change the
      *output shape* (e.g. add tokenized fields, yield multiple docs per
      structure, attach extra metadata). The callback covers most needs;
      subclassing is for callers who want to enrich the yielded dict.

    Parameters
    ----------
    dataset_path : str
        One path that names the whole CSR store. Can be:

        * a directory: ``"gs://marin-us-central1/exp42/csr/"``
          (pyarrow auto-discovers all ``*.parquet`` files inside),
        * a single file: ``"/tmp/csr/shard-00000.parquet"``,
        * a glob:       ``"/tmp/csr/shard-*.parquet"``.

        The user doesn't need to glob, list, or sort fragments themselves —
        pyarrow.dataset handles fragment discovery and filesystem details.
    generator : DocumentGenerator
        Required. Callable applied to each reconstructed
        :class:`ParsedStructure` to produce the doc string. The substrate
        ships with no built-in default — the doc format always comes from
        the caller's experiment.
    epoch : int
        Reseeds RNG so the same structure produces a different doc each
        epoch. The dataloader rewrites ``structure.entry_id`` to
        ``"{entry_id}|e{epoch}"`` before calling ``generator`` — anything
        that seeds off ``entry_id`` gets fresh draws each epoch for free.
    read_batch_size : int
        Parquet read batch size — amortizes decode cost over many rows while
        keeping per-row memory bounded. Default 256 is a good
        latency/throughput balance.
    shuffle_within_fragment : bool
        If True, draw rows in a per-fragment random order each epoch (still
        deterministic given ``epoch``). One fragment is materialized at a
        time (~100 MB at our density); the dataset as a whole never is.
    filter : pyarrow.compute.Expression, optional
        Evaluated at the C++ scan layer (e.g.
        ``pc.field("split") == "train"``). Non-matching rows are never
        decoded — much cheaper than a Python-side filter on materialized rows.
    columns : tuple[str, ...], optional
        Column projection. Default reads only what the zero-copy reconstructor
        needs (``csr_store.CSR_READ_COLUMNS``); pass a superset to also
        surface passthrough columns like ``split`` or ``gcs_uri`` (necessary
        if the ``filter`` references one of them).
    """

    dataset_path: str
    generator: DocumentGenerator
    epoch: int = 0
    read_batch_size: int = 256
    shuffle_within_fragment: bool = False
    filter: Optional[Any] = None
    columns: Optional[tuple[str, ...]] = None

    def _assigned_fragments(self, ds: pa_ds.Dataset) -> list:
        """Slice the dataset's fragments to this worker's share.

        Outside a torch DataLoader (``worker_info is None``), this worker
        owns every fragment. Inside one with N workers, worker k gets
        fragments ``[k, k+N, k+2N, …]`` — balances load when fragments are
        roughly equal in size (which they are for AFDB-derived CSR shards).
        Fragment discovery + scheme dispatch happens once inside pyarrow;
        the worker just sub-selects the resulting fragment list.
        """
        all_fragments = list(ds.get_fragments())
        info = _torch_get_worker_info()
        if info is None:
            return all_fragments
        return all_fragments[info.id::info.num_workers]

    def _seed_for(self, entry_id: str) -> int:
        """Per-(entry_id, epoch) seed. Reproduces a given doc given just the
        epoch counter, which is what makes failed-step recovery tractable."""
        h = hashlib.sha1(f"{entry_id}|{self.epoch}".encode()).hexdigest()
        return int(h[:8], 16)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        ds = _open_dataset(self.dataset_path)
        fragments = self._assigned_fragments(ds)
        columns = self.columns if self.columns is not None else csr_store.CSR_READ_COLUMNS
        col_list = list(columns)

        for fragment in fragments:
            if self.shuffle_within_fragment:
                # Read the whole fragment, permute rows, yield. Epoch-keyed
                # RNG keeps it reproducible. Cross-fragment shuffle (epoch-
                # level fragment reordering) is a separate axis we can add
                # when training needs it.
                table = fragment.to_table(columns=col_list, filter=self.filter)
                if table.num_rows == 0:
                    continue
                [batch] = table.combine_chunks().to_batches()
                structures = list(csr_store._iter_batch_rows_zero_copy(batch))
                rng = random.Random(hashlib.sha1(
                    f"{fragment.path}|{self.epoch}".encode()).hexdigest()[:16])
                indices = list(range(len(structures)))
                rng.shuffle(indices)
                for k in indices:
                    yield from self._structure_to_doc(structures[k])
            else:
                # Streaming path: scanner pulls batches as needed. Predicate
                # pushdown + column projection happen at the C++ scan layer.
                scanner = fragment.scanner(
                    columns=col_list, filter=self.filter,
                    batch_size=self.read_batch_size,
                )
                for batch in scanner.to_batches():
                    if batch.num_rows == 0:
                        continue
                    for structure in csr_store._iter_batch_rows_zero_copy(batch):
                        yield from self._structure_to_doc(structure)

    def _structure_to_doc(self, structure: ParsedStructure
                          ) -> Iterator[dict[str, Any]]:
        """Yield one output row per ``structure``. Override this in a
        subclass to enrich the dict (e.g. attach token ids, surface
        passthrough columns) or yield multiple rows per structure.

        The default rewrites ``entry_id`` to ``"{entry_id}|e{epoch}"`` so
        any generator that seeds off ``entry_id`` automatically gets fresh
        draws each epoch — and then yields the original ``entry_id`` back
        out, so downstream consumers can correlate generated docs with
        manifest rows.
        """
        from dataclasses import replace
        seeded = replace(structure, entry_id=f"{structure.entry_id}|e{self.epoch}")
        doc = self.generator(seeded)
        if doc is None:
            return
        yield {"entry_id": structure.entry_id, "document": doc}


def _open_dataset(dataset_path: str) -> pa_ds.Dataset:
    """Construct a ``pyarrow.dataset.Dataset`` from one path / URI string.

    pyarrow's ``dataset()`` natively infers the filesystem from the URI scheme
    (local, ``file://``, ``gs://``, ``s3://``, …) — so the common path is
    one line. We only intervene to expand a shell-style glob, which
    ``pa_ds.dataset`` itself doesn't do.
    """
    if "*" not in dataset_path and "?" not in dataset_path:
        return pa_ds.dataset(dataset_path, format="parquet")

    # Glob: split into (directory, pattern), list the directory through the
    # inferred filesystem, fnmatch the basenames, hand the matching paths
    # back to pa_ds. Works uniformly across local / gs:// / s3:// since
    # ``FileSystem.from_uri`` returns the right backend.
    dir_uri, pattern = os.path.split(dataset_path)
    fs, dir_path = pa_fs.FileSystem.from_uri(dir_uri) if "://" in dir_uri \
        else (pa_fs.LocalFileSystem(), dir_uri)
    entries = fs.get_file_info(pa_fs.FileSelector(dir_path, recursive=False))
    matches = sorted(e.path for e in entries
                     if e.is_file and fnmatch.fnmatch(os.path.basename(e.path), pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {dataset_path!r}")
    return pa_ds.dataset(matches, format="parquet", filesystem=fs)
