# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""On-the-fly training-time document generator over a CSR parquet store.

Reads parquet shards produced by ``cli.py parse-to-csr`` and yields freshly-
generated v2 documents one at a time. Per-epoch reseeding gives the trainer
a different doc per (entry_id, epoch) without storing N variants offline —
the whole point of the on-the-fly architecture.

Designed to compose cleanly into a PyTorch-style dataloader (subclasses
``torch.utils.data.IterableDataset`` when torch is importable, otherwise falls
back to a plain Python iterable so the POC runs/tests in environments without
torch installed).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

import pyarrow.dataset as pa_ds
import pyarrow.fs as pa_fs

import csr_store
import generate
from generate import GenerationConfig

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
    """Stream generated v2 documents from a CSR parquet dataset.

    Yields ``{"entry_id": str, "document": str}`` dicts. A real training
    integration would wrap this with a tokenizer (cheap, ~0.3 ms/doc); kept
    out of the POC so the throughput numbers below isolate generate cost.

    Parameters
    ----------
    dataset_path : str
        One path that names the whole CSR store. Can be:

        * a directory: ``"gs://marin-us-central1/exp5/csr/"``
          (pyarrow auto-discovers all ``*.parquet`` files inside),
        * a single file: ``"/tmp/csr/shard-00000.parquet"``,
        * a glob:       ``"/tmp/csr/shard-*.parquet"`` (expanded via the
          relevant fsspec filesystem before being handed to pyarrow).

        The user doesn't need to glob, list, or sort fragments themselves —
        pyarrow.dataset handles fragment discovery and filesystem details.
    epoch : int
        Reseeds RNG so the same structure produces a different doc each epoch.
        Caller bumps this between epochs.
    context_length : int
        Token budget per document.
    cfg : GenerationConfig
        v2 generation hyperparameters; defaults match exp34's reference.
    read_batch_size : int
        Parquet read batch size — amortizes decode cost over many rows while
        keeping per-row memory bounded. Default 256 is a good
        latency/throughput balance.
    shuffle_within_fragment : bool
        If True, draw rows in a per-fragment random order each epoch (still
        deterministic given ``epoch``). The default streaming behavior
        already gives between-fragment variety; this is for stricter
        permutation guarantees.
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
    epoch: int = 0
    context_length: int = 8192
    cfg: GenerationConfig = field(default_factory=GenerationConfig)
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

    def _structure_to_doc(self, structure) -> Iterator[dict[str, Any]]:
        # Override generate's default ``sha1(entry_id)`` seeding by patching
        # the entry_id with an epoch suffix. Keeps the generate.py contract
        # (entry_id → seed) intact while giving per-epoch variation.
        from dataclasses import replace
        seeded = replace(structure, entry_id=f"{structure.entry_id}|e{self.epoch}")
        doc = generate.generate_one(
            seeded, context_length=self.context_length, cfg=self.cfg,
        )
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
