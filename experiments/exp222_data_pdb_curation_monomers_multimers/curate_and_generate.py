# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 1+2 -- curate each PDB entry and emit its contacts-v1 documents.

One pass over the mirror producing both corpora, because both need the same
expensive thing: the parsed coordinates.

* **monomers** -- every protein chain of the asymmetric unit that survives
  curation, pulled out and analyzed *on its own* (``assembly=None``). One
  document per chain, entry id ``<pdb>_<chain>``.
* **multimers** -- biological assembly 1 whenever it holds two or more
  surviving protein chains and they fit the 2000-index ring. One document per
  entry, entry id ``<pdb>_assembly1``.

The two are disjoint by construction (a monomer document is one isolated
chain; a multimer document is a whole assembly), so a training mixture that
wants both just reads both prefixes.

Every entry also emits a ledger row saying what happened to it and why, so
the funnel from 195,858 entries down to the corpora is fully attributable --
no silent drops.

Usage::

    uv run python curate_and_generate.py \
        --entries /data/exp222_pdb_curation/metadata/entries.parquet \
        --out /data/exp222_pdb_curation
"""

import argparse
import hashlib
import os
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from curate import (
    assembly_subchain_entities,
    build_assembly,
    clean_structure,
    curate_chains,
    load_clusters,
    protein_subchains,
    read_entry,
    single_chain_structure,
)
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    GenerationResult,
    generate_document,
)
from marinfold.document_structures.contacts_v1.vocab import NUM_POSITION_INDICES


# The Protenix / AF3 cutoff this experiment is defined by.
DEFAULT_RELEASE_CUTOFF = "2021-09-30"
DEFAULT_MAX_RESOLUTION = 9.0

# contacts-v1 generation settings. Everything except max_chains is the format
# default, so a PDB monomer document is directly comparable to an AFDB one.
MONOMER_CONFIG = GenerationConfig()
# One <n-term>/<c-term> pair costs 4 tokens and each chain needs a gap index,
# so the ring, not this number, is the real cap; 60 is a generous ceiling that
# keeps a pathological entry from spending minutes in pyconfind.
MAX_MULTIMER_CHAINS = 60
MULTIMER_CONFIG = GenerationConfig(max_chains=MAX_MULTIMER_CHAINS)

# Skip assembly expansion outright above this many protein chains in the
# header. Expansion is where the cost lives: a viral capsid is a handful of
# chains in the asymmetric unit and a symmetry operator that repeats them
# thousands of times (5j7v: 3 -> 8,280 chains), and gemmi will faithfully
# materialise all of it -- one worker reached 110 GB RSS before the built
# assembly could be rejected for having too many chains.
#
# The threshold is 2x MAX_MULTIMER_CHAINS rather than MAX_MULTIMER_CHAINS
# because the header counts protein *asyms* while curation counts gemmi
# *chains*, and one author chain can carry more than one asym. The factor of
# two makes the guard conservative: it only ever skips assemblies that could
# not possibly survive the exact post-curation check.
MAX_ASSEMBLY_CHAINS_BEFORE_EXPANSION = 2 * MAX_MULTIMER_CHAINS

# Worker-global cluster table, populated once per process (see _init_worker).
_CLUSTERS: dict[str, int] = {}


@dataclass(frozen=True)
class Task:
    pdb_id: str
    path: str
    release_date: str
    resolution: float | None
    method: str
    # Rough number of protein residues in the asymmetric unit, used only to
    # schedule the biggest entries first (see main()). pyconfind cost grows
    # steeply with size, and a 30-chain photosystem arriving last would sit
    # alone on one core for minutes after everything else is done.
    est_residues: int = 0
    # Protein chains biological assembly 1 would have, read off the header
    # (see MAX_ASSEMBLY_CHAINS_BEFORE_EXPANSION). Lets the multimer pass
    # decline hopeless entries without materialising them.
    assembly1_protein_chains: int = 0


def _document_row(
    result: GenerationResult,
    *,
    pdb_id: str,
    subset: str,
    release_date: str,
    resolution: float | None,
    method: str,
    entity_ids: list[str],
    cluster_ids: list[int],
    asu_chain_ids: list[str],
) -> dict[str, Any]:
    """A generated document plus everything a training mixture needs to weight it."""
    row = result.metadata_row()
    sequence = "".join(r.resname for r in result.residues)
    row.update({
        "pdb_id": pdb_id,
        "subset": subset,
        "release_date": release_date,
        "resolution": resolution,
        "method": method,
        "entity_ids": entity_ids,
        "cluster_ids": cluster_ids,
        "asu_chain_ids": asu_chain_ids,
        # 100%-identity key over the *resolved* sequence actually serialized
        # (which is what a duplicate document would share, not the full
        # deposited entity sequence).
        "resolved_seq_sha1": hashlib.sha1(sequence.encode()).hexdigest(),
    })
    return row


def _cluster_id(pdb_id: str, entity_id: str) -> int:
    return _CLUSTERS.get(f"{pdb_id.upper()}_{entity_id}", -1)


def _done(
    task: Task,
    monomers: list[dict[str, Any]],
    multimers: list[dict[str, Any]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Package a worker's result, carrying the entry's cost back to the driver.

    Results come back out of order, so the driver cannot tell which task a
    result belongs to; it needs ``est_residues`` echoed to track how much of
    the *work* (rather than how many entries) is finished.
    """
    return {
        "monomers": monomers,
        "multimers": multimers,
        "ledger": ledger,
        "est_residues": task.est_residues,
    }


def process_entry(task: Task) -> dict[str, Any]:
    """Curate one entry and build every document it yields.

    Returns the documents plus a ledger row. Exceptions are caught and
    recorded in the ledger's ``error`` column: one unparseable entry out of
    ~196k must not take the run down, but it must not vanish either.
    """
    ledger: dict[str, Any] = {
        "pdb_id": task.pdb_id,
        "asu_chains_kept": 0,
        "asu_chains_dropped": 0,
        "asu_drop_reasons": [],
        "monomer_docs": 0,
        "monomer_failures": [],
        "assembly_chains_kept": 0,
        "assembly_chains_dropped": 0,
        "assembly_drop_reasons": [],
        "multimer_status": "not_attempted",
        "error": None,
    }
    monomer_rows: list[dict[str, Any]] = []
    multimer_rows: list[dict[str, Any]] = []

    try:
        # The raw entry is kept because assembly expansion needs it: the
        # assembly generator references ligand and water asym ids that
        # cleaning removes.
        raw = read_entry(task.path)
        structure = clean_structure(raw.clone())
    except Exception as exc:  # noqa: BLE001 - recorded in the ledger
        ledger["error"] = f"read: {type(exc).__name__}: {exc}"
        return _done(task, [], [], ledger)

    asu_protein_subchains = protein_subchains(structure)

    # --- monomer pass: each ASU protein chain, analyzed in isolation -------
    try:
        asu = curate_chains(
            structure, asu_protein_subchains, max_residues=NUM_POSITION_INDICES
        )
        ledger["asu_chains_kept"] = len(asu.kept)
        ledger["asu_chains_dropped"] = len(asu.dropped)
        ledger["asu_drop_reasons"] = sorted(asu.dropped.values())
        for chain in asu.kept:
            entry_id = f"{task.pdb_id}_{chain.chain_id}"
            try:
                isolated = single_chain_structure(structure, chain.chain_id)
                with warnings.catch_warnings():
                    # generate_document warns on every unserializable chain;
                    # the ledger records those, so the warning is noise here.
                    warnings.simplefilter("ignore")
                    result = generate_document(
                        isolated, entry_id=entry_id, config=MONOMER_CONFIG
                    )
            except Exception as exc:  # noqa: BLE001 - recorded in the ledger
                ledger["monomer_failures"].append(
                    f"{chain.chain_id}: {type(exc).__name__}: {exc}"
                )
                continue
            if result is None:
                ledger["monomer_failures"].append(f"{chain.chain_id}: unserializable")
                continue
            monomer_rows.append(_document_row(
                result,
                pdb_id=task.pdb_id,
                subset="monomer",
                release_date=task.release_date,
                resolution=task.resolution,
                method=task.method,
                entity_ids=[chain.entity_id],
                cluster_ids=[_cluster_id(task.pdb_id, chain.entity_id)],
                asu_chain_ids=[chain.asu_chain_id],
            ))
        ledger["monomer_docs"] = len(monomer_rows)
    except Exception as exc:  # noqa: BLE001 - recorded in the ledger
        ledger["error"] = f"monomer: {type(exc).__name__}: {exc}"

    # --- multimer pass: biological assembly 1, kept whole ------------------
    # Decide from the header whether expansion is worth attempting at all;
    # see MAX_ASSEMBLY_CHAINS_BEFORE_EXPANSION. The cheap cases (no assembly,
    # a single chain) are skipped too -- there is no complex to describe and
    # the expansion plus its neighbour search would be pure waste on well
    # over half the PDB.
    header_chains = task.assembly1_protein_chains
    if header_chains == 0:
        ledger["multimer_status"] = "no_assembly_1_protein_chains"
        return _done(task, monomer_rows, [], ledger)
    if header_chains == 1:
        ledger["multimer_status"] = "not_a_complex"
        return _done(task, monomer_rows, [], ledger)
    if header_chains > MAX_ASSEMBLY_CHAINS_BEFORE_EXPANSION:
        ledger["multimer_status"] = "too_many_chains_by_header"
        return _done(task, monomer_rows, [], ledger)

    try:
        expanded = build_assembly(raw, "1")
        assembly = clean_structure(expanded) if expanded is not None else None
        if assembly is None:
            ledger["multimer_status"] = "no_assembly_1"
        else:
            built = curate_chains(
                assembly,
                assembly_subchain_entities(asu_protein_subchains, assembly),
                max_residues=NUM_POSITION_INDICES,
            )
            ledger["assembly_chains_kept"] = len(built.kept)
            ledger["assembly_chains_dropped"] = len(built.dropped)
            ledger["assembly_drop_reasons"] = sorted(built.dropped.values())
            n_chains = len(built.kept)
            total_residues = sum(c.n_residues for c in built.kept)
            if n_chains < 2:
                ledger["multimer_status"] = "not_a_complex"
            elif n_chains > MAX_MULTIMER_CHAINS:
                ledger["multimer_status"] = "too_many_chains"
            elif total_residues + n_chains > NUM_POSITION_INDICES:
                ledger["multimer_status"] = "does_not_fit_ring"
            else:
                row = _build_multimer(task, assembly, built)
                if isinstance(row, str):
                    ledger["multimer_status"] = row
                else:
                    multimer_rows.append(row)
                    ledger["multimer_status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - recorded in the ledger
        ledger["multimer_status"] = "error"
        previous = ledger["error"]
        detail = f"multimer: {type(exc).__name__}: {exc}"
        ledger["error"] = detail if previous is None else f"{previous}; {detail}"

    return _done(task, monomer_rows, multimer_rows, ledger)


def _build_multimer(task: Task, assembly, built) -> dict[str, Any] | str:
    """Generate the multimer document, or return a rejection reason string."""
    keep = {c.chain_id for c in built.kept}
    for model in assembly:
        # Names first -- removing a chain shifts the index-backed proxies of
        # every chain after it, so a live iteration skips half of them.
        for name in [chain.name for chain in model]:
            if name not in keep:
                model.remove_chain(name)
    assembly.setup_entities()

    entry_id = f"{task.pdb_id}_assembly1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = generate_document(
            assembly, entry_id=entry_id, config=MULTIMER_CONFIG
        )
    if result is None:
        return "unserializable"
    if result.num_chains < 2:
        # pyconfind saw fewer chains than curation did (e.g. a chain whose
        # residues it does not consider protein). Such a document is a
        # monomer and belongs in the other subset, not here.
        return "collapsed_to_monomer"

    by_chain = {c.chain_id: c for c in built.kept}
    entity_ids = [by_chain[name].entity_id if name in by_chain else "" for name in result.chain_ids]
    return _document_row(
        result,
        pdb_id=task.pdb_id,
        subset="multimer",
        release_date=task.release_date,
        resolution=task.resolution,
        method=task.method,
        entity_ids=entity_ids,
        cluster_ids=[_cluster_id(task.pdb_id, e) for e in entity_ids],
        asu_chain_ids=list(result.chain_ids),
    )


DOC_SCHEMA = pa.schema([
    ("document", pa.string()),
    ("entry_id", pa.string()),
    ("seq_len", pa.int32()),
    ("global_plddt", pa.float64()),
    ("start_index", pa.int32()),
    ("n_term_index", pa.int32()),
    ("c_term_index", pa.int32()),
    ("min_seq_separation", pa.int32()),
    ("contacts_pre_filter", pa.int32()),
    ("contacts_passing_min_degree", pa.int32()),
    ("contacts_emitted", pa.int32()),
    ("contacts_excluded", pa.int32()),
    ("truncated", pa.bool_()),
    ("highest_contact_degree", pa.float64()),
    ("lowest_nonzero_contact_degree", pa.float64()),
    ("lowest_included_contact_degree", pa.float64()),
    ("num_tokens", pa.int32()),
    ("think_tokens", pa.int32()),
    ("num_chains", pa.int32()),
    ("chain_ids", pa.list_(pa.string())),
    ("chain_lengths", pa.list_(pa.int32())),
    ("n_term_indices", pa.list_(pa.int32())),
    ("c_term_indices", pa.list_(pa.int32())),
    ("contacts_pre_filter_inter_chain", pa.int32()),
    ("contacts_emitted_inter_chain", pa.int32()),
    ("sha1", pa.string()),
    ("pdb_id", pa.string()),
    ("subset", pa.string()),
    ("release_date", pa.string()),
    ("resolution", pa.float64()),
    ("method", pa.string()),
    ("entity_ids", pa.list_(pa.string())),
    ("cluster_ids", pa.list_(pa.int32())),
    ("asu_chain_ids", pa.list_(pa.string())),
    ("resolved_seq_sha1", pa.string()),
])

LEDGER_SCHEMA = pa.schema([
    ("pdb_id", pa.string()),
    ("asu_chains_kept", pa.int32()),
    ("asu_chains_dropped", pa.int32()),
    ("asu_drop_reasons", pa.list_(pa.string())),
    ("monomer_docs", pa.int32()),
    ("monomer_failures", pa.list_(pa.string())),
    ("assembly_chains_kept", pa.int32()),
    ("assembly_chains_dropped", pa.int32()),
    ("assembly_drop_reasons", pa.list_(pa.string())),
    ("multimer_status", pa.string()),
    ("error", pa.string()),
])


def _init_worker(cluster_path: str) -> None:
    """Load the cluster table once per worker rather than per entry."""
    global _CLUSTERS
    _CLUSTERS = load_clusters(cluster_path)


class ShardWriter:
    """Buffer rows and flush them as fixed-size parquet shards."""

    def __init__(self, directory: Path, schema: pa.Schema, rows_per_shard: int):
        self.directory = directory
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0
        self.total = 0
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, rows: list[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        self.total += len(rows)
        while len(self.buffer) >= self.rows_per_shard:
            self._flush(self.buffer[: self.rows_per_shard])
            del self.buffer[: self.rows_per_shard]

    def _flush(self, rows: list[dict[str, Any]]) -> None:
        path = self.directory / f"shard-{self.shard_index:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=self.schema), path, compression="zstd"
        )
        self.shard_index += 1

    def close(self) -> None:
        if self.buffer:
            self._flush(self.buffer)
            self.buffer.clear()


def select_entries(
    entries_path: Path,
    excluded_ids: set[str],
    cutoff: str,
    max_resolution: float,
) -> tuple[list[Task], dict[str, int]]:
    """Apply the entry-level filters and return the work list plus a tally."""
    table = pq.read_table(entries_path).to_pylist()
    counts = {
        "total": len(table),
        "scan_error": 0,
        "no_release_date": 0,
        "released_after_cutoff": 0,
        "resolution_too_low": 0,
        "no_protein_entity": 0,
        "eval_set": 0,
        "selected": 0,
    }
    tasks: list[Task] = []
    for row in table:
        if row["error"]:
            counts["scan_error"] += 1
            continue
        release_date = row["release_date"]
        if not release_date:
            counts["no_release_date"] += 1
            continue
        if release_date > cutoff:
            counts["released_after_cutoff"] += 1
            continue
        resolution = row["resolution"]
        # A missing resolution means the method does not report one (NMR);
        # AF3's rule removes structures *worse* than 9 A, which those are not.
        if resolution is not None and resolution >= max_resolution:
            counts["resolution_too_low"] += 1
            continue
        if row["n_protein_entities"] < 1:
            counts["no_protein_entity"] += 1
            continue
        if row["pdb_id"] in excluded_ids:
            counts["eval_set"] += 1
            continue
        counts["selected"] += 1
        lengths = row["protein_entity_lengths"] or []
        entities = max(1, len(lengths))
        chains = max(1, row["n_protein_asym_chains"])
        tasks.append(Task(
            pdb_id=row["pdb_id"],
            path=row["path"],
            release_date=release_date,
            resolution=resolution,
            method=row["method"] or "",
            est_residues=int(sum(lengths) * chains / entities),
            assembly1_protein_chains=int(row["n_assembly1_protein_chains"]),
        ))
    return tasks, counts


def load_excluded_ids(path: Path) -> set[str]:
    import csv

    with path.open() as handle:
        return {row["pdb_id"].strip().lower() for row in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entries", type=Path,
        default=Path("/data/exp222_pdb_curation/metadata/entries.parquet"),
    )
    parser.add_argument(
        "--clusters", type=Path,
        default=Path("/data/exp222_pdb_curation/metadata/clusters-by-entity-40.txt"),
    )
    parser.add_argument(
        "--exclude", type=Path, default=Path("data/eval_set_pdb_ids.csv"),
        help="CSV of PDB ids to hold out (the contact eval set)",
    )
    parser.add_argument("--out", type=Path, default=Path("/data/exp222_pdb_curation"))
    parser.add_argument("--release-cutoff", default=DEFAULT_RELEASE_CUTOFF)
    parser.add_argument("--max-resolution", type=float, default=DEFAULT_MAX_RESOLUTION)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    parser.add_argument("--rows-per-shard", type=int, default=20_000)
    parser.add_argument("--limit", type=int, default=None, help="first N entries (smoke test)")
    args = parser.parse_args(argv)

    excluded = load_excluded_ids(args.exclude)
    tasks, counts = select_entries(
        args.entries, excluded, args.release_cutoff, args.max_resolution
    )
    if args.limit:
        tasks = tasks[: args.limit]
    # Longest-processing-time-first: hand the biggest entries out while every
    # worker is still free, so the run does not end with one core grinding
    # through a ribosome alone.
    tasks.sort(key=lambda t: -t.est_residues)
    print(f"entry-level funnel: {counts}", flush=True)
    print(f"processing {len(tasks)} entries with {args.workers} workers", flush=True)

    monomers = ShardWriter(args.out / "docs" / "monomers", DOC_SCHEMA, args.rows_per_shard)
    multimers = ShardWriter(args.out / "docs" / "multimers", DOC_SCHEMA, args.rows_per_shard)
    ledger = ShardWriter(args.out / "ledger", LEDGER_SCHEMA, 50_000)

    started = time.time()
    done = 0
    # Report ~20 times over the run, so a smoke test of a few hundred entries
    # is as observable as the full 160k-entry pass.
    report_every = max(1, min(2_000, len(tasks) // 20))
    # The ETA is projected from *work* done, not entries done. Because tasks
    # are handed out largest-first, entries/s climbs steadily through the run
    # and an entry-count ETA reads absurdly high early on -- 6,000 of 177,710
    # entries is 3% of the count but 32% of the work.
    total_work = sum(t.est_residues for t in tasks) or 1
    work_done = 0
    try:
        with Pool(args.workers, initializer=_init_worker, initargs=(str(args.clusters),)) as pool:
            for result in pool.imap_unordered(process_entry, tasks, chunksize=1):
                monomers.add(result["monomers"])
                multimers.add(result["multimers"])
                ledger.add([result["ledger"]])
                done += 1
                work_done += result["est_residues"]
                if done % report_every == 0:
                    elapsed = time.time() - started
                    fraction = work_done / total_work
                    eta = elapsed / fraction - elapsed if fraction else float("nan")
                    print(
                        f"  {done}/{len(tasks)}  {done/elapsed:.1f} entries/s  "
                        f"{fraction*100:.1f}% of work  eta {eta/60:.1f} min  "
                        f"mono={monomers.total} multi={multimers.total}",
                        flush=True,
                    )
    finally:
        monomers.close()
        multimers.close()
        ledger.close()

    elapsed = time.time() - started
    print(
        f"done in {elapsed/60:.1f} min: {monomers.total} monomer docs, "
        f"{multimers.total} multimer docs, {ledger.total} ledger rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
