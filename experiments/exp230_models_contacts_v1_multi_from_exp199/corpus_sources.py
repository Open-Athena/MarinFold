# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""The three corpora exp230 draws its protein pool from, behind one iterator.

| arm | prefix | docs | what it is |
|---|---|---|---|
| ``afdb`` | ``contacts_v1/train`` | 4,129,682 | AlphaFold-DB predictions (#53) |
| ``esm_atlas`` | ``contacts_v1_esm_atlas/train`` | 66,759,922 | ESMFold2 distillation (#139) |
| ``pdb`` | ``contacts_v1_pdb_deduped_monomers`` | 41,661 | **experimental** structures (#222) |

The arm names are #213's, so a row here joins straight onto #225's drop list.

**One pool, both halves.**  The multi-draft documents and the plain rehearsal
documents are built from the *same* proteins.  That is not a convenience: if the
two halves came from different protein distributions the model could infer its
mode from protein statistics instead of from the token-0 marker, and the marker
is the thing this run has to make into a clean switch.  #163 did the same.

**Quality gates differ by arm and that is deliberate.**  AFDB and ESM-Atlas gate
on ``global_plddt`` — genuinely a confidence in both.  The PDB corpus gates on
``resolution``, because #222 **zeroed** its ``global_plddt``: the library fills
that column with mean CA B-factor, which for a crystal runs the *opposite*
direction from pLDDT, so a shared ``global_plddt >= 80`` filter across a mixture
would keep the good AFDB documents and the bad PDB ones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from _contacts_v1_doc import parse_doc

BUCKET = "open-athena/MarinFold"


@dataclass(frozen=True)
class CorpusSpec:
    #: #213's arm name; the join key against the drop list.
    arm: str
    prefix: str
    columns: list[str] = field(default_factory=list)
    #: pLDDT floor; ``None`` for corpora where the column carries no claim.
    min_plddt: float | None = None
    #: resolution ceiling in angstrom; ``None`` where there is no resolution.
    max_resolution: float | None = None
    #: First shard ordinal eligible for the draw, and why.  Only AFDB uses it.
    first_shard: int = 0
    first_shard_note: str = ""


AFDB = CorpusSpec(
    arm="afdb",
    prefix="data/document_structures/contacts_v1/train",
    columns=["document", "entry_id", "seq_len", "global_plddt", "truncated",
             "contacts_emitted", "round", "struct_cluster_id"],
    min_plddt=80.0,
    # exp53's shards are physically ordered round-descending (round 4 first,
    # round 0 last) and each shard holds a single round.  Round 0 is one
    # document per structural cluster at the highest pLDDT — maximum diversity,
    # no near-duplicates — which is why #98 and #200 both drew from it.
    # 1596 is exp98's binary search (`first_round0_shard`) re-run against the
    # published bucket copy: shards 1596..2066 are round 0, 471 of 2067.
    first_shard=1596,
    first_shard_note="round-0 shards only (exp98 first_round0_shard, re-derived 2026-08-13)",
)

ESM_ATLAS = CorpusSpec(
    arm="esm_atlas",
    prefix="data/document_structures/contacts_v1_esm_atlas/train",
    columns=["document", "entry_id", "seq_len", "global_plddt", "truncated",
             "num_contacts", "ptm"],
    min_plddt=80.0,
)

PDB_MONOMERS = CorpusSpec(
    arm="pdb",
    prefix="data/document_structures/contacts_v1_pdb_deduped_monomers/documents",
    columns=["document", "entry_id", "seq_len", "truncated", "contacts_emitted",
             "resolution", "method", "pdb_id", "cluster_ids"],
    # 9 A is #222's own admission bar (it follows Protenix/AF3); NMR entries
    # report no resolution at all and are kept, exactly as #222 keeps them.
    max_resolution=9.0,
)

ARMS = {spec.arm: spec for spec in (AFDB, ESM_ATLAS, PDB_MONOMERS)}


#: Written by ``stage.py``: ``{arm: [bucket path, ...]}``, the corpus-wide sorted
#: shard listing.  Kept on disk because reading it needs ``huggingface_hub>=1.5``
#: (buckets are invisible to older clients) while everything downstream of it
#: needs ``marinfold``, whose transformers pins ``huggingface_hub<1``.  The two
#: cannot share an interpreter, so they share a file instead.
MANIFEST_NAME = "manifest.json"


def manifest_path(work: Path) -> Path:
    return work / "cache" / MANIFEST_NAME


def list_shards(spec: CorpusSpec, work: Path) -> list[str]:
    """Corpus-wide **sorted** shard paths for one arm, from the staged manifest.

    Sorting before any shuffle is the fix for the reproducibility bug #163 hit:
    its selector seeded a shuffle of the shard list, but ``HfFileSystem.glob``
    did not return a stable order, so the same seed produced a different draw
    and a fresh 50k selection had *zero* overlap with the previous one.
    """
    import json

    path = manifest_path(work)
    if not path.exists():
        raise SystemExit(f"no shard manifest at {path} — run `python stage.py --work {work}` first")
    manifest = json.loads(path.read_text())
    if spec.arm not in manifest:
        raise SystemExit(f"arm {spec.arm!r} not in {path}; staged arms: {sorted(manifest)}")
    return manifest[spec.arm]


def local_path(work: Path, bucket_path: str) -> Path:
    return work / "cache" / bucket_path.replace("/", "__")


def eligible_shards(spec: CorpusSpec, work: Path) -> list[str]:
    """The shards an exp230 draw may sample from — ``first_shard`` onward."""
    return list_shards(spec, work)[spec.first_shard:]


def draw_shards(spec: CorpusSpec, work: Path, n: int, seed: int) -> list[str]:
    """A reproducible sample of ``n`` eligible shards.

    Shuffling a *sorted* list under a fixed seed is the whole point — #163's
    selector shuffled an unstably-ordered glob, so re-running it produced a
    disjoint draw and its 50k target list had to be pinned by hand afterwards.
    """
    import random

    pool = eligible_shards(spec, work)
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def iter_corpus_rows(
    spec: CorpusSpec,
    *,
    work: Path,
    log=print,
    shards: list[str] | None = None,
    max_len: int = 512,
    min_contacts: int = 5,
):
    """Yield one dict per usable protein: entry_id, shard, row, sequence, gt, L.

    ``shard``/``row`` are the corpus coordinates under #213's header grammar, so
    a FASTA written from this stream inverts back to the corpus with no join.
    ``row`` is the index **within the shard**, counted before filtering, so it
    stays a stable address into the published parquet.
    """
    all_paths = list_shards(spec, work)
    paths = shards if shards is not None else all_paths
    # The shard ordinal must be the corpus-wide one, not the position within a
    # subset — otherwise a FASTA header built from a partial draw addresses the
    # wrong row of the published parquet.
    index_of = {p: i for i, p in enumerate(all_paths)}

    for path in paths:
        local = local_path(work, path)
        if not local.exists():
            raise SystemExit(f"{local} not staged — run `python stage.py --work {work}`")
        cols = [c for c in spec.columns]
        table = pq.read_table(local, columns=cols)
        shard_idx = index_of[path]
        n_yield = 0
        for row_idx, row in enumerate(table.to_pylist()):
            if row.get("truncated"):
                continue
            if row["seq_len"] > max_len or row["seq_len"] < 2:
                continue
            n_contacts = row.get("contacts_emitted", row.get("num_contacts"))
            if n_contacts is None or n_contacts < min_contacts:
                continue
            if spec.min_plddt is not None and (row.get("global_plddt") or 0.0) < spec.min_plddt:
                continue
            if spec.max_resolution is not None:
                res = row.get("resolution")
                # NMR entries report no resolution; #222 admits them, so do we.
                if res is not None and res > spec.max_resolution:
                    continue
            parsed = parse_doc(row["document"])
            if parsed is None:
                continue
            L, seq, gt = parsed
            if len(gt) < min_contacts or L != row["seq_len"]:
                continue
            n_yield += 1
            yield {
                "arm": spec.arm,
                "entry_id": str(row["entry_id"]),
                "shard": shard_idx,
                "row": row_idx,
                "L": L,
                "sequence": seq,
                "n_gt": len(gt),
                "gt_contacts": [[int(i), int(j)] for i, j in gt],
                "global_plddt": float(row.get("global_plddt") or 0.0),
                "resolution": float(row["resolution"]) if row.get("resolution") is not None else None,
            }
        log(f"[iter] {spec.arm} shard {shard_idx}: {n_yield}/{table.num_rows} usable")
