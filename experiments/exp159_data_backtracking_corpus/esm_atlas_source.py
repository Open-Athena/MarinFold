# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Read proteins + ground-truth contacts from exp139's saved ESM-Atlas contacts.

exp139 ran pyconfind once over the whole 66.76M-protein ESMFold2 distillation
set and published the **raw contacts** alongside the contacts-v1 documents, so
a downstream job can rebuild ground truth without re-running pyconfind. That
is exactly what the backtracking corpus needs.

Source (3,338 shards x 20,000 rows, public + anonymously readable)::

    hf://buckets/open-athena/MarinFold/data/contacts/esm_atlas_esmfold2_distill/
        shard-{i:05d}-of-03338.parquet

    # byte-identical GCS mirror staged by exp147 (faster from GCP):
    gs://marin-us-east5/protein-structure/MarinFold/
        exp147_on_the_fly_contacts_v1_pilot/pilot_data/contacts/shard-{i:05d}-of-03338.parquet

Each row carries ``entry_id``, ``seq_len``, the residue arrays
(``residue_resname`` / ``resnum`` / ``chain``, sequence order) and the parallel
contact arrays (``contact_seq_i`` / ``_seq_j`` / ``_degree``, 0-based with
``seq_i < seq_j``). ``marinfold...contacts_v1.analyzed_from_row`` deserializes
a row straight into an ``AnalyzedStructure`` — no pyconfind, and no 3-letter →
1-letter round trip (we hand the residues to the document builder directly).

**The contacts are RAW**: every pair with degree > 0, with *no*
``min_seq_separation`` and *no* ``min_contact_degree`` applied. The contacts-v1
document format applies both downstream, so :func:`ground_truth_pairs` applies
them here — otherwise the "ground truth" would include thousands of
near-zero-degree and trivially-local pairs the base model was never trained to
emit, and every one of them would look like a false positive to the engine.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

# Document-format filters (contacts_v1.generate.GenerationConfig defaults) —
# see the module docstring for why these must be applied to the raw contacts.
MIN_SEQ_SEPARATION = 6
MIN_CONTACT_DEGREE = 0.001

NUM_SHARDS = 3338
HF_BUCKET_PREFIX = (
    "hf://buckets/open-athena/MarinFold/data/contacts/esm_atlas_esmfold2_distill"
)
GCS_MIRROR_PREFIX = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp147_on_the_fly_contacts_v1_pilot/pilot_data/contacts"
)

# Only the columns we need — the full row carries provenance we ignore, and a
# projection keeps ~95 MB shards from being pulled in their entirety.
READ_COLUMNS = [
    "entry_id",
    "seq_len",
    "global_plddt",
    "residue_resname",
    "residue_resnum",
    "residue_chain",
    "contact_seq_i",
    "contact_seq_j",
    "contact_degree",
]


def shard_name(index: int) -> str:
    return f"shard-{index:05d}-of-{NUM_SHARDS:05d}.parquet"


def read_shard(index: int, *, source: str = "hf", columns=READ_COLUMNS):
    """Read one contacts shard as a pyarrow Table.

    ``source`` is ``"hf"`` (public bucket, anonymous) or ``"gcs"`` (the exp147
    mirror). The HF bucket path is fetched through its resolve URL because
    pyarrow/fsspec cannot route ``hf://buckets/...``.
    """
    import pyarrow.parquet as pq

    if source == "gcs":
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(f"{GCS_MIRROR_PREFIX}/{shard_name(index)}", "rb") as fh:
            return pq.read_table(fh, columns=columns)

    import requests
    from huggingface_hub import get_token

    path = (
        "data/contacts/esm_atlas_esmfold2_distill/" + shard_name(index)
    )
    url = f"https://huggingface.co/buckets/open-athena/MarinFold/resolve/{path}"
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers, timeout=600)
    response.raise_for_status()
    return pq.read_table(io.BytesIO(response.content), columns=columns)


def ground_truth_pairs(analyzed) -> frozenset[tuple[int, int]]:
    """Canonical GT contact pairs for one structure, in **sequence-index** space.

    Applies the contacts-v1 document filters the saved raw contacts do not
    have: minimum contact degree and minimum primary-sequence separation. The
    result is exactly the pair set a contacts-v1 document for this protein
    would assert (modulo the token-budget truncation the engine handles).
    """
    return frozenset(
        (c.seq_i, c.seq_j)
        for c in analyzed.contacts
        if c.degree >= MIN_CONTACT_DEGREE
        and (c.seq_j - c.seq_i) >= MIN_SEQ_SEPARATION
    )


def iter_structures(
    shard_indices,
    *,
    source: str = "hf",
    min_len: int = 30,
    max_len: int = 400,
    min_gt: int = 4,
    limit: int | None = None,
) -> Iterator[tuple[str, object, frozenset[tuple[int, int]]]]:
    """Yield ``(entry_id, AnalyzedStructure, gt_pairs)`` from the given shards.

    Filters out proteins outside ``[min_len, max_len]`` residues or with fewer
    than ``min_gt`` ground-truth contacts (nothing to learn from, and the
    engine needs some true contacts for the posterior trigger to lean on).
    """
    from marinfold.document_structures.contacts_v1 import analyzed_from_row

    produced = 0
    for shard in shard_indices:
        table = read_shard(shard, source=source)
        for row in table.to_pylist():
            if not (min_len <= int(row["seq_len"]) <= max_len):
                continue
            analyzed = analyzed_from_row(row)
            gt = ground_truth_pairs(analyzed)
            if len(gt) < min_gt:
                continue
            yield str(row["entry_id"]), analyzed, gt
            produced += 1
            if limit is not None and produced >= limit:
                return
