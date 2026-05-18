"""Deterministic sampling of test-split AFDB entries for the heatmap eval.

The training data is ``timodonnell/protein-docs`` subset
``contacts-and-distances-v1-5x``. Its ``split=test`` is a held-out
~1% slice (~53.5k rows) assigned by a hash of ``split_cluster_id``,
so any entry from that split was never seen during training.

We sample N entries deterministically given a seed. The sampling
streams the dataset (no full materialization) so the notebook
doesn't pay for the ~5 GB parquet download just to pick 10 rows.

Yielding ``ProteinSpec`` records makes the downstream notebook
tolerant to AFDB version churn: ``cif_url`` is resolved via the
public AlphaFold API at sample time, so we always land on the
currently-live ``model_v<N>.cif`` regardless of which version was
canonical when training data was generated.
"""

import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import urlopen

DATASET = "timodonnell/protein-docs"
SUBSET = "contacts-and-distances-v1-5x"
SPLIT = "test"

# Streaming window: how many rows from the test split we pull before
# shuffling. Larger = more uniform sample over the split; smaller =
# faster. 5_000 is plenty for picking 10 (and the dataset's row order
# is already cluster-shuffled by the upstream build pipeline).
DEFAULT_POOL_SIZE = 5_000


@dataclass(frozen=True)
class ProteinSpec:
    """A single sampled test-split entry, resolved to a downloadable AFDB cif."""

    entry_id: str           # e.g. "AF-A0A1C0V126-F1"
    uniprot_accession: str  # e.g. "A0A1C0V126"
    seq_len: int
    cif_url: str            # currently-live AFDB model_v<N>.cif URL


def _resolve_cif_url(uniprot_accession: str) -> str | None:
    """Ask the AlphaFold API for the current cif URL for an accession.

    AFDB versions roll (v4 → v6 …); the dataset's training-time
    ``entry_id`` is stable but the cif URL we download from is
    not. This API call returns whichever version is currently live,
    or ``None`` if the accession is no longer in AFDB (rare — happens
    when UniProt retires the accession after dataset generation).
    """
    api = f"https://www.alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}"
    try:
        with urlopen(api, timeout=30) as response:
            records = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not records:
        return None
    # The canonical (non-isoform) entry is the first record. Each record
    # has bcifUrl / cifUrl / pdbUrl pointing at the latest model version.
    return records[0]["cifUrl"]


def select_test_proteins(
    n: int = 10,
    *,
    seed: int = 0,
    max_seq_len: int | None = 150,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> list[ProteinSpec]:
    """Sample ``n`` AFDB entries from the test split, deterministically.

    Args:
        n: How many entries to return.
        seed: Random seed; same seed → same n entries.
        max_seq_len: Cap on sequence length. Heatmaps for very long
            proteins are dominated by the saturated bin (every pair
            > 32 Å). Default 150 keeps the run snappy on a single
            consumer GPU. Pass ``None`` to disable.
        pool_size: How many rows of the test split to stream before
            shuffling. Must be >= n.
    """
    if pool_size < n:
        raise ValueError(f"pool_size ({pool_size}) must be >= n ({n})")

    # Lazy import — datasets is heavy and not needed at module import time
    # for the dataclass / URL helper.
    import random
    from datasets import load_dataset

    ds = load_dataset(DATASET, SUBSET, split=SPLIT, streaming=True)
    pool: list[dict] = []
    for row in ds:
        if max_seq_len is not None and row["seq_len"] > max_seq_len:
            continue
        pool.append({
            "entry_id": row["entry_id"],
            "uniprot_accession": row["uniprot_accession"],
            "seq_len": int(row["seq_len"]),
        })
        if len(pool) >= pool_size:
            break

    if len(pool) < n:
        raise RuntimeError(
            f"Test split yielded only {len(pool)} entries under max_seq_len "
            f"{max_seq_len} (wanted {n}). Loosen max_seq_len or raise pool_size."
        )

    # Shuffle the pool deterministically and walk it, resolving each
    # candidate's cif URL. Skip entries whose UniProt accession has
    # been retired from AFDB since the dataset was built (rare, but
    # happens — e.g. the source UniProt record was deleted). Stop
    # once we have n valid specs.
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)

    out: list[ProteinSpec] = []
    skipped: list[str] = []
    for rec in shuffled:
        if len(out) >= n:
            break
        url = _resolve_cif_url(rec["uniprot_accession"])
        if url is None:
            skipped.append(rec["entry_id"])
            continue
        out.append(ProteinSpec(
            entry_id=rec["entry_id"],
            uniprot_accession=rec["uniprot_accession"],
            seq_len=rec["seq_len"],
            cif_url=url,
        ))

    if len(out) < n:
        raise RuntimeError(
            f"Only resolved {len(out)} / {n} AFDB URLs from a pool of "
            f"{len(pool)} entries (skipped {len(skipped)}: {skipped[:5]}...). "
            f"Raise pool_size."
        )
    if skipped:
        print(f"select_test_proteins: skipped {len(skipped)} retired entries: {skipped}")
    return out


def download_cif(spec: ProteinSpec, cache_dir) -> "pathlib.Path":
    """Fetch ``spec.cif_url`` to ``cache_dir / <entry_id>.cif`` if not present."""
    from pathlib import Path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{spec.entry_id}.cif"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with urlopen(spec.cif_url, timeout=60) as response:
        data = response.read()
    out_path.write_bytes(data)
    return out_path
