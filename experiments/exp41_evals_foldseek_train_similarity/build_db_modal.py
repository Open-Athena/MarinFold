# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Modal app: build a Foldseek DB of afdb-24M cluster representatives.

Why Modal and not the laptop: the structures the clusters were defined
on are AlphaFold **v4**, and v4 is now awkward to fetch from a laptop --
EBI serves only v6 (``model_v4.cif`` 404s), and the authoritative v4 GCS
bucket (``public-datasets-deepmind-alphafold-v4``) is requester-pays. The
structures *are* embedded in the dataset itself (the ``cif_content``
column), but each shard is a single 109 MB Parquet row group, so pulling
even one structure means downloading the whole 108 MB column -- ~1.3 TB to
cover all 12,005 shards. That read is free as ingress on Modal and the
data is co-located with HF's CDN, so the extraction belongs in the cloud.

Build strategy (the read is the cost; everything else rides along):

1. ``extract_batch`` (fan-out, CPU): each worker streams a batch of shards,
   reads ``[metadata..., cif_content]``, keeps the representative rows
   (``uniprot_accession == struct_cluster_id``; ALL of them by default, or
   the first ``sample_per_shard`` if a sample is requested), writes their
   CIFs to the worker's *local* /tmp, and runs ``foldseek createdb`` over
   that batch -> a small per-batch sub-DB committed to ``/out/subdbs/``.
   Building per-batch keeps the bulky CIFs (~5 GB/batch) on the ephemeral
   worker; only the compact sub-DB (~25 MB) and a manifest marker touch the
   Volume, so the Volume never holds the ~675 GB of decompressed CIFs that
   a single central ``createdb`` would require.
2. ``build_db`` (reducer, CPU): ``foldseek concatdbs``-merges the sub-DBs
   into one ``/out/db/targetDB`` (component by component, the documented
   way), concatenates the markers into ``/out/reps_manifest.csv``, commits.

``fetch_db.py`` then pulls the ~3 GB DB + manifest to the laptop and
``query_similarity.py`` runs against it unchanged.

Cost note: the ~1.3 TB shard read dominates and is the SAME whether we keep
one representative per shard (~12 K, a coarse sample) or every one (~1.5 M,
the full set). Reading the data is the expense; ``createdb`` is sub-second.

Run (after a cost check)::

    # SMOKE: cheap end-to-end validation of the extract+merge path over a few
    # shards (~$0.05, ~2 min), on a throwaway Volume so it can't collide with
    # the full build. --wait blocks until done; no --detach needed.
    MARINFOLD_FOLDSEEK_VOLUME=afdb-foldseek-smoke \
        modal run build_db_modal.py --wait --limit-shards 20 \
        --shards-per-batch 10 --snapshot-tag smoke-20shards
    # FULL: the 1-per-cluster build over all 12,005 shards (~5 h, run detached):
    modal run --detach build_db_modal.py
"""

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "afdb-foldseek-reps-exp41"
REPO_ID = "timodonnell/afdb-24M"
# Output Volume. Defaults to the full-build namespace; override with
# ``MARINFOLD_FOLDSEEK_VOLUME`` to point a smoke build (``--limit-shards N``)
# at a throwaway Volume so its per-batch markers don't collide with the full
# build's committed batches (which would otherwise be re-merged on idempotent
# skip). The chosen name is baked into the image env below so the value the
# worker re-import sees matches the one the client mounted.
OUT_VOLUME_NAME = os.environ.get("MARINFOLD_FOLDSEEK_VOLUME", "afdb-foldseek-reps-full")

# Static Foldseek build (same artifact foldseek_env.py installs locally).
FOLDSEEK_URL = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
FOLDSEEK_BIN = "/opt/foldseek/bin/foldseek"

# Narrow metadata columns + the bulky structure column. We must read
# cif_content (108 MB/shard) because that is the only laptop-reachable
# source of the frozen v4 structures; see the module docstring.
_META_COLS = ["entry_id", "uniprot_accession", "struct_cluster_id", "split", "seq_len"]

# A Foldseek structure DB is four parallel mmseqs DBs sharing an entry
# order: the amino-acid DB (no suffix), the 3Di DB (_ss), the Cα
# coordinate DB (_ca), and the header DB (_h). concatdbs merges one
# mmseqs DB at a time, so we merge each component separately.
_DB_COMPONENTS = ["", "_ss", "_ca", "_h"]

OUT_VOL = modal.Volume.from_name(OUT_VOLUME_NAME, create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .pip_install("huggingface_hub>=0.24,<1", "pyarrow>=15", "pandas>=2.2")
    .run_commands(
        f"cd /opt && wget -q {FOLDSEEK_URL} -O foldseek.tar.gz "
        "&& tar xzf foldseek.tar.gz && rm foldseek.tar.gz",
    )
    # Ship the resolved Volume name into the container so the worker's
    # module re-import resolves the same Volume the client mounted.
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "MARINFOLD_FOLDSEEK_VOLUME": OUT_VOLUME_NAME})
)

app = modal.App(APP_NAME, image=IMAGE)


# --------------------------------------------------------------------------
# Extraction worker (fan-out): shard batch -> per-batch foldseek sub-DB
# --------------------------------------------------------------------------


@app.function(volumes={"/out": OUT_VOL}, cpu=2.0, memory=6144, timeout=60 * 60 * 2, max_containers=10)
def extract_batch(batch_id: int, shard_paths: list[str], sample_per_shard: int) -> dict:
    """Extract representative CIFs for a batch of shards and build a sub-DB.

    ``sample_per_shard <= 0`` keeps every representative in each shard (the
    full 1-per-cluster build); a positive value keeps only the first N.

    Idempotent: if the sub-DB and marker already exist, the batch is skipped
    (no 108 MB/shard re-read). Returns ``{batch_id, n_reps, skipped}``.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    subdb_dir = Path(f"/out/subdbs/batch_{batch_id:05d}")
    marker = Path(f"/out/_done/batch_{batch_id:05d}.json")
    subdb = subdb_dir / "db"

    if marker.exists() and (subdb_dir / "db.dbtype").exists():
        rows = json.loads(marker.read_text())
        return {"batch_id": batch_id, "n_reps": len(rows), "skipped": True}

    fs = HfFileSystem()
    cif_tmp = Path(f"/tmp/cifs_{batch_id:05d}")
    if cif_tmp.exists():
        shutil.rmtree(cif_tmp)
    cif_tmp.mkdir(parents=True)

    cols = list(_META_COLS) + ["cif_content"]
    rows: list[dict] = []
    for shard in shard_paths:
        path = f"datasets/{REPO_ID}/{shard}"
        with fs.open(path, "rb") as f:
            df = pq.ParquetFile(f).read(columns=cols).to_pandas()
        is_rep = df["uniprot_accession"] == df["struct_cluster_id"]
        reps = df[is_rep]
        if sample_per_shard > 0:
            reps = reps.head(sample_per_shard)
        for _, r in reps.iterrows():
            rep_id = str(r["struct_cluster_id"])
            (cif_tmp / f"{rep_id}.cif").write_text(r["cif_content"])
            rows.append({
                "representative_id": rep_id,
                "split": str(r["split"]),
                "seq_len": int(r["seq_len"]),
                "shard": shard,
            })

    # Build this batch's sub-DB from the local CIFs, then drop the CIFs.
    subdb_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([FOLDSEEK_BIN, "createdb", str(cif_tmp), str(subdb), "-v", "1"], check=True)
    shutil.rmtree(cif_tmp, ignore_errors=True)

    Path("/out/_done").mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(rows))
    OUT_VOL.commit()
    return {"batch_id": batch_id, "n_reps": len(rows), "skipped": False}


# --------------------------------------------------------------------------
# Reducer: concatdbs-merge the sub-DBs + assemble the manifest
# --------------------------------------------------------------------------


def _copy_db(src_prefix: str, dst_prefix: str) -> None:
    """Copy every ``<prefix>*`` file from ``src_prefix`` to ``dst_prefix``."""
    for path in glob.glob(src_prefix + "*"):
        suffix = path[len(src_prefix):]
        shutil.copy(path, dst_prefix + suffix)


def _concat_pair(a_prefix: str, b_prefix: str, out_prefix: str) -> None:
    """Merge two Foldseek structure DBs (all four components) into out."""
    for comp in _DB_COMPONENTS:
        out = out_prefix + comp
        for stale in glob.glob(out + "*"):
            Path(stale).unlink()
        subprocess.run(
            [FOLDSEEK_BIN, "concatdbs", a_prefix + comp, b_prefix + comp, out,
             "--threads", "1", "-v", "1"],
            check=True,
        )


def _merge_dbs(prefixes: list[str], out_prefix: str, scratch: Path) -> None:
    """Linearly concatdbs-merge ``prefixes`` into ``out_prefix``.

    Uses two ping-pong scratch prefixes so each merge writes somewhere
    distinct from its inputs, then copies the final accumulator to
    ``out_prefix``. A single input is just copied.
    """
    if len(prefixes) == 1:
        _copy_db(prefixes[0], out_prefix)
        return
    ping = str(scratch / "acc_a")
    pong = str(scratch / "acc_b")
    cur = prefixes[0]
    use_ping = True
    for nxt in prefixes[1:]:
        out = ping if use_ping else pong
        _concat_pair(cur, nxt, out)
        cur = out
        use_ping = not use_ping
    _copy_db(cur, out_prefix)


@app.function(volumes={"/out": OUT_VOL}, cpu=4.0, memory=16384, timeout=60 * 60 * 4)
def build_db(snapshot_tag: str) -> dict:
    """Merge per-batch sub-DBs into one targetDB; write the manifest."""
    import pandas as pd

    OUT_VOL.reload()
    subdbs_root = Path("/out/subdbs")
    done_dir = Path("/out/_done")
    db_dir = Path("/out/db")
    db_dir.mkdir(parents=True, exist_ok=True)

    prefixes = sorted(
        str(d / "db") for d in subdbs_root.glob("batch_*") if (d / "db.dbtype").exists()
    )
    if not prefixes:
        raise RuntimeError(f"no sub-DBs under {subdbs_root}")
    print(f"merging {len(prefixes)} sub-DBs ...")

    scratch = Path("/tmp/merge_scratch")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    _merge_dbs(prefixes, str(db_dir / "targetDB"), scratch)
    shutil.rmtree(scratch, ignore_errors=True)

    # Manifest from per-batch markers (robust to a partial fan-out).
    rows: list[dict] = []
    for marker in done_dir.glob("batch_*.json"):
        rows.extend(json.loads(marker.read_text()))
    df = (
        pd.DataFrame(rows)
        .drop_duplicates("representative_id")
        .sort_values("representative_id")
        .reset_index(drop=True)
    )
    df.to_csv("/out/reps_manifest.csv", index=False)

    # Sanity: how many entries actually landed in the merged DB?
    n_db_entries = sum(1 for _ in (db_dir / "targetDB.index").open())
    split_counts = df["split"].value_counts().to_dict()
    db_files = {p.name: p.stat().st_size for p in db_dir.glob("targetDB*")}
    print(f"merged DB entries={n_db_entries} manifest reps={len(df)} splits={split_counts}")

    OUT_VOL.commit()
    return {
        "snapshot_tag": snapshot_tag,
        "n_subdbs": len(prefixes),
        "n_db_entries": int(n_db_entries),
        "n_reps_manifest": int(len(df)),
        "split_counts": split_counts,
        "db_total_bytes": int(sum(db_files.values())),
    }


# --------------------------------------------------------------------------
# Local entrypoint: fan out extraction, then merge
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Server-side driver: orchestrate fan-out + merge entirely on Modal
# --------------------------------------------------------------------------


@app.function(volumes={"/out": OUT_VOL}, cpu=1.0, memory=2048, timeout=60 * 60 * 8)
def run_build(
    sample_per_shard: int,
    shards_per_batch: int,
    limit_shards: int,
    snapshot_tag: str,
) -> dict:
    """Drive the whole build from *inside* Modal so it survives client exit.

    The full build is ~5 h of wall time -- longer than a local client (or
    this harness's background tasks, which are reaped after ~100 min) will
    stay alive. Tethering the orchestration to ``modal run`` therefore can't
    finish it: the client dies and the ``starmap`` stops scheduling. So we
    move the fan-out (``extract_batch.starmap``) and the merge
    (``build_db``) into this function and ``.spawn`` it detached; it then
    runs to completion on Modal no matter what the client does.
    ``timeout=8h`` is the server-side hard ceiling. Idempotent: batches
    already committed to the Volume skip on resume.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    shards = sorted(f for f in files if f.endswith(".parquet"))
    if limit_shards > 0:
        shards = shards[:limit_shards]
    batches = [shards[i : i + shards_per_batch] for i in range(0, len(shards), shards_per_batch)]
    mode = "all reps/shard" if sample_per_shard <= 0 else f"{sample_per_shard} reps/shard"
    print(f"{len(shards)} shards in {len(batches)} batches ({mode}, snapshot_tag={snapshot_tag})")

    args = [(i, batch, sample_per_shard) for i, batch in enumerate(batches)]
    total = 0
    failures = 0
    for res in extract_batch.starmap(args, return_exceptions=True):
        if isinstance(res, Exception):
            failures += 1
            print(f"  batch FAILED: {res!r}")
            continue
        total += res["n_reps"]
        flag = " (skipped)" if res.get("skipped") else ""
        print(f"  batch {res['batch_id']:05d}: +{res['n_reps']} reps{flag} (cumulative {total})")
    print(f"extraction done: {total} reps across {len(batches)} batches, {failures} failed")

    if failures:
        print(f"WARNING: {failures} batches failed; merging the rest (idempotent, re-run to fill gaps).")
    print("merging sub-DBs (reducer) ...")
    info = build_db.remote(snapshot_tag)
    info["extraction_failures"] = failures
    print("DB built:")
    print(json.dumps(info, indent=2))
    return info


@app.local_entrypoint()
def main(
    sample_per_shard: int = -1,
    shards_per_batch: int = 120,
    limit_shards: int = -1,
    snapshot_tag: str = "afdb-24M-main",
    wait: bool = False,
):
    """Run the build. Two modes, by run length.

    **Full build (default, ~5 h)** -- spawn server-side and exit::

        modal run --detach build_db_modal.py

    ``main`` ``.spawn``s the driver so the build runs detached on Modal and
    survives the client exiting (and the harness's ~100-min background-task
    reap). ``--detach`` keeps the app alive after the client returns; without
    it the app is torn down on exit, so the flag is required here. Track with
    ``modal app logs afdb-foldseek-reps-exp41`` and pull the result when done.

    **Smoke build (``--wait``, minutes)** -- block until the DB is ready::

        MARINFOLD_FOLDSEEK_VOLUME=afdb-foldseek-smoke \\
            modal run build_db_modal.py --wait --limit-shards 20 \\
            --shards-per-batch 10 --snapshot-tag smoke-20shards

    ``--wait`` runs the driver with ``.remote()`` so the client blocks for the
    couple of minutes a `--limit-shards` build takes, and ``fetch_db.py`` can
    run immediately after. No ``--detach`` needed. Point a smoke build at a
    throwaway Volume (``MARINFOLD_FOLDSEEK_VOLUME``) so its per-batch markers
    don't collide with the full build's committed batches.
    """
    if wait:
        print(f"running build to completion (volume={OUT_VOLUME_NAME}, snapshot_tag={snapshot_tag}) ...")
        info = run_build.remote(sample_per_shard, shards_per_batch, limit_shards, snapshot_tag)
        print("DB built:")
        print(json.dumps(info, indent=2))
        print(f"  fetch: uv run --extra modal python fetch_db.py --volume {OUT_VOLUME_NAME} --out db_smoke")
        return
    call = run_build.spawn(sample_per_shard, shards_per_batch, limit_shards, snapshot_tag)
    print(f"spawned server-side build: function call {call.object_id}")
    print("Runs detached on Modal (launched with `modal run --detach`); this client can exit now.")
    print("  progress: modal app logs afdb-foldseek-reps-exp41")
    print(f"  result:   uv run --extra modal python fetch_db.py --volume {OUT_VOLUME_NAME} --out db_full")
