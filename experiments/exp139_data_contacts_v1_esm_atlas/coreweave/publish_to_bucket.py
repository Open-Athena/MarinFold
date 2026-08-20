# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp139 Stage 2 — publish the CoreWeave `analyzed/` set to the HF MarinFold bucket.

Runs **on a CoreWeave pod** (s3 creds auto-injected; HF token via ``HF_TOKEN``):
the workstation uplink is ~2.4 MB/s (~61 h for this corpus), a pod is
cloud-to-cloud.

Per analyzed part (one pass, no intermediate materialization) it splits the
combined row into the two published views and uploads both:

- **documents corpus** (training): every column EXCEPT the residue/contact
  arrays -> ``data/document_structures/contacts_v1_esm_atlas/train/``
- **reusable pyconfind contacts**: ids + provenance + the raw residue/contact
  arrays -> ``data/contacts/esm_atlas_esmfold2_distill/``

Parts are re-numbered to a clean contiguous ``shard-{i:05d}-of-{N:05d}``
naming (the inputs carry a mix of ``analyzed-*`` and ``resume*-*`` prefixes
from the main run + its resume batches; the union is what matters).

Resumable: a target already present in the bucket with a non-zero size is
skipped, so a killed run just continues.

Env:
    ANALYZED_PREFIX  s3 prefix holding the analyzed parts
    BUCKET_ID        HF bucket (default open-athena/MarinFold)
    DOCS_PATH        in-bucket path for the documents corpus
    CONTACTS_PATH    in-bucket path for the reusable contacts
    HF_TOKEN         HF token with open-athena write (required)
    PUBLISH_WORKERS  concurrent parts (default 8)
    TMP_DIR          pod-local scratch (default /tmp/exp139_publish)
    LIMIT            only publish the first N parts (smoke)
    DRY_RUN          "1" to plan and exit
"""

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ANALYZED_PREFIX = os.environ.get(
    "ANALYZED_PREFIX",
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/analyzed",
).rstrip("/")
BUCKET_ID = os.environ.get("BUCKET_ID", "open-athena/MarinFold")
DOCS_PATH = os.environ.get(
    "DOCS_PATH", "data/document_structures/contacts_v1_esm_atlas/train"
).strip("/")
CONTACTS_PATH = os.environ.get(
    "CONTACTS_PATH", "data/contacts/esm_atlas_esmfold2_distill"
).strip("/")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
PUBLISH_WORKERS = int(os.environ.get("PUBLISH_WORKERS", "8"))
TMP_DIR = Path(os.environ.get("TMP_DIR", "/tmp/exp139_publish"))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

# The reusable pyconfind record's array columns (marinfold
# contacts_v1.ANALYZED_ROW_COLUMNS' array half). Hardcoded because this script
# runs in a minimal env (huggingface_hub>=1.5 conflicts with marinfold's
# transformers->hub<1.0 pin), so marinfold is not importable here.
_ARRAY_COLS = (
    "residue_resname", "residue_resnum", "residue_chain",
    "contact_seq_i", "contact_seq_j", "contact_degree",
)
_ID_COLS = ("entry_id", "seq_len", "global_plddt", "num_contacts")
_PROVENANCE = ("seq_cluster_id", "cluster_size", "ptm", "plddt_std", "source", "split")

# HF rate-limits the bucket xet-write-token endpoint hard under concurrency
# (16 workers => sustained 429s). Keep PUBLISH_WORKERS small (~4-6) and give the
# backoff a long tail so a burst of 429s rides out instead of exhausting retries.
_MAX_RETRIES = 9
_BASE_BACKOFF = 5.0


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _plan():
    """List analyzed parts (sorted) -> [(src_uri, out_name)] with clean numbering."""
    import fsspec

    fs, root = fsspec.core.url_to_fs(ANALYZED_PREFIX)
    parts = sorted(p for p in fs.ls(root, detail=False) if p.endswith(".parquet"))
    if LIMIT is not None:
        parts = parts[:LIMIT]
    n = len(parts)
    return [(fs.unstrip_protocol(p), f"shard-{i:05d}-of-{n:05d}.parquet")
            for i, p in enumerate(parts)], n


def _publish_one(hffs, src_uri: str, out_name: str) -> str:
    import fsspec
    import pyarrow.parquet as pq

    docs_dst = f"buckets/{BUCKET_ID}/{DOCS_PATH}/{out_name}"
    cont_dst = f"buckets/{BUCKET_ID}/{CONTACTS_PATH}/{out_name}"

    # Resume: skip when both targets already exist non-empty.
    try:
        if all(hffs.exists(d) and hffs.info(d).get("size", 0) > 0
               for d in (docs_dst, cont_dst)):
            return f"skip  {out_name}"
    except Exception:  # noqa: BLE001 — existence check is best-effort
        pass

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    docs_tmp = TMP_DIR / f"docs-{out_name}"
    cont_tmp = TMP_DIR / f"cont-{out_name}"
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            src_fs, src_path = fsspec.core.url_to_fs(src_uri)
            with src_fs.open(src_path, "rb") as fh:
                table = pq.read_table(fh)
            present = set(table.column_names)

            docs_cols = [c for c in table.column_names if c not in _ARRAY_COLS]
            cont_cols = list(dict.fromkeys(
                [c for c in (*_ID_COLS, *_ARRAY_COLS, *_PROVENANCE) if c in present]))

            pq.write_table(table.select(docs_cols), docs_tmp, compression="zstd")
            pq.write_table(table.select(cont_cols), cont_tmp, compression="zstd")

            for tmp, dst in ((docs_tmp, docs_dst), (cont_tmp, cont_dst)):
                with open(tmp, "rb") as fsrc, hffs.open(dst, "wb") as fdst:
                    while chunk := fsrc.read(32 << 20):
                        fdst.write(chunk)
            return (f"put   {out_name} rows={table.num_rows} "
                    f"docs={docs_tmp.stat().st_size / 1e6:.0f}MB "
                    f"contacts={cont_tmp.stat().st_size / 1e6:.0f}MB")
        except Exception as exc:  # noqa: BLE001 — retry transient s3/HF errors
            last_exc = exc
            sleep = min(_BASE_BACKOFF * (2 ** attempt), 120.0)  # cap the tail
            _log(f"retry {out_name} {attempt + 1}/{_MAX_RETRIES} after {sleep:.0f}s: {exc!r}")
            time.sleep(sleep)
        finally:
            for t in (docs_tmp, cont_tmp):
                try:
                    if t.exists():
                        t.unlink()
                except OSError:
                    pass
    raise RuntimeError(f"giving up on {out_name}: {last_exc!r}")


def main() -> int:
    plan, n = _plan()
    _log(f"[publish] {ANALYZED_PREFIX} -> hf://buckets/{BUCKET_ID}")
    _log(f"[publish]   docs     -> {DOCS_PATH}/")
    _log(f"[publish]   contacts -> {CONTACTS_PATH}/")
    _log(f"[publish] {len(plan)} parts (of {n}), {PUBLISH_WORKERS} workers"
         + (" [DRY_RUN]" if DRY_RUN else ""))
    for s, o in plan[:3]:
        _log(f"[publish]   e.g. {s} -> {o}")
    if DRY_RUN:
        return 0
    if not HF_TOKEN:
        _log("[publish] FATAL: HF_TOKEN is required to write the bucket")
        return 2

    from huggingface_hub import HfFileSystem

    hffs = HfFileSystem(token=HF_TOKEN)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=PUBLISH_WORKERS,
                            thread_name_prefix="publish") as pool:
        futs = {pool.submit(_publish_one, hffs, s, o): o for s, o in plan}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                _log(f"[publish] {fut.result()}  [{done + failed + 1}/{len(plan)}]")
                done += 1
            except Exception:  # noqa: BLE001 — one part's failure is not fatal
                failed += 1
                _log(f"[publish] FAIL {name}\n{traceback.format_exc()}")

    _log(f"[publish] complete: {done} ok, {failed} failed of {len(plan)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
