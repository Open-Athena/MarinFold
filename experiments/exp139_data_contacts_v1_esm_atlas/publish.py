# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 — split the combined ``analyzed/`` set into two published views + upload.

Stage 1 writes ONE combined parquet dataset per part (``analyzed/``) so pyconfind
runs only once. This script produces the two logical views the corpus consumers
want, by cheap **column projection** (no pyconfind, no re-generation):

- **documents corpus** (for training): ``entry_id``, ``structure``, ``document``
  + the contacts-v1 per-doc metadata + provenance — *drops* the residue/contact
  arrays. → HF bucket ``data/document_structures/contacts_v1_esm_atlas/train/``.
- **reusable contacts** (for future doc types): ``entry_id`` + the residue /
  contact arrays (:data:`ANALYZED_ROW_COLUMNS`) + provenance — *drops* the
  ``document``. → HF bucket ``data/contacts/esm_atlas_esmfold2_distill/``.

Run this **in-region** (an iris pod / us-central1 VM) so the ~150–250 GB read of
``analyzed/`` and the two writes stay local; the projected parquet is written
back to GCS, then copied to the HF bucket. It is I/O-bound and embarrassingly
parallel per part — use ``--workers``.

This is the finalize step (runs after the full Stage 1 completes); the exact
counts / tokenizer push / dataset READMEs are filled in against the real output.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Columns that make up the reusable pyconfind-contacts record (kept in the
# contacts view). Imported from marinfold so the two stay in lockstep.
from marinfold.document_structures.contacts_v1 import ANALYZED_ROW_COLUMNS

# Provenance carried into BOTH views (see generate_rows.PASSTHROUGH_COLUMNS).
PROVENANCE = ("seq_cluster_id", "cluster_size", "ptm", "plddt_std", "source", "split")

# The documents view drops the big arrays but keeps everything else (document
# text + per-doc metadata + provenance). We express it as "all columns except
# the array columns", so new contacts-v1 metadata fields are picked up
# automatically.
_ARRAY_COLUMNS = (
    "residue_resname", "residue_resnum", "residue_chain",
    "contact_seq_i", "contact_seq_j", "contact_degree",
)


def _list_parts(fs, glob: str) -> list[str]:
    matches = sorted(fs.glob(glob))
    if not matches:
        raise FileNotFoundError(f"no parquet matched {glob!r}")
    return matches


def _project_one(in_uri: str, docs_uri: str, contacts_uri: str) -> str:
    import fsspec
    import pyarrow.parquet as pq

    in_fs, in_path = fsspec.core.url_to_fs(in_uri)
    with in_fs.open(in_path, "rb") as f:
        table = pq.read_table(f)
    present = set(table.column_names)

    contacts_cols = [c for c in ("entry_id", *ANALYZED_ROW_COLUMNS, *PROVENANCE)
                     if c in present]
    # dedupe while preserving order (entry_id appears in ANALYZED_ROW_COLUMNS)
    contacts_cols = list(dict.fromkeys(contacts_cols))
    docs_cols = [c for c in table.column_names if c not in _ARRAY_COLUMNS]

    for cols, out_uri in ((docs_cols, docs_uri), (contacts_cols, contacts_uri)):
        out_fs, out_path = fsspec.core.url_to_fs(out_uri)
        with out_fs.open(out_path, "wb") as f:
            pq.write_table(table.select(cols), f, compression="zstd")
    return in_uri


def cmd_project(args: argparse.Namespace) -> None:
    import fsspec

    fs, _ = fsspec.core.url_to_fs(args.analyzed)
    parts = _list_parts(fs, args.analyzed)
    print(f"[publish] projecting {len(parts)} parts", file=sys.stderr)

    def plan(in_uri: str) -> tuple[str, str, str]:
        base = os.path.basename(in_uri).replace("analyzed", "shard")
        return (in_uri, f"{args.docs_out.rstrip('/')}/{base}",
                f"{args.contacts_out.rstrip('/')}/{base}")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_project_one, *plan(fs.unstrip_protocol(p))): p
                for p in parts}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(parts):
                print(f"[publish] {done}/{len(parts)}", file=sys.stderr)
    print("[publish] projection complete. Next: `hf buckets cp` the two GCS "
          "prefixes to the bucket, push the tokenizer, and write dataset READMEs.",
          file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python publish.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("project", help="Split analyzed/ into docs + contacts views (GCS→GCS).")
    pr.add_argument("--analyzed", required=True,
                    help="Glob of the Stage-1 combined output, e.g. "
                         "gs://marin-us-central1/.../exp139_esm_atlas_contacts_v1/"
                         "analyzed/analyzed-*.parquet")
    pr.add_argument("--docs-out", required=True,
                    help="GCS prefix for the documents-only projection.")
    pr.add_argument("--contacts-out", required=True,
                    help="GCS prefix for the reusable-contacts projection.")
    pr.add_argument("--workers", type=int, default=16)
    pr.set_defaults(func=cmd_project)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
