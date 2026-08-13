# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp222 corpora to the public ``open-athena/MarinFold`` bucket.

Run this one **outside** the experiment environment: the bucket API needs
``huggingface_hub>=1.5``, which conflicts with the ``transformers<5`` pin
marinfold carries (see ``pyproject.toml``)::

    uv run --no-project --with "huggingface_hub>=1.5" python publish_to_hf.py --dry-run
    uv run --no-project --with "huggingface_hub>=1.5" python publish_to_hf.py

Uploads, under ``data/document_structures/``:

* ``contacts_v1_pdb_monomers/documents/``  -- the document shards
* ``contacts_v1_pdb_multimers/documents/`` -- ditto
* ``contacts_v1_pdb_deduped/documents/``   -- one representative per sequence
  cluster, pre-shuffled and directly trainable
* ``contacts_v1_pdb_curation/{metadata,ledger}/`` -- the entry scan, the RCSB
  cluster file and the per-entry curation ledger, so the corpora can be
  re-derived and audited without the local mirror.

Each prefix also gets a ``README.md`` rendered on the bucket's web view, sourced
from ``<root>/readme/<name>/README.md``.

The contacts-v1 tokenizer is written next to each corpus (repo convention:
a corpus ships with the tokenizer that reads it). Build it from the library
first, so it is provably the vocabulary the documents were generated under
rather than whatever a pinned Hub revision happens to hold::

    uv run contacts-v1 tokenizer --save-local /data/exp222_pdb_curation/tokenizer

Buckets are **not** repos: ``upload_folder`` / ``upload_large_folder`` do not
address them (and ``upload_large_folder`` has no ``path_in_repo`` at all).
The bucket surface is ``HfApi.sync_bucket``, whose destinations are
``hf://buckets/<namespace>/<name>/<prefix>`` -- the Python form of
``hf buckets sync``.

Writing needs an **open-athena-scoped** token; ``hf auth whoami`` must list
the org. Reading back is anonymous.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


BUCKET_URI = "hf://buckets/open-athena/MarinFold"
PREFIX = "data/document_structures"

# Local scratch that should never be published: the smoke-test fixtures the
# pipeline was developed against.
EXCLUDE = ["_smoke*", "*.log"]

# Sanity ceiling on the per-prefix README, which is synced into a prefix that
# also holds `documents/` and `tokenizer/`. sync_bucket does not delete by
# default, so the subdirectories are untouched, but a stray large file in the
# readme dir would be silently published.
MAX_README_BYTES = 64 * 1024


def sync(api: HfApi, local: Path, remote: str, dry_run: bool) -> int:
    files = sorted(
        p for p in local.rglob("*")
        if p.is_file() and not p.name.startswith("_")
    )
    if local.name in ("monomers", "multimers", "deduped", "curation") and "readme" in local.parts:
        for f in files:
            if f.suffix != ".md" or f.stat().st_size > MAX_README_BYTES:
                raise SystemExit(f"unexpected file in a README dir: {f}")
    total = sum(p.stat().st_size for p in files)
    destination = f"{BUCKET_URI}/{remote}"
    print(f"  {local} -> {destination}  ({len(files)} files, {total/1e9:.2f} GB)")
    api.sync_bucket(
        str(local), destination, exclude=EXCLUDE, dry_run=dry_run, verbose=False
    )
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/exp222_pdb_curation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    api = HfApi()
    who = api.whoami()
    orgs = {o["name"] for o in who.get("orgs", [])}
    print(f"authenticated as {who.get('name')}; orgs={sorted(orgs)}")
    if "open-athena" not in orgs:
        raise SystemExit(
            "token is not open-athena-scoped; bucket writes will 403. "
            "Switch to the org token (see AGENTS.md)."
        )

    tokenizer = args.root / "tokenizer"
    if not tokenizer.is_dir():
        raise SystemExit(
            f"{tokenizer} is missing. Build it first so the published corpus "
            f"carries the exact vocabulary it was generated under:\n"
            f"    uv run contacts-v1 tokenizer --save-local {tokenizer}"
        )

    plan = []
    for subset, name in [
        ("monomers", "contacts_v1_pdb_monomers"),
        ("multimers", "contacts_v1_pdb_multimers"),
        ("deduped", "contacts_v1_pdb_deduped"),
    ]:
        plan.append((args.root / "docs" / subset, f"{PREFIX}/{name}/documents"))
        plan.append((tokenizer, f"{PREFIX}/{name}/tokenizer"))
        plan.append((args.root / "readme" / subset, f"{PREFIX}/{name}"))
    plan += [
        (args.root / "metadata", f"{PREFIX}/contacts_v1_pdb_curation/metadata"),
        (args.root / "ledger", f"{PREFIX}/contacts_v1_pdb_curation/ledger"),
        (args.root / "readme" / "curation", f"{PREFIX}/contacts_v1_pdb_curation"),
    ]
    uploaded = 0
    for local, remote in plan:
        if not local.is_dir():
            print(f"  (skipping {remote}: {local} missing)")
            continue
        uploaded += sync(api, local, remote, args.dry_run)

    print(f"{'would sync' if args.dry_run else 'synced'} {uploaded} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
