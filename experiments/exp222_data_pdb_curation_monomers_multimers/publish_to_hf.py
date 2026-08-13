# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the exp222 corpora to the public ``open-athena/MarinFold`` bucket.

Run this one **outside** the experiment environment: the bucket API needs
``huggingface_hub>=1.5``, which conflicts with the ``transformers<5`` pin
marinfold carries (see ``pyproject.toml``)::

    uv run --no-project --with "huggingface_hub>=1.5" python publish_to_hf.py

Uploads, under ``data/document_structures/``:

* ``contacts_v1_pdb_monomers/``  -- the document shards
* ``contacts_v1_pdb_multimers/`` -- ditto
* each with the entry metadata, the curation ledger, and the contacts-v1
  tokenizer co-located (repo rule: a corpus ships with the tokenizer that
  reads it).

Writing needs an **open-athena-scoped** token; ``hf auth whoami`` must list
the org. Reading back is anonymous.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


BUCKET = "open-athena/MarinFold"
PREFIX = "data/document_structures"
TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"


def upload_tree(api: HfApi, local: Path, remote: str, dry_run: bool) -> int:
    files = sorted(p for p in local.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"  {local} -> {remote}  ({len(files)} files, {total/1e9:.2f} GB)")
    if dry_run:
        return len(files)
    api.upload_large_folder(
        repo_id=BUCKET,
        repo_type="bucket",
        folder_path=str(local),
        path_in_repo=remote,
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
    if "open-athena" not in orgs and not args.dry_run:
        raise SystemExit(
            "token is not open-athena-scoped; bucket writes will 403. "
            "Switch to the org token (see AGENTS.md)."
        )

    uploaded = 0
    for subset in ("monomers", "multimers"):
        local = args.root / "docs" / subset
        if not local.is_dir():
            print(f"  (skipping {subset}: {local} missing)")
            continue
        uploaded += upload_tree(
            api, local, f"{PREFIX}/contacts_v1_pdb_{subset}/documents", args.dry_run
        )

    for name, local in [
        ("metadata", args.root / "metadata"),
        ("ledger", args.root / "ledger"),
    ]:
        if local.is_dir():
            uploaded += upload_tree(
                api, local, f"{PREFIX}/contacts_v1_pdb_curation/{name}", args.dry_run
            )

    print(f"{'would upload' if args.dry_run else 'uploaded'} {uploaded} files")
    print(
        "tokenizer: co-locate a copy of "
        f"{TOKENIZER_REPO} under each corpus prefix (repo rule: "
        "a corpus ships with the tokenizer that reads it)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
