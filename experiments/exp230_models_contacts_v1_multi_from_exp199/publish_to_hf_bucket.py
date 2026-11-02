# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Publish the exp230 fine-tune to the public ``open-athena/MarinFold`` bucket.

Follows exp163's publisher, with one difference that matters and one that does
not.  The one that does not: exp230 trained on a local node, so the source is a
LOCAL directory and there is no GCS hop.  The one that does:

**The export must be repaired before it is published.**  levanter's HF export
writes rope in the transformers-5 form only -- ``rope_parameters`` present,
top-level ``rope_theta`` and ``rope_scaling`` absent.  transformers 4.x does not
read ``rope_parameters``; it silently falls back to a default rope and the model
loses 0.76 nats/token with no error anywhere.  That bug already forced one
retraction in #163, and exp199's own published export carries it.  So this script
restores the 4.x keys FROM ``rope_parameters`` before upload, and refuses to
publish if it cannot.

Verified on this fine-tune (see the commit message): base and export resolve to
bit-identical ``inv_freq`` under transformers 5.15, so the repair is purely about
what OLDER readers see -- it changes no number measured on this node.

Three refusals, each one a bug that has actually shipped somewhere:

* no top-level ``rope_theta`` after repair -> the #163 retraction
* tokenizer files missing -> the standing rule is that a published checkpoint
  carries its tokenizer; a contacts-v1 checkpoint without one is unusable
* tokenizer id 7 is not ``<contacts-v1.multi>`` -> the wrong tokenizer was
  exported and multi-mode inference would silently mean something else

    HF_TOKEN=... python publish_to_hf_bucket.py \\
        --src ~/exp230_data/checkpoints/hf/step-1989 \\
        --run plm-exp230-cv1-multi-1_5b-lr1e-4-e1-cos-a100 --step 1989 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

BUCKET = "open-athena/MarinFold"
MULTI_TOKEN, MULTI_ID, VOCAB = "<contacts-v1.multi>", 7, 2845


def repair_rope(cfg: dict) -> tuple[dict, list[str]]:
    """Restore the transformers-4.x rope keys from ``rope_parameters``.

    Returns the config and a list of what changed.  Idempotent: a config that
    already carries the 4.x keys is returned untouched.
    """
    notes = []
    rp = cfg.get("rope_parameters")
    if not rp:
        return cfg, notes
    if cfg.get("rope_theta") is None and "rope_theta" in rp:
        cfg["rope_theta"] = rp["rope_theta"]
        notes.append(f"rope_theta <- {rp['rope_theta']}")
    # rope_scaling is the 4.x spelling of the same dict WITHOUT rope_theta in it.
    if cfg.get("rope_scaling") is None and rp.get("rope_type", "default") != "default":
        cfg["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
        notes.append(f"rope_scaling <- {cfg['rope_scaling']}")
    return cfg, notes


def check_tokenizer(src: Path) -> None:
    tj = src / "tokenizer.json"
    if not tj.exists():
        raise SystemExit(f"FATAL: no tokenizer.json in {src} -- a published checkpoint "
                         f"must carry its tokenizer")
    t = json.loads(tj.read_text())
    added = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
    vocab = t["model"]["vocab"]
    inv = {i: s for s, i in vocab.items()} if isinstance(vocab, dict) else {}
    got = added.get(MULTI_ID, inv.get(MULTI_ID))
    if got != MULTI_TOKEN:
        raise SystemExit(f"FATAL: tokenizer id {MULTI_ID} is {got!r}, expected {MULTI_TOKEN!r} "
                         f"-- the wrong tokenizer was exported")
    if len(vocab) != VOCAB:
        raise SystemExit(f"FATAL: vocab is {len(vocab)}, expected {VOCAB} (id 7 is renamed "
                         f"IN PLACE; any resize means the ids drifted)")
    print(f"[publish] tokenizer ok: id {MULTI_ID} = {MULTI_TOKEN}, vocab {len(vocab)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local HF export directory")
    ap.add_argument("--run", required=True, help="W&B run name (the bucket path key)")
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    if not src.is_dir():
        raise SystemExit(f"FATAL: {src} is not a directory")
    check_tokenizer(src)

    cfg = json.loads((src / "config.json").read_text())
    cfg, notes = repair_rope(cfg)
    if cfg.get("rope_theta") is None:
        raise SystemExit("FATAL: config.json still has no top-level rope_theta after repair; "
                         "transformers 4.x would silently mis-load this checkpoint")
    print(f"[publish] rope repair: {notes or 'nothing to do (already 4.x-readable)'}")
    print(f"[publish] rope_theta={cfg['rope_theta']} rope_scaling={cfg.get('rope_scaling')}")

    token = os.environ.get("HF_TOKEN")
    if not token and not a.dry_run:
        raise SystemExit("HF_TOKEN must be set (open-athena-scoped; a personal token 403s)")

    dest_prefix = f"checkpoints/{a.run}/hf/step-{a.step}"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in sorted(src.iterdir()):
            if f.is_file():
                shutil.copy2(f, stage / f.name)
        (stage / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        total = sum(f.stat().st_size for f in stage.iterdir())
        print(f"[publish] staged {len(list(stage.iterdir()))} file(s), {total/1e9:.2f} GB")
        dest = f"hf://buckets/{a.bucket}/{dest_prefix}"
        print(f"[publish] sync -> {dest}")
        if a.dry_run:
            for f in sorted(stage.iterdir()):
                print(f"[publish]   would upload {f.name} ({f.stat().st_size/1e6:.1f} MB)")
            print("[publish] DRY RUN, nothing uploaded")
            return 0
        from huggingface_hub import HfApi
        plan = HfApi(token=token).sync_bucket(source=str(stage), dest=dest, verbose=True)
        print(f"[publish] done: {plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
