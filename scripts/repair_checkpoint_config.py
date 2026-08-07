# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Find and fix checkpoints whose rope block only transformers 5.x can read.

Why this exists. A checkpoint exported by transformers 5.x states rope as a
``rope_parameters`` block. Anything reading it with transformers 4.x — which is
what MarinFold pins, and what a lot of downstream code uses — does not error,
it *ignores* the block and falls back to the architecture default. On the #117
checkpoint that means rope theta 10000 instead of the trained 500000, worth
**0.77 nats/token** on real documents (MarinFold #180, PR #184).

MarinFold's own inference path repairs this on load, so the CLI and the eval
harness are correct either way. This script is for everything else — a bare
``AutoModelForCausalLM.from_pretrained``, a foreign eval worker, a Colab. The
only thing that helps those is a corrected artifact.

The repair is **additive**: ``rope_parameters`` stays and ``rope_theta`` /
``rope_scaling`` are added alongside, so the file reads correctly under both
transformers majors. Verified on 4.57 and 5.14.

Two modes, both narrow on purpose:

``--survey``
    Read-only. Report which published checkpoints carry the defect. Writes
    nothing, anywhere. Sweeps both the HF *model repos* below **and** the
    ``open-athena/MarinFold`` bucket — the bucket was originally left out on
    the assumption that everything in it was a transformers 4.x export, which
    was wrong: exp120 and exp108 were 5.x exports and had been serving the
    defect since publication (MarinFold #197).

``<dir>``
    Repair a **local** checkpoint directory in place. This is the supported
    route: copy the checkpoint down, repair it, and republish under
    ``open-athena/MarinFold`` with a PROVENANCE.md — as
    ``contacts-v1-exp117-1.5B`` was. Deliberately *not* an in-place rewrite of
    somebody else's published repo: those belong to the experiment that
    produced them, and a silent third-party edit is worse than a clearly
    labelled republished copy.

    uv run --with huggingface-hub --with numpy --with fsspec --with pyyaml \\
        python scripts/repair_checkpoint_config.py --survey
    uv run ... python scripts/repair_checkpoint_config.py ~/staged/step-35679
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "marinfold"))

from marinfold.inference._config import (  # noqa: E402
    needs_rope_repair,
    repair_config_file,
    repair_rope,
)

# HF *model repos* swept by --survey.
SURVEY_REPOS = (
    "open-athena/marinfold-exp117",
    "open-athena/marinfold-exp146",
    "open-athena/marinfold-exp75",
)

# The HF *bucket*, swept as well. Buckets are a separate storage layer: the
# model-repo API (list_repo_files / hf_hub_download) does not see into them,
# hence the parallel code path below.
SURVEY_BUCKET = "open-athena/MarinFold"
SURVEY_BUCKET_PREFIX = "checkpoints/"


def _report(path: str, raw: dict, tally: dict) -> None:
    """Classify one config and print a line for it. Mutates ``tally``."""
    if needs_rope_repair(raw):
        tally["affected"] += 1
        fixed = repair_rope(raw)
        print(f"  AFFECTED {path}")
        print(f"             exported by transformers {raw.get('transformers_version')}")
        print(f"             4.x reads the architecture default; the trained "
              f"value is rope_theta={fixed['rope_theta']}")
        return

    # Neither shape present. Nothing in the file says what rope was trained, so
    # this script cannot repair it — but it is not "ok" either: every loader
    # silently takes the architecture default. Recover the true values from the
    # levanter run config (the ``rope=`` argument on the model config, or its
    # default) and write them in by hand.
    if raw.get("rope_theta") is None and not isinstance(raw.get("rope_parameters"), dict):
        tally["unknown"] += 1
        print(f"  NO ROPE  {path}")
        print(f"             exported by transformers {raw.get('transformers_version')}")
        print( "             no rope_theta, no rope_parameters — loaders fall back to "
               "the architecture default. Check the levanter run config.")
        return

    tally["clean"] += 1
    scaling = raw.get("rope_scaling")
    rope_type = scaling.get("rope_type") if isinstance(scaling, dict) else None
    print(f"  ok       {path}  (rope_theta={raw.get('rope_theta')}, rope_type={rope_type})")


def _survey_repos(repos, tally) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for repo in repos:
        try:
            configs = sorted(
                f for f in api.list_repo_files(repo)
                if f.endswith("config.json") and "/hf/" in f and "tokenizer" not in f
            )
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"!! {repo}: {type(exc).__name__}: {exc}")
            continue
        print(f"\n### {repo}  ({len(configs)} model configs)")
        for path in configs:
            _report(path, json.loads(Path(hf_hub_download(repo, path)).read_text()), tally)


def _survey_bucket(bucket: str, prefix: str, tally) -> None:
    """Sweep a HF bucket. Needs huggingface_hub >= 1.5 for the bucket API.

    ``token=False`` throughout: the bucket is anon-readable, and a stale cached
    credential would otherwise 401 a read-only survey.
    """
    try:
        from huggingface_hub import BucketFile, download_bucket_files, list_bucket_tree
    except ImportError:
        print(f"\n!! {bucket}: needs huggingface_hub>=1.5 for the bucket API — skipped")
        return

    import tempfile

    try:
        entries = list_bucket_tree(bucket, prefix=prefix, recursive=True, token=False)
        configs = sorted(
            (f for f in entries
             if isinstance(f, BucketFile)
             and f.path.endswith("/config.json") and "tokenizer" not in f.path),
            key=lambda f: f.path,
        )
    except Exception as exc:  # noqa: BLE001 — report and continue
        print(f"!! {bucket}: {type(exc).__name__}: {exc}")
        return

    print(f"\n### bucket {bucket}/{prefix}  ({len(configs)} model configs)")
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [(f, Path(tmp) / f"{i}.json") for i, f in enumerate(configs)]
        download_bucket_files(bucket, pairs, raise_on_missing_files=True, token=False)
        for f, local in pairs:
            _report(f.path, json.loads(local.read_text()), tally)


def survey(repos, bucket=SURVEY_BUCKET, prefix=SURVEY_BUCKET_PREFIX) -> int:
    tally = {"affected": 0, "clean": 0, "unknown": 0}
    _survey_repos(repos, tally)
    if bucket:
        _survey_bucket(bucket, prefix, tally)

    print(f"\n{tally['affected']} affected, {tally['unknown']} with no rope info, "
          f"{tally['clean']} correct")
    if tally["affected"]:
        print("Fix: download the checkpoint, run this script on the local copy, "
              "then republish with a PROVENANCE.md.")
    return 0


def repair_local(directory: Path) -> int:
    config_path = directory / "config.json"
    if not config_path.exists():
        print(f"!! no config.json in {directory}")
        return 1
    if repair_config_file(config_path):
        fixed = json.loads(config_path.read_text())
        print(f"repaired {config_path}")
        print(f"  rope_theta   = {fixed['rope_theta']}")
        print(f"  rope_scaling = {fixed['rope_scaling']}")
        print("  rope_parameters kept, so transformers 5.x is unaffected")
    else:
        print(f"{config_path} already reads correctly under transformers 4.x")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", type=Path,
                    help="local checkpoint directory to repair in place")
    ap.add_argument("--survey", action="store_true",
                    help="read-only: report affected published checkpoints")
    ap.add_argument("--repo", action="append", default=None,
                    help="override the --survey model-repo list (repeatable)")
    ap.add_argument("--no-bucket", action="store_true",
                    help="skip the bucket sweep during --survey")
    a = ap.parse_args()

    if a.survey:
        return survey(a.repo or SURVEY_REPOS,
                      bucket=None if a.no_bucket else SURVEY_BUCKET)
    if a.directory is None:
        ap.error("pass a checkpoint directory, or --survey")
    return repair_local(a.directory)


if __name__ == "__main__":
    raise SystemExit(main())
