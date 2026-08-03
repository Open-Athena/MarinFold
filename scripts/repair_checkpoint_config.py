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
    nothing, anywhere.

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

# Repos swept by --survey. Bucket-hosted checkpoints exported by transformers
# 4.x (exp75 via exp89, exp120) read correctly and are not listed.
SURVEY_REPOS = (
    "open-athena/marinfold-exp117",
    "open-athena/marinfold-exp146",
    "open-athena/marinfold-exp75",
)


def survey(repos) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    affected = clean = 0
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
            raw = json.loads(Path(hf_hub_download(repo, path)).read_text())
            if not needs_rope_repair(raw):
                print(f"  ok       {path}  (rope_theta={raw.get('rope_theta')})")
                clean += 1
                continue
            fixed = repair_rope(raw)
            affected += 1
            print(f"  AFFECTED {path}")
            print(f"             exported by transformers {raw.get('transformers_version')}")
            print(f"             4.x reads the architecture default; the trained "
                  f"value is rope_theta={fixed['rope_theta']}")
    print(f"\n{affected} affected, {clean} correct")
    if affected:
        print("Fix: download the checkpoint, run this script on the local copy, "
              "then republish under open-athena/MarinFold with a PROVENANCE.md.")
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
                    help="override the --survey repo list (repeatable)")
    a = ap.parse_args()

    if a.survey:
        return survey(a.repo or SURVEY_REPOS)
    if a.directory is None:
        ap.error("pass a checkpoint directory, or --survey")
    return repair_local(a.directory)


if __name__ == "__main__":
    raise SystemExit(main())
