# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6c — assemble the 577-unit GT universe and publish it to the HF bucket.

#89's `gt_universe.jsonl` covers 554 proteins; :mod:`build_gt_contacts` adds the
23 that eval2 needs. Concatenating them gives the universe an eval2 run scores
against. The merged file is ~8 MB — too big for git, and exactly what the root
`AGENTS.md` says belongs in the public bucket — so the small 23-record delta is
committed here and the merged file is published.

Published alongside it: `eval2_manifest.csv` and `eval2.fasta`, so a downstream
eval needs one prefix and no access to this checkout.

Nothing is overwritten under #89's prefix; this writes its own.

    uv run python publish_gt_to_hf.py --merge-only    # build locally, don't upload
    uv run python publish_gt_to_hf.py                 # build + upload
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

BUCKET_PREFIX = "hf://buckets/open-athena/MarinFold/data/contacts-v1-eval2-exp226"

#: The published 554-unit universe from #89. Fetch with:
#: hf buckets cp hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/gt_universe.jsonl <path>
EXP89_UNIVERSE = Path("/data/exp226_gt/gt_universe_554.jsonl")


def find_hf_cli() -> str:
    """An `hf` binary whose huggingface_hub is new enough to know `buckets`.

    Bucket repos need huggingface_hub>=1.5, but this experiment's `gt` extra
    pulls marinfold -> transformers<5 -> huggingface_hub<1, so the venv's `hf`
    (when it has one at all) cannot do bucket I/O. `uv run` puts the venv first
    on PATH, so resolving `hf` naively finds the wrong one. Look outside the
    venv instead; `HF_CLI` overrides.
    """
    override = os.environ.get("HF_CLI")
    if override:
        return override
    venv = os.environ.get("VIRTUAL_ENV")
    path_entries = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not (venv and Path(entry).parent == Path(venv))
    ]
    found = shutil.which("hf", path=os.pathsep.join(path_entries))
    if not found:
        raise SystemExit(
            "no `hf` CLI outside the venv. Install huggingface_hub>=1.5 somewhere "
            "on PATH, or set HF_CLI to its `hf` binary."
        )
    return found


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge(exp89: Path, new: Path, out: Path) -> dict:
    """Concatenate the two universes, refusing to merge overlapping stems."""
    published, added = read_jsonl(exp89), read_jsonl(new)
    overlap = {r["stem"] for r in published} & {r["stem"] for r in added}
    if overlap:
        raise SystemExit(
            f"{len(overlap)} stems appear in both universes ({sorted(overlap)[:5]}); "
            "merging would double-count them."
        )
    keys = {tuple(sorted(r)) for r in published} | {tuple(sorted(r)) for r in added}
    if len(keys) != 1:
        raise SystemExit(f"record schemas differ between the two universes: {keys}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for record in published + added:
            fh.write(json.dumps(record) + "\n")
    return {
        "n_published": len(published), "n_added": len(added),
        "n_total": len(published) + len(added),
        "n_unique_stems": len({r["stem"] for r in published + added}),
        "path": str(out), "bytes": out.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp89-universe", type=Path, default=EXP89_UNIVERSE)
    ap.add_argument("--new", type=Path, default=DATA / "gt_universe_eval2_new.jsonl")
    ap.add_argument("--merged", type=Path,
                    default=Path("/data/exp226_gt/gt_universe_eval2_577.jsonl"))
    ap.add_argument("--prefix", default=BUCKET_PREFIX)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if not args.exp89_universe.exists():
        raise SystemExit(
            f"{args.exp89_universe} not found. Fetch #89's universe first:\n"
            "  hf buckets cp hf://buckets/open-athena/MarinFold/data/"
            f"contacts-v1-model-eval-exp89/gt_universe.jsonl {args.exp89_universe}"
        )
    stats = merge(args.exp89_universe, args.new, args.merged)
    print(f"[merge] {stats['n_published']} + {stats['n_added']} = {stats['n_total']} "
          f"units, {stats['n_unique_stems']} unique stems, "
          f"{stats['bytes'] / 1e6:.1f} MB -> {args.merged}", flush=True)

    uploads = [(args.merged, "gt_universe_eval2.jsonl"),
               (args.new, "gt_universe_eval2_new_23.jsonl"),
               (DATA / "eval2_manifest.csv", "eval2_manifest.csv"),
               (DATA / "eval2.fasta", "eval2.fasta")]
    if args.merge_only:
        print("[publish] --merge-only; skipping upload of "
              f"{[name for _, name in uploads]}", flush=True)
        return 0
    hf = find_hf_cli()
    print(f"[publish] using {hf}", flush=True)
    for path, name in uploads:
        target = f"{args.prefix}/{name}"
        subprocess.run([hf, "buckets", "cp", str(path), target], check=True)
        print(f"[publish] {path.name} -> {target}", flush=True)
    (DATA / "publish_provenance.json").write_text(json.dumps({
        "prefix": args.prefix, "merge": stats,
        "files": [name for _, name in uploads],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
