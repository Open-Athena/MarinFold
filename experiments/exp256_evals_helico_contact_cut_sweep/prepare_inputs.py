# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract eval-val-only inputs from the two original Helico worktrees."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from analyze import CUT_ORDER, EVAL_SET, REFERENCE_ARMS

EXPERIMENT = Path("experiments/exp14_foldbench_held_out_monomers")
NEW_ARMS = {"mf_1p5L", "mf_2L", "mf_3L", "mf_5L", "mf_union"}


def prepare(helico: Path, exp14: Path, out: Path) -> None:
    """Save only eval-val records, retaining hashes of every source file."""
    roots = {"exp256": helico, "exp14": exp14}
    provenance = {
        "eval_set": EVAL_SET,
        "repositories": {
            name: {"repository": "Open-Athena/helico", "commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()}
            for name, root in roots.items()
        },
        "files": {},
    }
    out.mkdir(parents=True, exist_ok=True)

    def record(source: Path, dest: Path, run: str) -> None:
        provenance["files"][str(dest.relative_to(out))] = {
            "run": run, "source": str(source.relative_to(roots[run])),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }

    targets_path = helico / EXPERIMENT / "data/targets.csv"
    targets = pd.read_csv(targets_path)
    targets = targets[targets.eval_set.eq(EVAL_SET)]
    targets.to_csv(out / "targets.csv", index=False)
    record(targets_path, out / "targets.csv", "exp256")
    ids = set(targets.target_id)
    for name in ("token_map.json",):
        source = helico / EXPERIMENT / "data" / name
        obj = json.loads(source.read_text())
        (out / name).write_text(json.dumps({k: v for k, v in obj.items() if k in ids}) + "\n")
        record(source, out / name, "exp256")
    source = helico / EXPERIMENT / ".cache/upstream/gt_universe_scored.jsonl"
    with (out / "gt_universe_scored.jsonl").open("w") as handle:
        for line in source.read_text().splitlines():
            if json.loads(line)["stem"] in ids:
                handle.write(line + "\n")
    record(source, out / "gt_universe_scored.jsonl", "exp256")
    for stem in sorted(ids):
        for source_dir, dest_dir, suffix in (("data/gt", "gt", ".cif.gz"),
                                             (".cache/marinfold_dense", "dense", ".npz")):
            name = f"foldbench_monomer__{stem}{suffix}" if dest_dir == "dense" else f"{stem}{suffix}"
            source = helico / EXPERIMENT / source_dir / name
            dest = out / dest_dir / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(source.read_bytes())
            record(source, dest, "exp256")
    for tag in CUT_ORDER + REFERENCE_ARMS:
        run = "exp256" if tag in NEW_ARMS else "exp14"
        root = roots[run] / EXPERIMENT
        for suffix in (".csv", ".timings.csv", ".manifest.json"):
            source = root / "results" / f"{tag}{suffix}"
            dest = out / "results" / source.name
            dest.parent.mkdir(exist_ok=True)
            if suffix == ".manifest.json":
                # The manifest describes the original full run, before this
                # extraction; its counts therefore include other eval sets.
                dest.write_bytes(source.read_bytes())
            else:
                frame = pd.read_csv(source)
                frame = frame[frame.target_id.isin(ids)]
                frame.to_csv(dest, index=False)
            record(source, dest, run)
        results = pd.read_csv(out / "results" / f"{tag}.csv")
        for stem in results.loc[results.status.eq("ok"), "target_id"]:
            source = root / "results/predictions" / tag / f"{stem}.pdb.gz"
            dest = out / "predictions" / tag / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
            record(source, dest, run)
        if tag in CUT_ORDER:
            source = root / "data/arms" / f"{tag}.json"
            dest = out / "arms" / source.name
            dest.parent.mkdir(exist_ok=True)
            obj = json.loads(source.read_text())
            dest.write_text(json.dumps({k: v for k, v in obj.items() if k in ids}) + "\n")
            record(source, dest, run)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    """Build a portable input directory for analyze.py and public upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helico", type=Path, required=True)
    parser.add_argument("--exp14", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.helico, args.exp14, args.out)


if __name__ == "__main__":
    main()
