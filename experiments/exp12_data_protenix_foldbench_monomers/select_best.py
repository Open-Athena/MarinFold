# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per (protein, mode), pick the top-1 sample by Protenix's ``ranking_score``.

Protenix's dumper sorts samples within each seed by ``ranking_score``
(so ``_sample_0.cif`` is the best of that seed's 8 diffusion samples).
To pick the absolute best across all 5 seeds × 8 samples we still need
to compare ranking_scores *across* seeds — we read every
``..._summary_confidence_sample_*.json`` and pick the global top-1.
Its seed determines which distogram we keep.

The "best" tree lays out as::

    best/
    └── {mode}/
        └── {stem}/
            ├── structure.cif
            ├── confidence.json
            ├── distogram.npz
            └── provenance.json     # which seed + sample idx + ranking_score we picked

So a downstream consumer can iterate ``best/{mode}/*/`` without thinking
about Protenix's internal naming.
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BestPick:
    mode: str
    stem: str
    seed: int
    sample_idx: int
    ranking_score: float
    cif: Path
    confidence_json: Path


def _iter_seed_dirs(run_root: Path, mode: str, stem: str) -> Iterable[Path]:
    """Yield ``outputs/{mode}/{stem}/seed_*/`` dirs in seed order."""
    parent = run_root / mode / stem
    if not parent.exists():
        return
    for seed_dir in sorted(parent.glob("seed_*")):
        if seed_dir.is_dir():
            yield seed_dir


def _find_best_in_seed_dir(seed_dir: Path, stem: str) -> tuple[int, float, Path, Path]:
    """Scan ``..._summary_confidence_sample_*.json`` in one seed dir, return
    ``(sample_idx, ranking_score, cif, confidence_json)`` for the highest-scoring sample.

    Raises if no confidence JSONs are found (= the run failed for this seed).
    """
    candidates: list[tuple[int, float, Path, Path]] = []
    prefix = f"{stem}_summary_confidence_sample_"
    for conf_path in sorted(seed_dir.glob(f"{prefix}*.json")):
        idx_part = conf_path.stem[len(prefix):]
        if not idx_part.isdigit():
            continue
        idx = int(idx_part)
        cif_path = seed_dir / f"{stem}_sample_{idx}.cif"
        if not cif_path.exists():
            continue
        data = json.loads(conf_path.read_text())
        score = float(data.get("ranking_score", float("nan")))
        candidates.append((idx, score, cif_path, conf_path))
    if not candidates:
        raise FileNotFoundError(f"No summary_confidence JSONs in {seed_dir}")
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[0]


def select_best(*, runs_dir: Path, out_dir: Path, modes: list[str], stems: list[str]) -> list[BestPick]:
    """Pick best sample per (protein, mode); copy into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    picks: list[BestPick] = []
    for mode in modes:
        for stem in stems:
            best: BestPick | None = None
            for seed_dir in _iter_seed_dirs(runs_dir, mode, stem):
                seed = int(seed_dir.name.removeprefix("seed_"))
                try:
                    idx, score, cif, conf = _find_best_in_seed_dir(seed_dir, stem)
                except FileNotFoundError as e:
                    print(f"WARN: {e}; skipping seed.")
                    continue
                if best is None or score > best.ranking_score:
                    best = BestPick(
                        mode=mode, stem=stem, seed=seed,
                        sample_idx=idx, ranking_score=score,
                        cif=cif, confidence_json=conf,
                    )
            if best is None:
                print(f"WARN: no usable samples for {mode}/{stem}; skipping.")
                continue

            distogram_npz = runs_dir / mode / stem / f"seed_{best.seed}" / f"{stem}_distogram.npz"
            if not distogram_npz.exists():
                raise FileNotFoundError(
                    f"Top-ranked sample for {mode}/{stem} is seed {best.seed}, "
                    "but its distogram is missing. Refusing to silently "
                    "downgrade to a lower-ranked seed."
                )

            dst_dir = out_dir / mode / stem
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best.cif, dst_dir / "structure.cif")
            shutil.copy2(best.confidence_json, dst_dir / "confidence.json")
            shutil.copy2(distogram_npz, dst_dir / "distogram.npz")
            (dst_dir / "provenance.json").write_text(json.dumps({
                "stem": best.stem,
                "mode": best.mode,
                "seed": best.seed,
                "sample_idx": best.sample_idx,
                "ranking_score": best.ranking_score,
                "src_cif": str(best.cif),
                "src_confidence_json": str(best.confidence_json),
                "src_distogram_npz": str(distogram_npz),
            }, indent=2))
            picks.append(best)
            print(f"selected {mode}/{stem}: seed={best.seed} sample={best.sample_idx} ranking_score={best.ranking_score:.4f}")
    return picks


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("select-best", help="Pick top-1 sample per (protein, mode) by ranking_score.")
    p.add_argument("--runs", type=Path, required=True, help="Root of synced Modal outputs (contains {mode}/{stem}/seed_*/)")
    p.add_argument("--out", type=Path, required=True, help="Output dir for the clean best/ tree.")
    p.add_argument("--modes", default="single_seq,msa", help="Comma-separated mode names (default: single_seq,msa).")
    p.add_argument("--manifest", type=Path, required=True, help="Path to inputs/manifest.csv (for stem list).")
    p.set_defaults(func=lambda args: select_best(
        runs_dir=args.runs,
        out_dir=args.out,
        modes=args.modes.split(","),
        stems=_read_stems(args.manifest),
    ))


def _read_stems(manifest_csv: Path) -> list[str]:
    import csv
    with manifest_csv.open() as f:
        return [row["stem"] for row in csv.DictReader(f)]
