# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-protein ``provenance.json`` files into ``data/timings.csv``.

Every protein run by ``run_1b_eval.py`` (local) or ``modal_app.py``
(Modal) writes a ``provenance.json`` next to its ``distogram.npz``
with ``elapsed_seconds``, ``n_residues``, ``n_pairs``, ``batch_size``,
and a ``hardware`` block (``gpu_name``, ``gpu_total_memory_gb``,
``runner_tag``, ``hostname``, ``platform``, ``torch_version``).

This script walks all the ``provenance.json`` files under
``outputs/`` and emits a single flat CSV — one row per (protein,
run). The CSV is what ``plot_comparison.py:plot_timing`` reads to
build the sequence-length-vs-runtime plot, grouped by GPU.

This way, the same artifact captures both local A5000 timings and
Modal H100/A100 timings — no special-casing per backend.
"""

import argparse
import csv
import json
from pathlib import Path


_FIELDS = (
    "stem",
    "n_residues",
    "n_pairs",
    "elapsed_seconds",
    "model_load_seconds",
    "batch_size",
    "model_nickname",
    "gpu_name",
    "gpu_total_memory_gb",
    "gpu_compute_capability",
    "runner_tag",
    "hostname",
    "platform",
    "torch_version",
    "timestamp_utc",
    "provenance_path",
)


def _flatten(provenance: dict, *, provenance_path: Path) -> dict:
    hw = provenance.get("hardware") or {}
    return {
        "stem": provenance.get("stem", ""),
        "n_residues": provenance.get("n_residues", ""),
        "n_pairs": provenance.get("n_pairs", ""),
        "elapsed_seconds": provenance.get("elapsed_seconds", ""),
        "model_load_seconds": provenance.get("model_load_seconds", ""),
        "batch_size": provenance.get("batch_size", ""),
        "model_nickname": provenance.get("model_nickname", ""),
        "gpu_name": hw.get("gpu_name", ""),
        "gpu_total_memory_gb": hw.get("gpu_total_memory_gb", ""),
        "gpu_compute_capability": hw.get("gpu_compute_capability", ""),
        "runner_tag": hw.get("runner_tag", ""),
        "hostname": hw.get("hostname", ""),
        "platform": hw.get("platform", ""),
        "torch_version": hw.get("torch_version", ""),
        "timestamp_utc": provenance.get("timestamp_utc", ""),
        "provenance_path": str(provenance_path),
    }


def collect(outputs_dir: Path, out_csv: Path) -> int:
    """Walk ``outputs_dir`` for ``*/provenance.json`` and write the CSV."""
    rows: list[dict] = []
    for prov_path in sorted(outputs_dir.glob("*/provenance.json")):
        try:
            data = json.loads(prov_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"skip {prov_path}: failed to parse ({exc!r})")
            continue
        rows.append(_flatten(data, provenance_path=prov_path))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return len(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs", type=Path, default=here / "outputs",
        help="Dir containing <stem>/provenance.json (default: ./outputs/).",
    )
    parser.add_argument(
        "--out", type=Path, default=here / "data" / "timings.csv",
    )
    args = parser.parse_args()
    collect(outputs_dir=args.outputs, out_csv=args.out)


if __name__ == "__main__":
    main()
