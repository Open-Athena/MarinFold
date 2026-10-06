# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute ColabFold/Protenix MSA depth for the exp324 PDB-deduped 10k set.

The source contacts-v1 PDB-deduped corpus and annotation caches do not carry MSA
statistics, so this app runs the same Protenix ColabFold MSA search used by the
FoldBench Protenix evaluations, persists the A3Ms in a Modal volume, and then
measures raw depth plus Neff from the unpaired A3M.

Examples:

    uv run --with modal modal run msa_depth_modal.py::precompute --limit 5
    uv run --with modal modal run msa_depth_modal.py::measure_depths --limit 5
    uv run --with modal modal run msa_depth_modal.py::precompute
    uv run --with modal modal run msa_depth_modal.py::measure_depths
"""

import json
from pathlib import Path

import modal
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent
TARGETS = EXPERIMENT / "data" / "sample_10000_targets.parquet"
OUT = EXPERIMENT / "data" / "sample_10000_msa_depth.csv"
STATUS_OUT = EXPERIMENT / "data" / "sample_10000_msa_precompute_status.csv"

APP_NAME = "exp324-pdb10k-msa-depth"
MSA_VOLUME_NAME = "exp324-pdb10k-colabfold-msa"
MSA_VOL = modal.Volume.from_name(MSA_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "wget", "curl")
    .pip_install("protenix", "numpy>=2,<3", "pandas>=2.2,<3")
    .env(
        {
            "MMSEQS_SERVICE_HOST_URL": "https://api.colabfold.com",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file(EXPERIMENT.parent / "exp12_data_protenix_foldbench_monomers" / "msa_depth.py", "/root/msa_depth.py", copy=True)
)

app = modal.App(APP_NAME, image=image)


def _load_specs(limit: int = 0, offset: int = 0) -> list[dict[str, str]]:
    targets = pd.read_parquet(TARGETS)
    frame = targets[["stem", "input_seq"]].rename(columns={"input_seq": "sequence"})
    frame = frame.sort_values("stem", ignore_index=True)
    if offset:
        frame = frame.iloc[offset:]
    if limit:
        frame = frame.iloc[:limit]
    return frame.to_dict("records")


@app.function(volumes={"/msa": MSA_VOL}, timeout=60 * 45, cpu=2.0, memory=8192, retries=2, max_containers=40)
def precompute_one(spec: dict[str, str]) -> dict:
    """Run Protenix/ColabFold MSA search for one protein, idempotently."""

    from pathlib import Path
    import time

    from runner.msa_search import update_seq_msa

    stem = spec["stem"]
    sequence = spec["sequence"]
    target_dir = Path("/msa") / stem / "msa"
    non_paired = target_dir / "0" / "0" / "non_pairing.a3m"
    paired = target_dir / "0" / "pairing.a3m"
    if non_paired.exists():
        return {
            "stem": stem,
            "skipped": True,
            "failed": False,
            "paired_exists": paired.exists(),
            "non_paired_exists": True,
            "elapsed_seconds": 0.0,
            "error": "",
        }

    infer_data = {
        "name": stem,
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
        "covalent_bonds": [],
    }
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        update_seq_msa(infer_data, str(target_dir), mode="colabfold")
        failed = False
        error = ""
    except Exception as exc:  # noqa: BLE001 - public ColabFold API can fail transiently.
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        print(f"WARN [{stem}] {error}", flush=True)
    elapsed = time.monotonic() - started
    MSA_VOL.commit()
    return {
        "stem": stem,
        "skipped": False,
        "failed": failed,
        "paired_exists": paired.exists(),
        "non_paired_exists": non_paired.exists(),
        "elapsed_seconds": elapsed,
        "error": error,
    }


@app.function(volumes={"/msa": MSA_VOL}, timeout=60 * 60, cpu=8.0, memory=16384, max_containers=80)
def measure_one(stem: str) -> dict:
    """Measure raw MSA depth and Neff for one persisted A3M."""

    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root")
    import msa_depth as md

    path = Path("/msa") / stem / "msa" / "0" / "0" / "non_pairing.a3m"
    row = {"stem": stem, "found": path.exists()}
    if not row["found"]:
        return row
    started = time.monotonic()
    text = path.read_text()
    depth = md.msa_depth(text)
    row.update(
        {
            "a3m_bytes": len(text),
            "n_seqs": depth.n_seqs,
            "query_len": depth.query_len,
            **{f"n_eff_{threshold}": value for threshold, value in depth.n_eff.items()},
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return row


@app.local_entrypoint()
def precompute(limit: int = 0, offset: int = 0) -> None:
    """Launch/idempotently resume ColabFold MSA search for the 10k set."""

    specs = _load_specs(limit=limit, offset=offset)
    print(f"precomputing {len(specs)} MSAs from {TARGETS} offset={offset} limit={limit}")
    rows = list(precompute_one.map(specs))
    frame = pd.DataFrame(rows).sort_values("stem", ignore_index=True)
    destination = STATUS_OUT if not limit and not offset else STATUS_OUT.with_name(f"sample_10000_msa_precompute_status.offset{offset}.limit{limit}.csv")
    frame.to_csv(destination, index=False)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "failed": int(frame.failed.sum()) if "failed" in frame else None,
                "non_paired_exists": int(frame.non_paired_exists.sum()) if "non_paired_exists" in frame else None,
                "out": str(destination),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def measure_depths(limit: int = 0, offset: int = 0) -> None:
    """Measure depths for persisted MSAs."""

    specs = _load_specs(limit=limit, offset=offset)
    stems = [spec["stem"] for spec in specs]
    rows = list(measure_one.map(stems))
    frame = pd.DataFrame(rows).sort_values("stem", ignore_index=True)
    destination = OUT if not limit and not offset else OUT.with_name(f"sample_10000_msa_depth.offset{offset}.limit{limit}.csv")
    frame.to_csv(destination, index=False)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "found": int(frame.found.sum()) if "found" in frame else None,
                "median_n_seqs": float(frame.n_seqs.median()) if "n_seqs" in frame else None,
                "median_n_eff_0.8": float(frame["n_eff_0.8"].median()) if "n_eff_0.8" in frame else None,
                "out": str(destination),
            },
            indent=2,
        )
    )
