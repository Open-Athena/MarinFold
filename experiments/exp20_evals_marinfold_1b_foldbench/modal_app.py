# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Modal app: run MarinFold 1B on the FoldBench monomers.

Mirrors the local ``run_1b_eval.py`` but on Modal's hardware
(H100 / A100), so we can fan out across proteins in parallel and
finish the 100-protein run in well under an hour.

Volumes:

- ``marinfold-1b-weights`` — local cache of the 1B HF snapshot, so
  containers don't re-download on every cold start.
- ``marinfold-1b-foldbench-runs`` — output volume; one
  ``{stem}/{distogram.npz, provenance.json}`` per protein.

The Protenix GT mmCIFs are not stored on Modal — they're committed
via ``modal.Mount.from_local_dir`` from the local ``protenix_data/``
mirror (~26 MB, well under the inline limit). Run
``fetch_protenix_data.py`` first.

Entry points (called by the included CLI or via
``modal run modal_app.py::<fn> ...``):

- ``setup_weights()`` — one-time snapshot of the 1B model into the
  weights volume.
- ``predict_one(stem)`` — inference for one protein. Idempotent:
  skip if ``/outputs/{stem}/distogram.npz`` already exists with the
  expected shape.
- ``main`` (CLI) — fan out across the manifest.
"""

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "marinfold-1b-foldbench-monomers-exp20"

WEIGHTS_VOLUME_NAME = "marinfold-1b-weights"
OUTPUTS_VOLUME_NAME = "marinfold-1b-foldbench-runs"

WEIGHTS_VOL = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)


# We bundle the protenix_data dir (GT mmCIFs + manifest) into the
# container at /protenix_data so workers can read GT structures
# without round-tripping to the bucket. The local fetch via
# ``fetch_protenix_data.py`` must have run first.
_HERE = Path(__file__).resolve().parent
_PROTENIX_DATA = _HERE / "protenix_data"
_EXP1_DIR = _HERE.parent / "exp1_document_structures_contacts_and_distances_v1"


MARINFOLD_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "wget", "curl")
    .pip_install(
        # Pins mirror the local exp20/exp9 venv. vllm 0.7.x is what
        # runs against the 1B weights; transformers 4.x is its hard
        # cap; gemmi + numpy are for GT-side parsing.
        "vllm==0.7.3",
        "torch==2.5.1",
        "transformers>=4.45,<5",
        "huggingface_hub[hf_transfer]>=0.24,<1",
        "gemmi>=0.6",
        "numpy",
        "pyyaml>=6",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MARINFOLD_RUNNER_TAG": "modal",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # The pure-python helpers we wrote locally need to ship with the image.
    .add_local_file(str(_HERE / "canonical_sequence.py"), remote_path="/exp20/canonical_sequence.py")
    .add_local_file(str(_HERE / "run_1b_eval.py"), remote_path="/exp20/run_1b_eval.py")
    .add_local_file(str(_EXP1_DIR / "vocab.py"), remote_path="/exp20/vocab.py")
    # Manifest + GT CIFs ship with the image (read-only at runtime).
    .add_local_dir(str(_PROTENIX_DATA), remote_path="/protenix_data")
)


app = modal.App(APP_NAME, image=MARINFOLD_IMAGE)


# --------------------------------------------------------------------------
# Weights bootstrap (one-time)
# --------------------------------------------------------------------------


@app.function(volumes={"/weights": WEIGHTS_VOL}, timeout=60 * 30)
def setup_weights(repo_id: str, subdir: str) -> dict:
    """Pull the 1B snapshot into the weights Volume.

    Idempotent: re-running does nothing if the files already exist.
    """
    from huggingface_hub import snapshot_download
    target = f"/weights/{subdir}"
    Path("/weights").mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subdir}/*"],
        local_dir="/weights",
    )
    WEIGHTS_VOL.commit()
    files = sorted(p.relative_to("/weights").as_posix() for p in Path(target).rglob("*") if p.is_file())
    print(f"weights laid out at {target} ({len(files)} files)")
    return {"target": target, "n_files": len(files)}


# --------------------------------------------------------------------------
# Inference worker
# --------------------------------------------------------------------------


@app.cls(
    volumes={"/weights": WEIGHTS_VOL, "/outputs": OUTPUTS_VOL},
    gpu="H100",
    timeout=60 * 60 * 2,
)
@modal.concurrent(max_inputs=1)
class MarinFoldWorker:
    """Long-lived inference worker. vLLM stays resident across calls."""

    repo_id: str = modal.parameter()
    subdir: str = modal.parameter()

    @modal.enter()
    def setup(self) -> None:
        # Make our helper modules importable. `/exp20/` is the path
        # we mounted them to in the image build; prepend so they
        # shadow any homonyms.
        if "/exp20" not in sys.path:
            sys.path.insert(0, "/exp20")
        # vLLM helpers — load the model into GPU memory now.
        from run_1b_eval import _load_vllm, _resolve_distance_token_ids, _hardware_info
        model_path = f"/weights/{self.subdir}"
        t0 = time.time()
        self.llm, self.tokenizer = _load_vllm(model_path)
        self.distance_token_ids = _resolve_distance_token_ids(self.tokenizer)
        self.model_path = model_path
        self.model_load_seconds = round(time.time() - t0, 3)
        self.hw = _hardware_info()
        print(
            f"vLLM ready ({self.model_load_seconds:.1f} s). "
            f"GPU={self.hw.get('gpu_name')} ({self.hw.get('gpu_total_memory_gb')} GB)."
        )

    @modal.method()
    def predict_one(
        self,
        stem: str,
        *,
        batch_size: int = 128,
        force: bool = False,
        model_nickname: str = "1B",
    ) -> dict:
        """Predict the [N, N, 64] distogram for one protein.

        Idempotent unless ``force=True``: an existing valid output
        on /outputs is left alone and reported back.
        """
        import numpy as np
        from canonical_sequence import read_canonical_sequence
        from run_1b_eval import (
            _NUM_DISTANCE_BINS, _DISTANCE_MAX_A, _BIN_MIDPOINTS,
            _predict_distogram,
        )

        out_dir = Path(f"/outputs/{stem}")
        out_path = out_dir / "distogram.npz"
        gt_cif = Path(f"/protenix_data/data/protenix-foldbench-monomers/gt/{stem}.cif")
        if not gt_cif.exists():
            return {"stem": stem, "status": "missing_gt"}
        seq = read_canonical_sequence(gt_cif)

        if not force and out_path.exists():
            try:
                with np.load(out_path) as data:
                    if (
                        "probs" in data.files
                        and data["probs"].shape == (seq.n_residues, seq.n_residues, _NUM_DISTANCE_BINS)
                    ):
                        return {"stem": stem, "status": "skipped", "n_residues": seq.n_residues}
            except (OSError, ValueError):
                pass  # corrupt; recompute

        t0 = time.time()
        probs = _predict_distogram(
            llm=self.llm, tokenizer=self.tokenizer,
            residue_names=seq.residue_names,
            distance_token_ids=self.distance_token_ids,
            batch_size=batch_size,
        )
        elapsed = time.time() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, probs=probs)
        n_pairs = seq.n_residues * (seq.n_residues - 1) // 2
        (out_dir / "provenance.json").write_text(json.dumps({
            "stem": stem,
            "n_residues": seq.n_residues,
            "n_pairs": n_pairs,
            "model_nickname": model_nickname,
            "model_path": self.model_path,
            "atom_convention": "CB-CB (CA for GLY/UNK)",
            "bin_scheme": {
                "min_A": 0.0,
                "max_A": _DISTANCE_MAX_A,
                "n_bins": _NUM_DISTANCE_BINS,
                "midpoints_A": _BIN_MIDPOINTS.tolist(),
            },
            "elapsed_seconds": round(elapsed, 3),
            "model_load_seconds": self.model_load_seconds,
            "batch_size": batch_size,
            "hardware": self.hw,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2) + "\n")
        OUTPUTS_VOL.commit()
        return {
            "stem": stem, "status": "ok",
            "n_residues": seq.n_residues, "n_pairs": n_pairs,
            "elapsed_seconds": round(elapsed, 2),
        }


# --------------------------------------------------------------------------
# Local entry point — fan out across proteins, then rsync results back.
# --------------------------------------------------------------------------


def _read_manifest(local_protenix_dir: Path) -> list[dict[str, Any]]:
    with (local_protenix_dir / "manifest.csv").open() as f:
        return list(csv.DictReader(f))


def _hf_url_to_repo_subdir(models_yaml: Path, nickname: str) -> tuple[str, str]:
    """Parse MODELS.yaml → (repo_id, subdir) for the named model."""
    import yaml
    entries = yaml.safe_load(models_yaml.read_text())
    matched = [e for e in entries if e.get("nickname") == nickname]
    if not matched:
        raise ValueError(f"no entry for {nickname!r} in {models_yaml}")
    url = matched[0]["url"]
    prefix = "https://huggingface.co/"
    if not url.startswith(prefix):
        raise ValueError(f"unexpected model URL {url!r}")
    rest = url[len(prefix):].split("/")
    repo_id = "/".join(rest[:2])
    subdir = rest[4] if len(rest) > 4 and rest[2] == "tree" else ""
    return repo_id, subdir


@app.local_entrypoint()
def main(
    limit: int = -1,
    batch_size: int = 128,
    force: bool = False,
    download_back: bool = True,
    model_nickname: str = "1B",
    models_yaml: str = "",
):
    """Fan out across proteins, optionally rsync results back.

    Usage::

        # Smoke: 3 proteins
        modal run modal_app.py --limit 3

        # Full 100-protein run
        modal run modal_app.py

        # Re-download outputs to ./outputs/ at the end
        modal run modal_app.py --download-back
    """
    repo_root = _HERE.parent.parent
    models_yaml_path = Path(models_yaml) if models_yaml else (repo_root / "MODELS.yaml")
    repo_id, subdir = _hf_url_to_repo_subdir(models_yaml_path, model_nickname)
    print(f"model: nickname={model_nickname} repo={repo_id} subdir={subdir!r}")

    # Bootstrap weights (idempotent).
    print("setting up weights ...")
    setup_weights.remote(repo_id, subdir)

    # Build the worklist.
    manifest = _read_manifest(_PROTENIX_DATA / "data" / "protenix-foldbench-monomers")
    if limit > 0:
        manifest = manifest[:limit]
    stems = [m["stem"] for m in manifest]
    print(f"dispatching {len(stems)} proteins ...")

    worker = MarinFoldWorker(repo_id=repo_id, subdir=subdir)
    results: list[dict] = []
    for res in worker.predict_one.map(
        stems,
        kwargs={"batch_size": batch_size, "force": force, "model_nickname": model_nickname},
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(res, Exception):
            print(f"  FAIL: {res!r}")
            continue
        results.append(res)
        print(
            f"  {res.get('status')} {res.get('stem')}: "
            f"n_res={res.get('n_residues')} "
            f"elapsed={res.get('elapsed_seconds')}"
        )

    if download_back:
        local_out = _HERE / "outputs"
        local_out.mkdir(parents=True, exist_ok=True)
        print(f"downloading outputs volume → {local_out} ...")
        OUTPUTS_VOL.batch_download(local_out)

    print(f"done. {len([r for r in results if r.get('status') == 'ok'])} new, "
          f"{len([r for r in results if r.get('status') == 'skipped'])} skipped, "
          f"{len(results)} total.")
