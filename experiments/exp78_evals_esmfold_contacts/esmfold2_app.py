# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Modal app: run ESMFold2 (``biohub/ESMFold2``) on the exp78 eval set.

ESMFold2 (Biohub, released 2026-05-27) is a diffusion-based all-atom
structure predictor built on the ESMC-6B protein language model — the
successor to ESMFold and a single-sequence SOTA folder. We run it in
**single-sequence mode** (no MSA), drawing several diffusion samples per
protein and keeping the **top-1 by the model's confidence**, mirroring
exp74's Protenix top-1-of-(5 seeds × 8 samples) selection so the
comparison is apples-to-apples. The chosen structure is persisted as
mmCIF for pyconfind contact scoring.

Two Volumes:
- ``esmfold2-weights`` — the HF ``biohub/ESMFold2`` snapshot (ESMC-6B is
  large; snapshot once).
- ``esmfold2-exp78-runs`` — per-protein outputs (``{stem}/structure.cif``,
  ``{stem}/provenance.json``, ``{stem}/timings.json``).

Recommended sampling settings (logged per run): ``num_loops=20``,
``num_sampling_steps=100`` (the documented defaults), and
``n_samples`` diffusion draws (distinct seeds) per protein.

NOTE: the exact ESMFold2 Python API (confidence attribute names, sample
fan-out) was pinned against the installed ``biohub/esm`` package during a
feasibility spike (``spike_esmfold2.py``); see ``_score_confidence`` for
the resolved accessor and its fallbacks.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import modal

APP_NAME = "esmfold2-contacts-eval-exp78"

WEIGHTS_VOLUME_NAME = "esmfold2-weights"
OUTPUTS_VOLUME_NAME = "esmfold2-exp78-runs"

WEIGHTS_VOL = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)

HF_MODEL_ID = "biohub/ESMFold2"

ESMFOLD2_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        # Biohub's `esm` package provides esm.models.esmfold2 (input builder)
        # and registers the transformers ESMFold2Model. Pinned-ish; resolved
        # by the spike.
        "esm",
        "transformers>=4.40",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "gemmi>=0.6",
        "numpy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/weights/hf",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App(APP_NAME, image=ESMFOLD2_IMAGE)


@app.function(volumes={"/weights": WEIGHTS_VOL}, timeout=60 * 60)
def setup_weights() -> dict:
    """Snapshot the ESMFold2 weights from HF into the weights Volume."""
    from huggingface_hub import snapshot_download

    target = snapshot_download(HF_MODEL_ID, cache_dir="/weights/hf")
    WEIGHTS_VOL.commit()
    print(f"snapshot -> {target}")
    return {"model": HF_MODEL_ID, "cache": target}


def _score_confidence(result) -> float:
    """Best-effort scalar confidence for a folded result (higher = better).

    ESMFold2's result object exposes a confidence summary; the exact path
    was confirmed by the spike. We try the known accessors in order and
    fall back to NaN (which makes best-of-N degrade gracefully to "first
    sample" rather than crashing).
    """
    import math

    for attr in ("ptm", "mean_plddt", "plddt", "confidence"):
        val = getattr(result, attr, None)
        if val is None and hasattr(result, "complex"):
            val = getattr(result.complex, attr, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            try:
                import numpy as np
                return float(np.asarray(val, dtype=float).mean())
            except Exception:  # noqa: BLE001
                continue
    return math.nan


@app.cls(
    volumes={"/weights": WEIGHTS_VOL, "/outputs": OUTPUTS_VOL},
    gpu="H100",
    timeout=60 * 60 * 4,
)
@modal.concurrent(max_inputs=1)
class ESMFold2Worker:
    """Long-lived ESMFold2 worker. Weights load once per container."""

    @modal.enter()
    def setup(self) -> None:
        import platform
        import socket

        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        self.model = ESMFold2Model.from_pretrained(HF_MODEL_ID).cuda().eval()

        meta: dict = {"hostname": socket.gethostname(), "platform": platform.platform(),
                      "torch_version": torch.__version__, "model": HF_MODEL_ID}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            meta["gpu_name"] = props.name
            meta["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
        self.worker_meta = meta
        print(f"worker meta: {meta}")

    @modal.method()
    def predict_one(
        self,
        *,
        stem: str,
        sequence: str,
        n_samples: int = 5,
        num_loops: int = 20,
        num_sampling_steps: int = 100,
        force_run: bool = False,
    ) -> dict:
        """Fold one sequence with best-of-N sampling; persist the top-1.

        Draws ``n_samples`` independent diffusion samples (distinct seeds),
        scores each by model confidence, and writes the best as
        ``{stem}/structure.cif`` with a ``provenance.json`` recording which
        seed won and every sample's confidence.
        """
        from esm.models.esmfold2 import (
            ESMFold2InputBuilder,
            ProteinInput,
            StructurePredictionInput,
        )

        out_dir = Path("/outputs") / stem
        OUTPUTS_VOL.reload()
        if (out_dir / "structure.cif").exists() and not force_run:
            print(f"[{stem}] already complete; skipping.")
            return {"stem": stem, "skipped": True, "output_dir": str(out_dir)}

        builder = ESMFold2InputBuilder()
        spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])

        t0 = time.time()
        best = None  # (confidence, seed, mmcif_str)
        sample_scores: list[dict] = []
        for seed in range(n_samples):
            result = builder.fold(
                self.model, spi,
                num_loops=num_loops, num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=1, seed=seed,
            )
            conf = _score_confidence(result)
            mmcif = result.complex.to_mmcif()
            sample_scores.append({"seed": seed, "confidence": conf})
            # NaN-safe: first sample always wins ties / missing confidence.
            if best is None or (conf == conf and conf > best[0]):
                best = (conf, seed, mmcif)
        elapsed = time.time() - t0

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "structure.cif").write_text(best[2])
        (out_dir / "provenance.json").write_text(json.dumps({
            "stem": stem, "chosen_seed": best[1], "chosen_confidence": best[0],
            "n_samples": n_samples, "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps, "sample_scores": sample_scores,
        }, indent=2))

        from datetime import datetime, timezone
        timings = {
            "stem": stem, "n_residues": len(sequence),
            "elapsed_seconds": round(elapsed, 4),
            "n_samples": n_samples, "num_loops": num_loops, "num_sampling_steps": num_sampling_steps,
            "model_nickname": "esmfold2", "runner_tag": "modal",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **self.worker_meta,
        }
        (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))
        OUTPUTS_VOL.commit()
        return {"stem": stem, "skipped": False, "chosen_seed": best[1],
                "chosen_confidence": best[0], "elapsed_seconds": round(elapsed, 4),
                "output_dir": str(out_dir)}


def run_cli(args: argparse.Namespace) -> None:
    """Fan ESMFold2 out across every protein in the eval manifest(s)."""
    rows: list[dict] = []
    for manifest in args.manifest:
        with open(manifest) as f:
            rows.extend(csv.DictReader(f))
    seen: dict[str, str] = {}
    for r in rows:
        seen.setdefault(r["stem"], r["input_seq"])
    print(f"run_cli: {len(seen)} unique proteins "
          f"(n_samples={args.n_samples}, num_loops={args.num_loops}, "
          f"num_sampling_steps={args.num_sampling_steps})")

    with app.run():
        worker = ESMFold2Worker()
        futures = {
            stem: worker.predict_one.spawn(
                stem=stem, sequence=seq, n_samples=args.n_samples,
                num_loops=args.num_loops, num_sampling_steps=args.num_sampling_steps,
            )
            for stem, seq in seen.items()
        }
        for stem, fut in futures.items():
            try:
                print(f"[done] {stem}: {fut.get()}")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {stem}: {e}")
    print("run_cli: done. Sync with: modal volume get", OUTPUTS_VOLUME_NAME, ". <pred_root>/esmfold2/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="One or more eval manifest CSVs (need stem,input_seq columns).")
    ap.add_argument("--n-samples", type=int, default=5, help="Diffusion samples per protein (best-of-N).")
    ap.add_argument("--num-loops", type=int, default=20)
    ap.add_argument("--num-sampling-steps", type=int, default=100)
    ap.set_defaults(func=run_cli)
    args = ap.parse_args()
    args.func(args)
