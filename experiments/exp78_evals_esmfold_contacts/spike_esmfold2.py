# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Feasibility spike for ESMFold2 (``biohub/ESMFold2``) on Modal.

Run before the full fan-out to confirm, on a single short sequence:
  1. the weights download (and whether the HF repo is gated),
  2. the model loads and fits on one H100,
  3. the documented ``ESMFold2InputBuilder().fold(...)`` API works,
  4. what the result object actually exposes (so ``_score_confidence`` in
     ``esmfold2_app.py`` targets the right attribute for best-of-N), and
  5. ``result.complex.to_mmcif()`` yields a parseable structure.

Usage::

    modal run spike_esmfold2.py

Prints a diagnostic dump; writes nothing persistent.
"""

import modal

from esmfold2_app import ESMFOLD2_IMAGE, HF_MODEL_ID, HF_SECRET, WEIGHTS_VOL

# The spike module imports esmfold2_app at top level, which also re-executes
# inside the remote container — so the container needs esmfold2_app.py present.
app = modal.App(
    "esmfold2-spike-exp78",
    image=ESMFOLD2_IMAGE.add_local_python_source("esmfold2_app"),
)

# A short well-folded test sequence (villin headpiece HP35).
TEST_SEQ = "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"


@app.function(volumes={"/weights": WEIGHTS_VOL}, gpu="H100", timeout=60 * 40, secrets=[HF_SECRET])
def spike() -> dict:
    import json
    import time

    import torch

    report: dict = {}

    # 1. package + API surface
    import esm
    report["esm_version"] = getattr(esm, "__version__", "?")
    from esm.models import esmfold2 as ef2
    report["esmfold2_exports"] = [n for n in dir(ef2) if not n.startswith("_")]

    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    # 2. weights + load
    t0 = time.time()
    model = ESMFold2Model.from_pretrained(HF_MODEL_ID).cuda().eval()
    report["load_seconds"] = round(time.time() - t0, 1)
    report["param_billions"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)

    # 3. fold
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=TEST_SEQ)])
    t0 = time.time()
    result = ESMFold2InputBuilder().fold(
        model, spi, num_loops=20, num_sampling_steps=100, num_diffusion_samples=1, seed=0
    )
    report["fold_seconds"] = round(time.time() - t0, 1)

    # 4. introspect result for the confidence accessor
    report["result_type"] = type(result).__name__
    report["result_attrs"] = [n for n in dir(result) if not n.startswith("_")]
    if hasattr(result, "complex"):
        report["complex_attrs"] = [n for n in dir(result.complex) if not n.startswith("_")]
    for attr in ("ptm", "mean_plddt", "plddt", "confidence"):
        for obj_name, obj in (("result", result), ("complex", getattr(result, "complex", None))):
            if obj is not None and hasattr(obj, attr):
                v = getattr(obj, attr)
                report[f"conf::{obj_name}.{attr}"] = str(type(v))

    # 5. mmCIF
    mmcif = result.complex.to_mmcif()
    report["mmcif_len"] = len(mmcif)
    report["mmcif_head"] = mmcif[:200]

    report["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(spike.remote(), indent=2))
