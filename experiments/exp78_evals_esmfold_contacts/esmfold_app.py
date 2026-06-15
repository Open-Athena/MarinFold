# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Modal app: run ESMFold (``facebook/esmfold_v1``) on the exp78 eval set.

ESMFold is a **single-sequence** structure predictor (an OpenFold-style
folding trunk on top of the ESM2-3B language model) — no MSA, no diffusion
sampling, deterministic given the sequence and recycle count. We run it
once per protein and persist the predicted structure as mmCIF so
``contact_eval.py`` can run pyconfind on it and rank contacts by degree,
exactly like exp74's Protenix "structure" predictor.

Two Volumes:
- ``esmfold-weights`` — the HF ``facebook/esmfold_v1`` snapshot, downloaded
  once so reruns don't re-pull ~5 GB of weights.
- ``esmfold-exp78-runs`` — per-protein outputs (``{stem}/structure.cif``,
  ``{stem}/plddt.json``, ``{stem}/timings.json``).

Entry points:
- ``setup_weights()`` — one-time snapshot of the weights Volume.
- ``ESMFoldWorker.predict_one(...)`` — fold one sequence, persist outputs.

Recommended settings (logged per run): ``num_recycles=4`` (the model
default), language-model head cast to fp16 to fit comfortably on one GPU,
and a trunk ``chunk_size`` for attention to keep memory bounded. The eval
set tops out at 460 aa so chunking is rarely engaged.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import modal

APP_NAME = "esmfold-contacts-eval-exp78"

WEIGHTS_VOLUME_NAME = "esmfold-weights"
OUTPUTS_VOLUME_NAME = "esmfold-exp78-runs"

WEIGHTS_VOL = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
OUTPUTS_VOL = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)

HF_MODEL_ID = "facebook/esmfold_v1"

ESMFOLD_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        # transformers vendors the OpenFold structure module ESMFold needs;
        # no separate openfold install. accelerate eases device placement.
        "torch",
        "transformers>=4.40",
        "accelerate",
        "huggingface_hub[hf_transfer]",
        "gemmi>=0.6",   # PDB -> mmCIF conversion in-worker
        "numpy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/weights/hf",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App(APP_NAME, image=ESMFOLD_IMAGE)


@app.function(volumes={"/weights": WEIGHTS_VOL}, timeout=60 * 30)
def setup_weights() -> dict:
    """Snapshot the ESMFold v1 weights from HF into the weights Volume."""
    from huggingface_hub import snapshot_download

    target = snapshot_download(HF_MODEL_ID, cache_dir="/weights/hf")
    WEIGHTS_VOL.commit()
    print(f"snapshot -> {target}")
    return {"model": HF_MODEL_ID, "cache": target}


@app.cls(
    volumes={"/weights": WEIGHTS_VOL, "/outputs": OUTPUTS_VOL},
    gpu="H100",
    timeout=60 * 60 * 2,
)
@modal.concurrent(max_inputs=1)
class ESMFoldWorker:
    """Long-lived ESMFold worker. Weights load once per container."""

    @modal.enter()
    def setup(self) -> None:
        import platform
        import socket

        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = EsmForProteinFolding.from_pretrained(HF_MODEL_ID)
        model = model.cuda().eval()
        # fp16 the ESM2 language-model stem (the bulk of the params); keep the
        # folding trunk in fp32 for numerical stability — the standard recipe.
        model.esm = model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        # Chunk the trunk's attention to bound memory on longer sequences.
        model.trunk.set_chunk_size(128)
        self.model = model

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
        num_recycles: int = 4,
        force_run: bool = False,
    ) -> dict:
        """Fold one sequence; persist ``{stem}/structure.cif`` + plddt + timings."""
        import gemmi
        import torch

        out_dir = Path("/outputs") / stem
        OUTPUTS_VOL.reload()
        if (out_dir / "structure.cif").exists() and not force_run:
            print(f"[{stem}] already complete; skipping.")
            return {"stem": stem, "skipped": True, "output_dir": str(out_dir)}

        t0 = time.time()
        tokenized = self.tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"].cuda()
        with torch.no_grad():
            output = self.model(tokenized, num_recycles=num_recycles)
        pdb_str = self.model.output_to_pdb(output)[0]
        # Mean pLDDT (ESMFold writes pLDDT*100 into the b-factor; `plddt` head
        # is [batch, res, atom]). Report the per-residue mean over CA-ish atoms.
        plddt = float(output["plddt"][0, :, 1].mean().item())
        elapsed = time.time() - t0

        out_dir.mkdir(parents=True, exist_ok=True)
        # Convert the PDB to mmCIF so the structure file matches the rest of
        # the pipeline (`structure.cif`); gemmi round-trips coordinates.
        st = gemmi.read_pdb_string(pdb_str)
        st.setup_entities()
        (out_dir / "structure.pdb").write_text(pdb_str)
        st.make_mmcif_document().write_file(str(out_dir / "structure.cif"))
        (out_dir / "plddt.json").write_text(json.dumps({"stem": stem, "mean_plddt": plddt}))

        from datetime import datetime, timezone
        timings = {
            "stem": stem, "n_residues": len(sequence),
            "elapsed_seconds": round(elapsed, 4), "num_recycles": num_recycles,
            "model_nickname": "esmfold", "runner_tag": "modal",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **self.worker_meta,
        }
        (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))
        OUTPUTS_VOL.commit()
        return {"stem": stem, "skipped": False, "mean_plddt": round(plddt, 2),
                "elapsed_seconds": round(elapsed, 4), "output_dir": str(out_dir)}


def run_cli(args: argparse.Namespace) -> None:
    """Fan ESMFold out across every protein in the eval manifest(s)."""
    rows: list[dict] = []
    for manifest in args.manifest:
        with open(manifest) as f:
            rows.extend(csv.DictReader(f))
    # De-dup by stem (FoldBench + exp65 are disjoint, but be safe).
    seen: dict[str, str] = {}
    for r in rows:
        seen.setdefault(r["stem"], r["input_seq"])
    print(f"run_cli: {len(seen)} unique proteins (num_recycles={args.num_recycles})")

    with app.run():
        worker = ESMFoldWorker()
        futures = {
            stem: worker.predict_one.spawn(stem=stem, sequence=seq, num_recycles=args.num_recycles)
            for stem, seq in seen.items()
        }
        for stem, fut in futures.items():
            try:
                print(f"[done] {stem}: {fut.get()}")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {stem}: {e}")
    print("run_cli: done. Sync with: modal volume get", OUTPUTS_VOLUME_NAME, ". <pred_root>/esmfold/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="One or more eval manifest CSVs (need stem,input_seq columns).")
    ap.add_argument("--num-recycles", type=int, default=4)
    ap.set_defaults(func=run_cli)
    args = ap.parse_args()
    args.func(args)
