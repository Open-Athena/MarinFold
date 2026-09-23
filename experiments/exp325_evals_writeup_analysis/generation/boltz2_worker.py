"""Resident Boltz-2 adapter using the pinned upstream inference components.

Only structure prediction runs. Inputs carry archived MSAs, no templates,
constraints, ligands or affinity request. The upstream writer saves all 25
samples in confidence rank order; selection never sees experimental structures.
"""

import gzip
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Callback, Trainer, seed_everything
from rdkit import Chem

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import MSA, Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2, process_inputs
from boltz.model.models.boltz2 import Boltz2


class InferenceTimer(Callback):
    """Time GPU prediction only, before the upstream file-writing callback."""

    def __init__(self) -> None:
        self.elapsed_seconds = 0.0
        self.started = 0.0
        self.batches = 0

    def on_predict_batch_start(self, trainer: Trainer, pl_module: Boltz2, batch: dict,
                               batch_idx: int, dataloader_idx: int = 0) -> None:
        torch.cuda.synchronize()
        self.started = time.monotonic()

    def on_predict_batch_end(self, trainer: Trainer, pl_module: Boltz2, outputs: dict,
                             batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        torch.cuda.synchronize()
        self.elapsed_seconds += time.monotonic() - self.started
        self.batches += 1
        if outputs["exception"]:
            raise RuntimeError("Boltz-2 reported a failed prediction")


class Boltz2Predictor:
    """Load one immutable checkpoint and reuse it across independent targets."""

    def __init__(self, cache: Path) -> None:
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        self.cache = cache
        self.model = Boltz2.load_from_checkpoint(
            cache / "boltz2_conf.ckpt", strict=True, map_location="cpu", ema=False,
            predict_args=dict(recycling_steps=10, sampling_steps=200, diffusion_samples=25,
                              max_parallel_samples=5, write_confidence_summary=True,
                              write_full_pae=False, write_full_pde=False),
            diffusion_process_args=asdict(Boltz2DiffusionParams()),
            pairformer_args=asdict(PairformerArgsV2()), msa_args=asdict(MSAModuleArgs()),
            steering_args=asdict(BoltzSteeringParams()), use_kernels=True,
        ).eval().cuda()
        torch.cuda.synchronize()

    def predict(self, record: dict, msa_path: Path, output: Path) -> dict:
        """Generate all candidates and record confidence-selected coordinates."""
        seed_everything(42, workers=True)
        if hashlib.sha256(msa_path.read_bytes()).hexdigest() != record["msa_sha256"]:
            raise ValueError("Archived MSA digest mismatch")
        msa = output / "input.a3m"
        msa.write_text(gzip.decompress(msa_path.read_bytes()).decode())
        request = output / f"{record['stem']}.yaml"
        request.write_text(yaml.safe_dump({"version": 1, "sequences": [{"protein": {
            "id": "A", "sequence": record["sequence"], "msa": str(msa),
        }}]}))
        process_inputs(data=[request], out_dir=output, ccd_path=self.cache / "ccd.pkl",
                       mol_dir=self.cache / "mols", msa_server_url="https://api.colabfold.com",
                       msa_pairing_strategy="greedy", max_msa_seqs=8192,
                       use_msa_server=False, boltz2=True, preprocessing_threads=1)
        processed = output / "processed"
        manifest = Manifest.load(processed / "manifest.json")
        if [r.id for r in manifest.records] != [record["stem"]]:
            raise ValueError("Upstream preprocessing did not produce the requested target")
        msa_id = manifest.records[0].chains[0].msa_id
        n_msa_sequences_processed = len(MSA.load(processed / "msa" / f"{msa_id}.npz").sequences)
        timer = InferenceTimer()
        writer = BoltzWriter(processed / "structures", output / "predictions", boltz2=True)
        trainer = Trainer(default_root_dir=output, callbacks=[timer, writer], accelerator="gpu",
                          devices=1, precision="bf16-mixed", logger=False, enable_checkpointing=False,
                          enable_progress_bar=False)
        data = Boltz2InferenceDataModule(manifest=manifest, target_dir=processed / "structures",
                                         msa_dir=processed / "msa", mol_dir=self.cache / "mols",
                                         num_workers=0, constraints_dir=processed / "constraints",
                                         template_dir=processed / "templates", extra_mols_dir=processed / "mols")
        trainer.predict(self.model, datamodule=data, return_predictions=False)
        if timer.batches != 1 or writer.failed:
            raise ValueError("Expected one successful prediction batch")
        candidates = []
        directory = output / "predictions" / record["stem"]
        for rank in range(25):
            tag = f"{record['stem']}_model_{rank}"
            path = directory / f"{tag}.cif"
            confidence = json.loads((directory / f"confidence_{tag}.json").read_text())
            candidates.append(dict(rank=rank, file=str(path.relative_to(output)),
                                   confidence=confidence["confidence_score"],
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        best = max(candidates, key=lambda r: r["confidence"])
        if best["rank"] != 0:
            raise ValueError("Upstream candidate order differs from confidence order")
        shutil.copyfile(output / best["file"], output / "selected.cif")
        (output / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
        return dict(elapsed_seconds=timer.elapsed_seconds, selected=best["file"],
                    selection_confidence=best["confidence"], n_samples=25, n_seeds=1,
                    seed=42, recycling_steps=10, sampling_steps=200, max_parallel_samples=5,
                    n_msa_sequences_processed=n_msa_sequences_processed)
