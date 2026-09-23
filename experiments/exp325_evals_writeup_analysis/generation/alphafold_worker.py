"""Predictor adapters: fixed budgets, confidence selection and direct timings.

Optional predictor imports live inside these GPU-only adapters. Both operate on
sequence/MSA inputs alone; the CPU scorer receives the experimental structures
later. Inference timing includes JIT compilation of previously unseen shapes,
which is recorded explicitly and is not used for a speed comparison.
"""

import gzip
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np


class AF2:
    """ColabFold's public AlphaFold2 feature/model interfaces, five pTM weights."""

    def __init__(self, weights: Path) -> None:
        from colabfold.alphafold.models import load_models_and_params

        self.models = load_models_and_params(
            num_models=5, use_templates=False, num_recycles=3,
            recycle_early_stop_tolerance=0.0, model_order=[1, 2, 3, 4, 5],
            model_type="alphafold2_ptm", data_dir=weights, rank_by="plddt",
            max_seq=512, max_extra_seq=1024, compile_mode="fast",
        )

    def predict(self, record: dict, msa: str, output: Path) -> dict:
        """Save all five structures and choose solely by mean pLDDT."""
        from alphafold.common import protein
        from colabfold.batch import generate_input_feature, mk_mock_template, pad_input

        sequence = record["sequence"]
        features, _ = generate_input_feature(
            [sequence], [1], [msa], None, [mk_mock_template(sequence)],
            is_complex=False, model_type="alphafold2_ptm", max_seq=512,
        )
        prepared = None
        samples = []
        elapsed = 0.0
        for name, runner, params in self.models:
            # The public loader intentionally shares a runner among parameter
            # sets. Assigning its parameter state is the upstream inference API.
            runner.params = params
            if prepared is None:
                prepared = runner.process_features(features, random_seed=42)
                repeats = prepared["aatype"].shape[0]
                prepared["asym_id"] = np.tile(features["asym_id"], repeats).reshape(repeats, -1)
                padded = math.ceil(len(sequence) / 128) * 128
                if padded > len(sequence):
                    prepared = pad_input(prepared, runner, name, padded, False)
            started = time.monotonic()
            result, recycles = runner.predict(prepared, random_seed=42, return_representations=False)
            duration = time.monotonic() - started
            elapsed += duration
            mask = result["structure_module"]["final_atom_mask"]
            predicted = protein.from_prediction(
                features=prepared, result=result, b_factors=result["plddt"][:, None] * mask,
                remove_leading_feature_dimension=True,
            )
            # ColabFold's prediction API removes padding before returning result.
            # Retain only the query's residues in case feature padding is present.
            predicted = protein.Protein(
                atom_positions=predicted.atom_positions[:len(sequence)],
                aatype=predicted.aatype[:len(sequence)], atom_mask=predicted.atom_mask[:len(sequence)],
                residue_index=predicted.residue_index[:len(sequence)],
                chain_index=predicted.chain_index[:len(sequence)], b_factors=predicted.b_factors[:len(sequence)],
            )
            path = output / f"{name}.pdb"
            path.write_text(protein.to_pdb(predicted))
            confidence = float(np.mean(result["plddt"][:len(sequence)]))
            samples.append(dict(model=name, confidence=confidence, ptm=float(result["ptm"]),
                                recycles=int(recycles), elapsed_seconds=duration, file=path.name))
        best = max(samples, key=lambda row: row["confidence"])
        shutil.copyfile(output / best["file"], output / "selected.pdb")
        (output / "candidates.json").write_text(json.dumps(samples, indent=2) + "\n")
        return dict(elapsed_seconds=elapsed, n_samples=5, selected=best["model"],
                    selection_confidence=best["confidence"], n_cycles=3, n_seeds=1)


class AF3:
    """Official AlphaFold3 interfaces, five fixed seeds and five samples each."""

    def __init__(self, weights: Path) -> None:
        import jax

        sys.path.insert(0, "/opt/alphafold3")
        import run_alphafold

        self.api = run_alphafold
        self.runner = run_alphafold.ModelRunner(
            config=run_alphafold.make_model_config(num_diffusion_samples=5, num_recycles=10),
            device=jax.local_devices(backend="gpu")[0], model_dir=weights,
        )
        # Materialize the lazy weight property during separately timed setup.
        if not self.runner.model_params:
            raise ValueError("AlphaFold3 parameter load returned no tensors")

    def predict(self, record: dict, msa: str, output: Path) -> dict:
        """Use the official confidence-selection/output writer, including notices."""
        import jax
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation

        name = record["stem"].lower()
        payload = dict(name=name, modelSeeds=[42, 43, 44, 45, 46], dialect="alphafold3", version=1,
                       sequences=[{"protein": {"id": "A", "sequence": record["sequence"],
                                                "unpairedMsa": msa, "pairedMsa": "", "templates": []}}])
        fold_input = folding_input.Input.from_json(json.dumps(payload))
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
        examples = featurisation.featurise_input(
            fold_input=fold_input, buckets=(256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048),
            ccd=ccd, verbose=False,
        )
        results = []
        elapsed = 0.0
        for seed, example in zip(fold_input.rng_seeds, examples, strict=True):
            started = time.monotonic()
            prediction = self.runner.run_inference(example, jax.random.PRNGKey(seed))
            elapsed += time.monotonic() - started
            structures = self.runner.extract_inference_results(example, prediction, name)
            results.append(self.api.ResultsForSeed(seed=seed, inference_results=structures,
                                                   full_fold_input=fold_input))
        self.api.write_outputs(results, output_dir=output, job_name=name)
        # PAE/contact-probability tensors are quadratic and are not inputs to
        # these analyses. Keep every CIF and summary confidence, which suffice
        # to reproduce selection, structural scores and pyconfind contacts.
        for path in output.rglob("*_confidences.json"):
            if not path.name.endswith("_summary_confidences.json"):
                path.unlink()
        shutil.copyfile(output / f"{name}_model.cif", output / "selected.cif")
        scores = [(r.seed, i, float(s.metadata["ranking_score"])) for r in results
                  for i, s in enumerate(r.inference_results)]
        best = max(scores, key=lambda r: r[2])
        return dict(elapsed_seconds=elapsed, n_samples=25, selected=f"seed-{best[0]}_sample-{best[1]}",
                    selection_confidence=best[2], n_cycles=10, n_seeds=5)


def read_msa(path: Path) -> str:
    """Read the frozen query alignment without changing sequence membership."""
    return gzip.decompress(path.read_bytes()).decode()
