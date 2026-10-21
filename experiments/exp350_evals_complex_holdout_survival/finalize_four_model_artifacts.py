"""Validate four-arm coverage and stage public structures, inputs, and timings."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import gemmi
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Return a reproducible SHA256 for a small input or result artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    """Copy an artifact after creating its destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    """Reject missing targets or incorrect confidence-selected structures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm", type=Path, required=True)
    parser.add_argument("--af3", type=Path, required=True)
    parser.add_argument("--af3-inputs", type=Path, required=True)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--multichain", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    targets = json.loads((HERE / "data/four_model_v1/targets.json").read_text())
    timings = []
    coverage = []
    for t in targets:
        stem = t["stem"]
        esm = args.esm / stem
        af3 = args.af3 / stem
        provenance = json.loads((esm / "provenance.json").read_text())
        best = max(provenance["samples"], key=lambda r: r["ranking_score"])
        if provenance["selected_seed"] != best["seed"] or digest(
            esm / "structure.cif"
        ) != digest(esm / f"sample_{best['seed']}.cif"):
            raise ValueError(f"{stem}: ESMFold2 confidence selection mismatch")
        if provenance["input"]["chain_sequences"] != t["chain_sequences"]:
            raise ValueError(f"{stem}: ESMFold2 sequence mismatch")
        sample_timings = [
            next(iter(csv.DictReader((esm / f"timing_{seed}.csv").open())))
            for seed in range(5)
        ]
        timings.extend({**r, "comparison_model": "ESMFold2"} for r in sample_timings)
        for p in esm.iterdir():
            if p.suffix in (".cif", ".json", ".csv"):
                copy_file(p, args.stage / "structures/esmfold2" / stem / p.name)
        rank = list(csv.DictReader((af3 / f"{stem}_ranking_scores.csv").open()))
        winner = max(rank, key=lambda r: float(r["ranking_score"]))
        sample_name = f"seed-{winner['seed']}_sample-{winner['sample']}"
        chosen = af3 / sample_name / f"{stem}_{sample_name}_model.cif"
        # AF3 rewrites the emission timestamp when copying its top-ranked model.
        # Atom identities, coordinates, occupancies, and confidence must still
        # match exactly; byte equality would reject metadata-only differences.
        selected_atoms = (
            gemmi.cif.read_file(str(chosen))
            .sole_block()
            .get_mmcif_category("_atom_site.")
        )
        top_atoms = (
            gemmi.cif.read_file(str(af3 / f"{stem}_model.cif"))
            .sole_block()
            .get_mmcif_category("_atom_site.")
        )
        if selected_atoms != top_atoms:
            raise ValueError(f"{stem}: AF3 confidence selection mismatch")
        raw = json.loads((args.af3_inputs / f"{stem}.json").read_text())
        if [p["protein"]["sequence"] for p in raw["sequences"]] != t["chain_sequences"]:
            raise ValueError(f"{stem}: AF3 sequence mismatch")
        timing = next(iter(csv.DictReader((af3 / "timings.csv").open())))
        timings.append({**timing, "comparison_model": "AlphaFold3"})
        for p in af3.rglob("*"):
            if p.is_file() and (
                p.suffix in (".cif", ".csv", ".md")
                or p.name.endswith("_summary_confidences.json")
            ):
                copy_file(
                    p, args.stage / "structures/alphafold3" / stem / p.relative_to(af3)
                )
        copy_file(
            args.af3_inputs / f"{stem}.json",
            args.stage / "inputs/alphafold3" / f"{stem}.json",
        )
        coverage.append(
            {
                "stem": stem,
                "split": t["split"],
                "n_residues": t["L"],
                "marinfold_multichain_rollouts": 100,
                "marinfold_default_rollouts": 100,
                "esmfold2_samples": len(provenance["samples"]),
                "alphafold3_samples": len(rank),
            }
        )
    for label, root in [
        ("MarinFold multichain", args.multichain),
        ("MarinFold default + 10G", args.default),
    ]:
        ts = [
            r
            for p in (root / "timings").glob("*.parquet")
            for r in pq.read_table(p).to_pylist()
        ]
        if {r["stem"] for r in ts} != {t["stem"] for t in targets} or len(ts) != 23:
            raise ValueError(f"{label}: incomplete timing coverage")
        if any(r["n_rollouts"] != 100 or r["unfinished_rollouts"] for r in ts):
            raise ValueError(f"{label}: incomplete rollouts")
        timings.extend({**r, "comparison_model": label} for r in ts)
    for name, rows in [("timings.csv", timings), ("coverage.csv", coverage)]:
        p = HERE / "data/four_model_v1" / name
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with p.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "cohort_sha256": digest(
            HERE / "data/foldbench_complex_contact_eval_targets.parquet"
        ),
        "n_targets": 23,
        "n_test": 17,
        "n_dev": 6,
        "all_models_complete": True,
        "contact_operator": {
            "pyconfind": "0.6.0",
            "native_only": True,
            "contact_distance": 3.0,
            "dcut": 25.0,
            "clash_distance": 2.0,
            "ground_truth_degree_threshold": 0.001,
        },
        "multichain_checkpoint": "contacts-v1-exp343-m2-p06-complex-1.5B-step-280154",
        "default_checkpoint": "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        "default_input": "chain A + 10 glycines + chain B; linker excluded from scoring",
        "esmfold2_revision": "8fc3ff471022fdce52c77030685eb775de0c00a3",
        "esmc_revision": "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
        "esmfold2_modal_image": "im-43jtzo7qIRzctCEmR2QSuM",
        "esmfold2_modal_app": "ap-BZ3G3WASU2AuiVg95w52fQ",
        "esmfold2_status": "All predictions recovered from saved Modal outputs; original caller returned a TorchVersion serialization error after saving each target. No predictions were discarded or rerun to improve accuracy.",
        "alphafold3_docker_image": "sha256:75155f945d2d561484c31a425801d3f6b949655fd8eb156620c12b1b69d571a7",
        "alphafold3_weights_release": "2024-11-13",
        "alphafold3_weights_sha256": "0dd0290af76eec0119f4c0c41f441fbce4ec18aa21a0a02cac4a3af01d841369",
        "alphafold3_input": "native two-chain input, ColabFold paired/unpaired MSAs, empty templates",
        "public_root": "hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/contact_eval_v1/four_model_v1",
        "files": [
            {
                "path": str(p.relative_to(args.stage)),
                "size": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(args.stage.rglob("*"))
            if p.is_file()
        ],
    }
    (HERE / "data/four_model_v1/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        "Validated all four models on",
        len(targets),
        "complexes;",
        len(timings),
        "timing rows",
    )


if __name__ == "__main__":
    main()
