"""Freeze AlphaFold baselines on the exact MSA queries used for depth strata.

Both baselines predict the protein chain with the shared precomputed alignment,
without templates. The full MarinFold prompt sequence remains the contact-score
coordinate system; the existing sequence-alignment scorer handles the mapping.
No experimental coordinates are supplied to either predictor.
"""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HELICO = Path("/home/bizon/git/helico")
MSA = Path.home() / ".cache/helico/data/benchmarks/FoldBench/foldbench-msas"
DEST = ROOT / "scratch/alphafold/inputs"
AF3_SHA = "3c89cc7b89aa7042b72885af9016a45b262da008"


def digest(path: Path) -> str:
    """Hash inputs without loading checkpoint weights into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Pin sequences, alignments, weights and selection before baseline inference."""
    DEST.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(HELICO / "experiments/exp14_foldbench_held_out_monomers/data/targets.csv")
    depth = pd.read_csv(ROOT / "data/inputs/helico_msa_depth.csv").set_index("target_id")
    if len(targets) != 333 or targets.stem.duplicated().any():
        raise ValueError("Require the fixed 333-target universe")
    records = []
    for row in targets.sort_values("stem").itertuples():
        msa_key = hashlib.sha256((row.input_seq + "\n").encode()).hexdigest()
        source = MSA / f"{msa_key}.a3m.gz"
        text = gzip.decompress(source.read_bytes()).decode()
        lines = text.splitlines()
        query = lines[1]
        if query != row.input_seq or sum(s.startswith(">") for s in lines) != depth.loc[row.stem, "n_sequences"]:
            raise ValueError(f"Alignment/query/depth mismatch: {row.stem}")
        shutil.copyfile(source, DEST / f"{row.stem}.a3m.gz")
        records.append(dict(stem=row.stem, sequence=row.input_seq, n_residues=len(row.input_seq),
                            eval_set=row.eval_set, msa_depth=int(depth.loc[row.stem, "n_sequences"]),
                            msa_sha256=digest(source)))
    (DEST / "targets.json").write_text(json.dumps(records, indent=2) + "\n")
    af2 = Path.home() / ".cache/colabfold/params"
    weights = {f"af2/params/{p.name}": {"sha256": digest(p), "bytes": p.stat().st_size}
               for p in sorted(af2.glob("*ptm.npz"))}
    af3 = Path.home() / "Dropbox/AF3/params-20241113/af3.bin.zst"
    weights["af3/af3.bin.zst"] = {"sha256": digest(af3), "bytes": af3.stat().st_size}
    manifest = {
        "status": "fixed_before_inference", "n_targets": 333,
        "inputs_sha256": digest(DEST / "targets.json"), "weights": weights,
        "shared_protocol": "Protein chain only; exact archived MSA query/alignment; no templates; no relaxation; confidence-only selection; no test tuning",
        "af2": {"colabfold": "1.6.2", "alphafold_colabfold": "2.3.18", "jax": "0.10.2",
                "model_type": "alphafold2_ptm", "models": [1, 2, 3, 4, 5], "seeds": [42],
                "recycles": 3, "early_stop_tolerance": 0, "selection": "mean pLDDT",
                "max_seq": 512, "max_extra_seq": 1024},
        "af3": {"source_sha": AF3_SHA, "seeds": [42, 43, 44, 45, 46],
                "diffusion_samples_per_seed": 5, "recycles": 10, "selection": "ranking_score"},
        "msa_interpretation": "Depth is now the supplied alignment count for both new baselines, not the depth from the original AF3 search pipeline.",
        "parameters_private": "Do not include weights in the public artifact package.",
    }
    (ROOT / "data/alphafold_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(records)} shared-MSA inputs; {sum(w['bytes'] for w in weights.values()) / 1e9:.2f} GB of private weights")


if __name__ == "__main__":
    main()
