"""Score confidence-selected AF structures with the existing structure/contact metrics.

CPU preprocessing only. Sequence/atom correspondence is verified before scoring;
contact scores use exp245's frozen truth and exp89's resolved-pair universe.
Run in the existing Helico environment, which contains the scientific parsers.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT.parent
HELICO = Path("/home/bizon/git/helico")
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
sys.path.insert(0, str(HELICO / "src"))
sys.path.insert(0, str(EXPERIMENTS / "exp78_evals_esmfold_contacts"))
sys.path.insert(0, str(EXPERIMENTS / "exp89_evals_contacts_v1_model_on_eval_set"))
sys.path.insert(0, str(EXPERIMENTS.parent / "marinfold"))
from compute_metrics import degree_matrix, metric_rows, resolved_pairs, true_matrix
from helico.bench import compute_gdt_ts, compute_lddt, compute_rmsd, compute_tm_score, structure_to_chains
from helico.data import parse_mmcif
from pyconfind_contacts import compute_contacts


def structure_scores(path: Path, ground_truth: Path, sequence: str) -> dict:
    """Match protein atoms by verified query position and atom name."""
    gt = parse_mmcif(ground_truth, max_resolution=float("inf"))
    if gt is None:
        raise ValueError(f"Cannot parse truth: {ground_truth}")
    chains = [c for c in structure_to_chains(gt) if c["type"] == "protein"]
    if len(chains) != 1 or chains[0]["sequence"] != sequence:
        raise ValueError("Ground-truth protein sequence differs from inference input")
    truth = next(c for c in gt.chains if c.chain_id == chains[0]["id"])
    pred = gemmi.read_structure(str(path))
    pred.setup_entities()
    if len(pred[0]) != 1:
        raise ValueError("Require one predicted protein chain")
    residues = {int(r.seqid.num) - 1: r for r in pred[0][0]}
    required = {i for i, aa in enumerate(sequence) if aa != "X"}
    if not required.issubset(residues) or not set(residues).issubset(range(len(sequence))):
        raise ValueError("Prediction residue indices differ from input sequence")
    predicted_atoms, true_atoms, backbone = [], [], []
    for i, residue in enumerate(truth.residues):
        if i not in residues:  # AF2 has no atom definition for unknown residue X.
            continue
        predicted = residues[i]
        letter = gemmi.find_tabulated_residue(predicted.name).one_letter_code
        if letter.upper() != sequence[i] and sequence[i] != "X":
            raise ValueError(f"Predicted sequence mismatch at {i}: {predicted.name}")
        atoms = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in predicted}
        for atom in residue.atoms:
            if atom.name in atoms:
                predicted_atoms.append(atoms[atom.name])
                true_atoms.append(atom.coords)
                backbone.append(atom.name == "CA")
    p, g, mask = np.asarray(predicted_atoms), np.asarray(true_atoms), np.asarray(backbone)
    if mask.sum() < 3:
        raise ValueError("Too few matched backbone atoms")
    return dict(gdt_ts=compute_gdt_ts(p[mask], g[mask]), lddt=compute_lddt(p, g),
                tm_score=compute_tm_score(p[mask], g[mask]), rmsd=compute_rmsd(p[mask], g[mask]),
                n_matched_atoms=len(p), n_matched_ca=int(mask.sum()),
                missing_unknown_residues="|".join(map(str, sorted(set(range(len(sequence))) - set(residues)))))


def score_one(task: tuple[dict, dict, dict, str]) -> dict:
    """Read one complete result, verify selection and persist an auditable CPU cache."""
    record, truth, target, variant = task
    stem = record["stem"]
    directory = ROOT / "scratch/alphafold/results" / variant / stem
    marker = json.loads((directory / "complete.json").read_text())
    if any(marker[key] != value for key, value in record.items()):
        raise ValueError(f"{stem}: completed prediction has different inputs")
    path = directory / ("selected.pdb" if variant == "af2" else "selected.cif")
    if variant == "af2":
        candidates = json.loads((directory / "candidates.json").read_text())
        if {r["model"] for r in candidates} != {f"model_{i}" for i in range(1, 6)} or len(candidates) != 5:
            raise ValueError("Incomplete AF2 candidate budget")
        if any(not (directory / r["file"]).is_file() for r in candidates):
            raise ValueError("Missing AF2 candidate structure")
        best = max(candidates, key=lambda r: r["confidence"])
        selected_path = directory / best["file"]
        if (marker["selected"] != best["model"] or marker["selection_confidence"] != best["confidence"]
                or path.read_bytes() != selected_path.read_bytes()):
            raise ValueError("AF2 output is not the confidence-selected candidate")
    else:
        with (directory / f"{stem.lower()}_ranking_scores.csv").open() as stream:
            scores = list(csv.DictReader(stream))
        expected = {(seed, sample) for seed in range(42, 47) for sample in range(5)}
        if {(int(r["seed"]), int(r["sample"])) for r in scores} != expected or len(scores) != 25:
            raise ValueError("Incomplete AF3 candidate budget")
        for seed, sample in expected:
            tag = f"seed-{seed}_sample-{sample}"
            if not (directory / tag / f"{stem.lower()}_{tag}_model.cif").is_file():
                raise ValueError("Missing AF3 candidate structure")
        best = max(scores, key=lambda r: float(r["ranking_score"]))
        selected = f"seed-{best['seed']}_sample-{best['sample']}"
        selected_path = directory / selected / f"{stem.lower()}_{selected}_model.cif"
        # The official writer stamps each CIF separately, so the selected copy
        # can differ in its generation timestamp. Require identical atom tables.
        selected_atoms = gemmi.cif.read_file(str(path)).sole_block().get_mmcif_category("_atom_site.")
        candidate_atoms = gemmi.cif.read_file(str(selected_path)).sole_block().get_mmcif_category("_atom_site.")
        if (marker["selected"] != selected or marker["selection_confidence"] != float(best["ranking_score"])
                or not selected_atoms or selected_atoms != candidate_atoms):
            raise ValueError("AF3 output is not the confidence-selected candidate")
    gt_path = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / f"{target['pdb_id']}.cif.gz"
    structural = structure_scores(path, gt_path, record["sequence"])
    contacts = compute_contacts(path, target["prompt_sequence"], stem=stem)
    if contacts.alignment_identity < 0.9:
        raise ValueError(f"{stem}: contact coordinate identity {contacts.alignment_identity}")
    score = degree_matrix(truth["L"], contacts.contacts)
    metrics = metric_rows(score, true_matrix(truth["L"], truth["contacts"]),
                          *resolved_pairs(np.asarray(truth["resolved"])), truth["L"], with_precision=True)
    return dict(stem=stem, method=variant, eval_set=record["eval_set"],
                selected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ground_truth_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                selected_file=str(path.relative_to(ROOT)), selection_confidence=marker["selection_confidence"],
                contact_alignment_identity=contacts.alignment_identity, **structural,
                contact_metrics=metrics, contacts=contacts.contacts, timing=marker)


def main() -> None:
    """Require all fixed targets unless explicitly validating a smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["af2", "af3"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELICO, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico metric implementation differs from the archived comparisons")
    records = json.loads((ROOT / "scratch/alphafold/inputs/targets.json").read_text())
    truth_path = ROOT / "scratch/contacts/inputs/gt_universe_scored.jsonl"
    truth_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    if truth_sha != "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5":
        raise ValueError("Ground-truth contact universe differs from the frozen benchmark")
    truth = {r["stem"]: r for r in map(json.loads, truth_path.read_text().splitlines())}
    targets = pd.read_csv(HELICO / "experiments/exp14_foldbench_held_out_monomers/data/targets.csv").set_index("stem")
    prompts = pd.read_csv(EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv").set_index("stem")
    present = {p.parent.name for p in (ROOT / "scratch/alphafold/results" / args.variant).glob("*/complete.json")}
    if not present:
        raise ValueError(f"No {args.variant} predictions found")
    if not args.smoke and present != {r["stem"] for r in records}:
        raise ValueError(f"Incomplete {args.variant}: {len(present)}/333; resume before scoring")
    tasks = [(r, truth[r["stem"]], {**targets.loc[r["stem"]].to_dict(), "prompt_sequence": prompts.loc[r["stem"], "sequence"]}, args.variant)
             for r in records if r["stem"] in present]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score_one, tasks))
    output = ROOT / ("scratch/alphafold/smoke_scores" if args.smoke else "data")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k not in {"contacts", "contact_metrics", "timing"}} for r in results]).to_csv(output / f"{args.variant}_structure_metrics.csv", index=False)
    pd.DataFrame([dict(stem=r["stem"], method=args.variant, eval_set=r["eval_set"], **m)
                  for r in results for m in r["contact_metrics"]]).to_csv(output / f"{args.variant}_contact_metrics.csv", index=False)
    pd.DataFrame([r["timing"] for r in results]).to_csv(output / f"{args.variant}_timings.csv", index=False)
    raw = ROOT / "scratch/alphafold" / f"{args.variant}_contacts.json"
    raw.write_text(json.dumps({r["stem"]: r["contacts"] for r in results}))
    inputs = ROOT / "data/alphafold_inputs.json"
    (output / f"{args.variant}_run.json").write_text(json.dumps({
        "variant": args.variant, "n_targets": len(results),
        "n_candidates": sum(r["timing"]["n_samples"] for r in results),
        "protocol_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helico_metric_source": HELICO_SHA,
        "contact_source": "exp78 pyconfind_contacts; exp89 compute_metrics; exp245 frozen truth",
        "contact_truth_sha256": truth_sha,
        "scoring_packages": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "gemmi", "pyconfind", "tmtools", "scikit-learn")},
        "runtime_jax": sorted({r["timing"]["jax_version"] for r in results}),
        "runner": "Modal H100, us-east, up to eight resident workers per predictor",
        "selection": "Verified complete candidate budget, confidence argmax and selected coordinates (AF2 bytes; AF3 atom table excluding volatile header timestamps)",
        "timing_scope": "Inference calls include cold-shape JIT; exclude feature preparation and file writes",
        "selected_sha256": {r["stem"]: r["selected_sha256"] for r in results},
    }, indent=2) + "\n")
    print(f"Scored {args.variant}: {len(results)} proteins")


if __name__ == "__main__":
    main()
