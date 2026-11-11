"""Freeze eval-val targets, DSSP elements, nested contact orders, and dose grids."""

import argparse
import difflib
import hashlib
import json
import tempfile
from pathlib import Path

import gemmi
import mdtraj as md
import numpy as np
import pandas as pd

from protocol import REPEATS, ROLLOUTS, contact_order, coverage_milestones, doses

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "exp245_evals_foldbench_held_out_monomers/data"


def secondary_structure(cif: Path, chain_name: str, sequence: str) -> tuple[list[int], str]:
    """Map DSSP's contiguous H/E segments from the deposited chain to the input."""
    source = gemmi.read_structure(str(cif))
    chain = source[0][chain_name]
    cleaned = gemmi.Structure()
    cleaned.add_model(gemmi.Model("1"))
    cleaned[0].add_chain(gemmi.Chain("A"))
    residues = [res for res in chain if res.het_flag == "A" and res.find_atom("CA", "*")]
    for residue in residues:
        cleaned[0][0].add_residue(residue.clone())
    cleaned.remove_alternative_conformations()
    observed = "".join(gemmi.find_tabulated_residue(res.name).one_letter_code.upper() for res in residues)
    mapping = {}
    matcher = difflib.SequenceMatcher(a=observed, b=sequence, autojunk=False)
    for block in matcher.get_matching_blocks():
        mapping.update({block.a + offset: block.b + offset for offset in range(block.size)})
    if len(mapping) / len(observed) < .98:
        raise ValueError(f"Poor DSSP sequence mapping: {cif} chain {chain_name}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "chain.pdb"
        cleaned.write_pdb(str(path))
        trajectory = md.load_pdb(str(path))
        assignment = md.compute_dssp(trajectory, simplified=True)[0]
    if len(assignment) != len(observed):
        raise ValueError(f"DSSP residue count mismatch for {cif}")
    ss = ["-"] * len(sequence)
    for index, position in mapping.items():
        ss[position] = str(assignment[index])
    elements = [-1] * len(sequence)
    counter = -1
    previous = "-"
    for index, code in enumerate(ss):
        if code in ("H", "E"):
            if code != previous:
                counter += 1
            elements[index] = counter
        previous = code
    return elements, "".join(ss)


def main() -> None:
    """Prepare all 97 proteins without accessing eval-test predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-cache", type=Path, default=Path("/data/exp245/cif"))
    parser.add_argument("--out", type=Path, default=Path("/data/exp369/spiking_plan.json"))
    args = parser.parse_args()
    sets = pd.read_csv(DATA / "eval_sets.csv").query("eval_set == 'eval-val' and scorable == 1")
    wanted = set(sets.stem)
    truth = {r["stem"]: r for r in map(json.loads, (DATA / "gt_universe_scored.jsonl").read_text().splitlines()) if r["stem"] in wanted}
    targets = pd.read_parquet(DATA / "eval_targets_foldbench_monomers.parquet")
    targets = targets.loc[targets.stem.isin(wanted)].sort_values(["L", "stem"])
    if len(targets) != 97 or len(truth) != 97:
        raise ValueError("Expected exactly 97 eval-val units")
    records, summary = [], []
    for row in targets.itertuples(index=False):
        gt = truth[row.stem]
        cif = args.cif_cache / f"{row.stem.split('_')[0]}-assembly1.cif"
        elements, ss = secondary_structure(cif, gt["gt_chain"], row.input_seq)
        resolved = set(gt["resolved"])
        pairs = sorted({(int(i), int(j)) for i, j, degree in gt["contacts"]
                        if degree >= .001 and j - i >= 6 and i in resolved and j in resolved})
        pairs = [list(pair) for pair in pairs]
        orders = {f"{policy}:{repeat}": contact_order(pairs, elements, policy, f"exp369:{row.stem}:{repeat}:{policy}")
                  for policy in ("random", "sse_balanced") for repeat in range(REPEATS)}
        grids = {str(repeat): doses(len(pairs)) + coverage_milestones(orders[f"sse_balanced:{repeat}"], elements)
                 for repeat in range(REPEATS)}
        coverable = {elements[i] for pair in pairs for i in pair} - {-1}
        records.append({"dataset": row.dataset, "stem": row.stem, "L": int(row.L), "sequence": row.input_seq,
                        "truth": pairs, "resolved": gt["resolved"], "elements": elements, "dssp": ss,
                        "n_coverable_elements": len(coverable), "orders": orders, "doses_by_repeat": grids,
                        "cif_sha256": hashlib.sha256(cif.read_bytes()).hexdigest()})
        # Zero-dose baseline is generated once per repeat, shared by both policies.
        conditions = sum(1 + 2 * len({dose["k"] for dose in grid if dose["k"]}) for grid in grids.values())
        summary.append({"stem": row.stem, "L": row.L, "n_true": len(pairs), "n_elements": len(set(elements) - {-1}),
                        "n_coverable_elements": len(coverable), "unique_counts": len({dose["k"] for grid in grids.values() for dose in grid}),
                        "conditions": conditions, "completions": conditions * ROLLOUTS})
    plan = {"schema": 1, "model": "contacts-v1-exp277-m2-p06-full-epoch-1.5B", "step": 266344,
            "eval_set": "eval-val", "n_rollouts": ROLLOUTS, "repeats": REPEATS,
            "temperature": 1.0, "top_p": .95, "top_k": -1, "generation_tokens": "6L+128",
            "context_tokens": 8192, "targets": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, separators=(",", ":")) + "\n")
    pd.DataFrame(summary).to_csv(HERE / "data/spiking_targets.csv", index=False)
    manifest = {"plan_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "n_proteins": len(records), "n_conditions": sum(r["conditions"] for r in summary),
                "n_completions": sum(r["completions"] for r in summary), "plan_bytes": args.out.stat().st_size,
                "mdtraj_version": md.__version__, "gemmi_version": gemmi.__version__,
                "source_files": {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (DATA / "eval_sets.csv", DATA / "gt_universe_scored.jsonl", DATA / "eval_targets_foldbench_monomers.parquet")}}
    (HERE / "data/spiking_plan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
