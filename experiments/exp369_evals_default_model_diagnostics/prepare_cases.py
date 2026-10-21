"""Freeze diverse AFDB/PDB disagreement cases and plot the verified label conflict."""

import json
from pathlib import Path

import gemmi
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
WORK = Path("/data/exp369")
CASES = {
    "Q24GZ2": "Large assembly-bound discrepancy; centrosomal protein",
    "Q384U1": "Large assembly-bound discrepancy; dynein regulatory complex subunit 2",
    "A0A7M7RBC4": "Large assembly-bound discrepancy; microtubule-associated protein",
    "O00483": "Contact disagreement in complex IV; substantial backbone similarity",
    "Q89G85": "Membrane-transporter alternate-state control; not an established AFDB error",
    "P74102": "Photoactivated OCP state control; not an established AFDB error",
}


def main() -> None:
    """Prepare unconditioned paired-structure inference, without using labels in prompts."""
    cases = {case["uniprot"]: case for case in json.loads((WORK / "afdb_cases.json").read_text())}
    records, rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2), layout="constrained")
    for ax, (accession, interpretation) in zip(axes.flat, CASES.items()):
        case = cases[accession]
        block = gemmi.cif.read(str(WORK / "pdb_chains" / f"{case['pdb']}_{case['chain']}.cif")).sole_block()
        entity = block.find_values("_atom_site.label_entity_id")[0]
        entities = block.get_mmcif_category("_entity.")
        name = dict(zip(entities["id"], entities["pdbx_description"]))[entity]
        case = case | {"interpretation": interpretation, "protein_name": name}
        af = {tuple(pair) for pair in case["afdb_pairs"]}
        pdb = {tuple(pair) for pair in case["pdb_pairs"]}
        for pairs, color, label in ((af & pdb, "#636B74", "Shared"), (af - pdb, "#CC6B25", "AFDB only"), (pdb - af, "#167CA0", "PDB only")):
            ax.scatter([i for i, j in pairs], [j for i, j in pairs], s=11, color=color, label=label)
        ax.set_xlim(0, case["L"])
        ax.set_ylim(0, case["L"])
        ax.set_aspect("equal")
        ax.set_xlabel("Residue index")
        ax.set_ylabel("Residue index")
        ax.set_title(f"{accession} / {case['pdb'].upper()} chain {case['chain']}\nTM={case['tm_score_common_residues']:.2f}; contact F1={case['contact_f1']:.2f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        records.append(case | {"stem": f"{case['uniprot']}__{case['pdb']}_{case['chain']}", "dataset": "afdb_training_disagreement",
                               "truth": case["pdb_pairs"], "orders": {f"{policy}:{repeat}": [] for policy in ("random", "sse_balanced") for repeat in range(2)},
                               "doses_by_repeat": {str(repeat): [{"axis": "count", "dose": 0, "k": 0}] for repeat in range(2)}})
        rows.append({key: value for key, value in case.items() if key not in ("sequence", "resolved", "afdb_pairs", "pdb_pairs", "afdb_only", "pdb_only")})
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exact AFDB v4 training contacts versus later experimental structures\nUpper row: large assembly-bound discrepancies • lower row: contact/state controls", fontsize=13)
    save_plot_with_meta(fig, str(HERE / "plots/afdb_pdb_disagreements.png"),
                        caption="No MarinFold predictions yet. Contacts use the canonical pyconfind ≥0.001 and sequence-separation ≥6 definition on common resolved residues. AFDB contacts exactly reproduce the training documents. TM-score is structure-aligned over common residues; these assembly/state changes are not all unambiguous fold errors.")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(HERE / "data/afdb_selected_cases.csv", index=False)
    plan = {"schema": 1, "model": "contacts-v1-exp277-m2-p06-full-epoch-1.5B", "step": 266344,
            "eval_set": "afdb-training-disagreements", "n_rollouts": 100, "repeats": 2,
            "temperature": 1., "top_p": .95, "top_k": -1, "targets": records}
    (WORK / "afdb_inference_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # This small frozen target/label file is sufficient to rerun case inference.
    (HERE / "data/afdb_inference_plan.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
