"""Freeze oracle subsets for Figure 02's five natural proteins at MSA depth <10.

Run with the existing pinned Helico environment:
uv run --project /home/bizon/git/helico --no-sync python generation/prepare_oracle_budgets.py
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from helico.bench import oracle_contact_state, structure_to_chains
from helico.contacts import load_rotamer_library
from helico.data import parse_ccd, parse_mmcif, tokenize_sequences

from oracle_budgets import BUDGETS, CONTROLS, REPLICATES, build_maps, random_seed, requested_count

ROOT = Path(__file__).resolve().parent.parent
REPO = Path("/home/bizon/git/helico")
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"


def main() -> None:
    """Extract once, validate every conditioning map, and record exact input hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cohort-l2", action="store_true", help="Only L/2, all 305 natural comparison proteins")
    args = parser.parse_args()
    phase = "oracle_l2" if args.full_cohort_l2 else "oracle_budget"
    destination = ROOT / "scratch/helico" / (phase if args.full_cohort_l2 else "oracle_budget_low_msa")
    budgets = ("random_L2",) if args.full_cohort_l2 else BUDGETS
    controls = () if args.full_cohort_l2 else CONTROLS
    keys = [(arm, r) for r in range(REPLICATES) for arm in budgets] + [(arm, 0) for arm in controls]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico source differs from the frozen comparison")
    rows = pd.read_csv(ROOT / "data/figure_rows.csv")
    cohort = rows[(rows.figure == "01_predictors") & (rows.method == "af3") &
                  (rows.metric == "gdt_ts") & (rows.designed == 0)]
    if not args.full_cohort_l2:
        cohort = cohort[cohort.msa_depth < 10]
    if len(cohort) != (305 if args.full_cohort_l2 else 5) or cohort.stem.duplicated().any():
        raise ValueError("Unexpected natural comparison cohort")
    targets = pd.read_csv(REPO / "experiments/exp14_foldbench_held_out_monomers/data/targets.csv")
    targets = targets[targets.stem.isin(cohort.stem)].merge(
        cohort[["stem", "msa_depth", "tier"]], on="stem", validate="one_to_one")
    if set(targets.stem) != set(cohort.stem):
        raise ValueError("Missing frozen structural inputs")
    targets = targets.sort_values(["msa_depth", "stem"])
    (destination / "maps").mkdir(parents=True, exist_ok=True)
    (destination / "gt").mkdir(exist_ok=True)
    ccd, rotamers = parse_ccd(), load_rotamer_library()
    records, pins = [], {}
    for target in targets.to_dict("records"):
        stem = target["stem"]
        source = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / f"{target['pdb_id']}.cif.gz"
        path = destination / "gt" / f"{stem}.cif.gz"
        shutil.copyfile(source, path)
        gt = parse_mmcif(path, max_resolution=float("inf"))
        if gt is None:
            raise ValueError(f"Cannot parse {stem}")
        chains = structure_to_chains(gt)
        protein = [chain for chain in chains if chain["type"] == "protein"]
        if len(protein) != 1 or protein[0]["sequence"] != target["input_seq"]:
            raise ValueError(f"{stem}: sequence differs from frozen inputs")
        tokenized = tokenize_sequences(chains, ccd)
        oracle_tensor = oracle_contact_state(gt, tokenized, rotamers)
        if oracle_tensor is None or tokenized.n_tokens > 2048:
            raise ValueError(f"{stem}: oracle/tokenization failed")
        oracle = oracle_tensor.numpy()
        maps = build_maps(oracle, stem, int(target["L_exp245"]), budgets=budgets, controls=controls)
        map_path = destination / "maps" / f"{stem}.npz"
        np.savez_compressed(map_path, **maps)
        pins[stem] = {"gt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "maps_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest()}
        for arm, replicate in keys:
            state = maps[f"{arm}-{replicate}"]
            n_present = int(np.triu(state == 2, 1).sum())
            records.append(dict(stem=stem, arm=arm, map_seed=replicate,
                sampling_seed=str(random_seed(stem, replicate)) if arm in BUDGETS else "",
                L=int(target["L_exp245"]), n_tokens=tokenized.n_tokens,
                requested_contacts=requested_count(arm, int(target["L_exp245"])) if arm in BUDGETS else n_present,
                n_present=n_present, n_absent=int(np.triu(state == 1, 1).sum()),
                n_unknown=int(np.triu(state == 0, 1).sum()),
                state_sha256=hashlib.sha256(state.tobytes()).hexdigest(), **pins[stem]))
        print(f"{stem}: {len(maps)} maps; {int(np.triu(oracle == 2, 1).sum())} true contacts", flush=True)
    targets.to_csv(destination / "targets.csv", index=False)
    targets.to_csv(ROOT / f"data/{phase}_targets.csv", index=False)
    (destination / "ranked_pairs.json").write_text("{}\n")
    pd.DataFrame(records).to_csv(ROOT / f"data/{phase}_maps.csv", index=False)
    protocol = dict(phase=phase, cohort="All 305 natural structure-comparison proteins" if args.full_cohort_l2 else "All five natural Figure 02 proteins with MSA depth <10",
        source_sha=HELICO_SHA, n_targets=len(targets),
        n_maps_per_target=len(keys), n_random_replicates=REPLICATES,
        budgets=list(budgets), map_keys=[f"{a}-{r}" for a, r in keys],
        contact_definition="Helico oracle_contact_state; pyconfind 0.6.0 native_only, 3A, degree >=0.001, separation >=6",
        sampling="Uniform without replacement among upper-triangle true contacts; nested budget prefixes within each replicate",
        sparse_encoding="Selected pairs PRESENT=2, every other pair UNKNOWN=0; no ABSENT=1 entries",
        controls="None; only random L/2" if args.full_cohort_l2 else "Fresh top_0, positive_all (no non-contacts), and oracle (full positives and negatives)",
        length="L_exp245: frozen input sequence length, not the number of resolved tokens",
        shortage="Relative L/5 and L/2 budgets cap at all available true contacts; requested/effective counts are saved. Fixed 5/10 must be exact.",
        inference="Same MSA-free step6000 checkpoint; three diffusion samples, six recycles, seed42 per map",
        selection="Main comparison: highest ranking_score among three samples, matching Figure 02. pTM selection saved as a sensitivity analysis. Average the two random subsets within each protein; never select the best subset.",
        files_sha256=pins)
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (ROOT / f"data/{phase}_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
