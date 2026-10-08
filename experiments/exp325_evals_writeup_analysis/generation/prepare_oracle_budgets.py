"""Freeze oracle subsets for the exact 305-protein Figure 02 population.

Run with the existing pinned Helico environment:
uv run --project /home/bizon/git/helico --no-sync python generation/prepare_oracle_budgets.py
"""

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

from oracle_budgets import BUDGETS, REPLICATES, build_maps, map_keys, random_seed, requested_count

ROOT = Path(__file__).resolve().parent.parent
REPO = Path("/home/bizon/git/helico")
HELICO_SHA = "b10385d736673c81b10e70d1099962af6f2573c0"
DESTINATION = ROOT / "scratch/helico/oracle_budget"


def main() -> None:
    """Extract once, validate every conditioning map, and record exact input hashes."""
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != HELICO_SHA:
        raise ValueError("Helico source differs from the frozen comparison")
    rows = pd.read_csv(ROOT / "data/figure_rows.csv")
    cohort = rows[(rows.figure == "02_oracle") & (rows.method == "oracle") &
                  (rows.metric == "gdt_ts") & (rows.designed == 0)]
    if len(cohort) != 305 or cohort.stem.duplicated().any():
        raise ValueError("Expected the 305 natural proteins in Figure 02")
    targets = pd.read_csv(REPO / "experiments/exp14_foldbench_held_out_monomers/data/targets.csv")
    targets = targets[targets.stem.isin(cohort.stem)].merge(
        cohort[["stem", "msa_depth", "tier"]], on="stem", validate="one_to_one")
    if set(targets.stem) != set(cohort.stem):
        raise ValueError("Missing frozen structural inputs")
    targets = targets.sort_values(["msa_depth", "stem"])
    (DESTINATION / "maps").mkdir(parents=True, exist_ok=True)
    (DESTINATION / "gt").mkdir(exist_ok=True)
    ccd, rotamers = parse_ccd(), load_rotamer_library()
    records, pins = [], {}
    for target in targets.to_dict("records"):
        stem = target["stem"]
        source = Path.home() / ".cache/helico/data/benchmarks/FoldBench/examples/ground_truths" / f"{target['pdb_id']}.cif.gz"
        path = DESTINATION / "gt" / f"{stem}.cif.gz"
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
        maps = build_maps(oracle, stem, int(target["L_exp245"]))
        map_path = DESTINATION / "maps" / f"{stem}.npz"
        np.savez_compressed(map_path, **maps)
        pins[stem] = {"gt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "maps_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest()}
        for arm, replicate in map_keys():
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
    targets.to_csv(DESTINATION / "targets.csv", index=False)
    targets.to_csv(ROOT / "data/oracle_budget_targets.csv", index=False)
    (DESTINATION / "ranked_pairs.json").write_text("{}\n")
    pd.DataFrame(records).to_csv(ROOT / "data/oracle_budget_maps.csv", index=False)
    protocol = dict(phase="oracle_budget", source_sha=HELICO_SHA, n_targets=len(targets),
        n_maps_per_target=len(map_keys()), n_random_replicates=REPLICATES,
        budgets=list(BUDGETS), map_keys=[f"{a}-{r}" for a, r in map_keys()],
        contact_definition="Helico oracle_contact_state; pyconfind 0.6.0 native_only, 3A, degree >=0.001, separation >=6",
        sampling="Uniform without replacement among upper-triangle true contacts; nested budget prefixes within each replicate",
        sparse_encoding="Selected pairs PRESENT=2, every other pair UNKNOWN=0; no ABSENT=1 entries",
        controls="Fresh top_0, positive_all (no non-contacts), and oracle (full positives and negatives)",
        length="L_exp245: frozen input sequence length, not the number of resolved tokens",
        shortage="Relative L/5 and L/2 budgets cap at all available true contacts; requested/effective counts are saved. Fixed 5/10 must be exact.",
        inference="Same MSA-free step6000 checkpoint; three diffusion samples, six recycles, seed42 per map",
        selection="Main comparison: highest ranking_score among three samples, matching Figure 02. pTM selection saved as a sensitivity analysis. Average the two random subsets within each protein; never select the best subset.",
        files_sha256=pins)
    (DESTINATION / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (ROOT / "data/oracle_budget_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
