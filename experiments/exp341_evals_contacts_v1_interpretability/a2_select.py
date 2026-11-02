# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""A2 protein set: eval-val, eval-denovo and 200 legacy-554 proteins, with covariates.

The plan (section 5.2) runs A2 on eval-val, eval-denovo and 200 legacy-554
proteins. The legacy 200 exclude ``foldbench100`` (its natural monomers are
eval-val), include all 37 proteins exp213 classed as novel fold with no
training homolog (where MarinFold trails Protenix single-sequence by 0.22
and cannot be recalling anything), and fill the rest by seeded random draw.

Covariates per protein: MSA depth (``msa_neff``; exp260's ``msa_depth.csv``
``n_eff_0.8`` from its ``foldbench`` MSA set for eval proteins, exp213's ``msa_neff`` for legacy;
three eval-denovo designs have no FoldBench MSA and are left empty), and
exp213/exp226's ``fold_verdict``, ``best_identity_covered`` and
``n_hits_significant``. Those training-homology columns are measured against
exp199's contaminated corpora. #225 decontaminated exp277's corpora against
all of these proteins, so for exp277 their training homology is near zero by
construction; A2 here tests contact dependence at zero training homology
across MSA depth, and the training-homology axis comes from the A4 subset on
the held-out panel.

``est_l40s_seconds`` is the expected one-seed A2 cost: one clean readout
plus up to four mutant readouts per residue, at a per-readout time fitted
to harness check 5's L40S timings.

Writes ``data/a2_proteins.csv`` and ``data/a2_legacy_gt.jsonl.gz`` (exp89
ground truth plus exp213 sequence for the 200 legacy proteins).

    uv run python a2_select.py --legacy-gt gt_universe.jsonl
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXPERIMENTS = HERE.parent
N_LEGACY = 200
SEED = 341


def read_fasta(path: Path) -> dict[str, str]:
    sequences, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            sequences[name] = ""
        elif name is not None:
            sequences[name] += line.strip()
    return sequences


def readout_seconds(lengths: np.ndarray) -> np.ndarray:
    """Per-readout, per-seed L40S seconds, quadratic in L, from check 5's timings."""
    timings = pd.read_csv(DATA / "readout_l40s" / "timings.csv")
    per_seed = timings["elapsed_seconds"] / timings["n_seeds"]
    return np.polyval(np.polyfit(timings["n_residues"], per_seed, 2), lengths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-gt", type=Path, required=True,
                        help="exp89 gt_universe.jsonl (data/contacts-v1-model-eval-exp89/ on the HF bucket)")
    args = parser.parse_args()

    exp213 = EXPERIMENTS / "exp213_evals_train_sequence_overlap_audit" / "data"
    covariates = ["fold_verdict", "best_identity_covered", "n_hits_significant"]

    sets = pd.read_csv(EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers" / "data" / "eval_sets.csv")
    evals = sets[sets["eval_set"].isin(["eval-val", "eval-denovo"]) & (sets["scorable"] == 1)]
    identity = pd.read_csv(EXPERIMENTS / "exp226_evals_expand_foldbench_eval_set" / "data" / "eval_train_identity_expanded.csv")
    depth = pd.read_csv(EXPERIMENTS / "exp260_evals_msa_depth_stratified" / "data" / "msa_depth.csv")
    # exp260 searched two MSA sets; eval proteins use FoldBench's own sequences.
    depth = depth[depth["msa_volume"] == "foldbench"]
    evals = (
        evals[["eval_set", "stem", "seq_len"]]
        .merge(identity[["stem", *covariates]].drop_duplicates("stem"), on="stem", how="left", validate="one_to_one")
        .merge(depth[["stem", "n_eff_0.8"]].rename(columns={"n_eff_0.8": "msa_neff"}), on="stem", how="left",
               validate="one_to_one")
        .rename(columns={"eval_set": "set", "seq_len": "L"})
    )
    evals["key"] = evals["stem"]

    legacy = pd.read_csv(exp213 / "eval_train_identity.csv")
    legacy["key"] = legacy["dataset"] + "__" + legacy["stem"]
    pool = legacy[legacy["dataset"] != "foldbench100"]
    novel = (pool["fold_verdict"] == "novel_fold") & (pool["stratum"] == "no_homolog")
    rng = np.random.default_rng(SEED)
    rest = pool[~novel]
    fill = rest.iloc[np.sort(rng.choice(len(rest), size=N_LEGACY - int(novel.sum()), replace=False))]
    chosen = pd.concat([pool[novel], fill])
    chosen = chosen.assign(set="legacy-" + chosen["dataset"], L=chosen["length"], novel_no_homolog=novel[chosen.index])
    evals["novel_no_homolog"] = False

    proteins = pd.concat([
        evals[["set", "key", "L", "msa_neff", *covariates, "novel_no_homolog"]],
        chosen[["set", "key", "L", "msa_neff", *covariates, "novel_no_homolog"]],
    ], ignore_index=True)
    if proteins["key"].duplicated().any():
        raise ValueError("duplicate protein keys")
    missing = proteins.loc[proteins["msa_neff"].isna(), "key"].tolist()
    if missing:
        print(f"no MSA depth (kept, excluded from depth regressions): {missing}")
    proteins["est_l40s_seconds"] = (1 + 4 * proteins["L"]) * readout_seconds(proteins["L"].to_numpy())

    sequences = read_fasta(exp213 / "eval_queries.fasta")
    ground_truth = {}
    with args.legacy_gt.open() as handle:
        for line in handle:
            record = json.loads(line)
            ground_truth[f"{record['dataset']}__{record['stem']}"] = record
    DATA.mkdir(exist_ok=True)
    with gzip.open(DATA / "a2_legacy_gt.jsonl.gz", "wt") as out:
        for key in chosen["key"]:
            record = ground_truth[key]
            out.write(json.dumps({
                "key": key, "dataset": record["dataset"], "stem": record["stem"], "L": record["L"],
                "sequence": sequences[key], "resolved": record["resolved"], "contacts": record["contacts"],
            }) + "\n")
    proteins.to_csv(DATA / "a2_proteins.csv", index=False, float_format="%.6g")

    print(proteins.groupby("set").agg(n=("key", "size"), median_L=("L", "median"),
                                      l40s_hours=("est_l40s_seconds", lambda s: s.sum() / 3600)).round(1))
    print(f"novel-fold no-homolog: {int(proteins['novel_no_homolog'].sum())}; "
          f"total {proteins['est_l40s_seconds'].sum() / 3600:.1f} L40S-hours at one seed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
