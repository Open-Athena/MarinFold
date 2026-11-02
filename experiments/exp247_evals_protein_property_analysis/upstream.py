# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The seam onto everything exp247 reads. Nothing here is recomputed.

exp247 is pure analysis: every score it explains was produced by
[#245](https://github.com/Open-Athena/MarinFold/issues/245), and most of the
protein annotation it needs was produced there too. This module resolves those
inputs and the two external services the remaining features come from, so no
other file in the experiment hard-codes a path.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
WORK = Path("/data/exp247")

EXP245_DIR = EXPERIMENTS / "exp245_evals_foldbench_held_out_monomers"
EXP226_DIR = EXPERIMENTS / "exp226_evals_expand_foldbench_eval_set"
if not EXP245_DIR.is_dir():  # pragma: no cover - layout guard
    raise SystemExit(f"exp245 not found at {EXP245_DIR}; exp247 explains its scores")

#: #245's outputs.
PER_PROTEIN = EXP245_DIR / "data" / "per_protein.csv.gz"
EVAL_SETS = EXP245_DIR / "data" / "eval_sets.csv"
GT_MANIFEST = EXP245_DIR / "data" / "gt_manifest.csv"
GT_UNIVERSE = EXP245_DIR / "data" / "gt_universe_foldbench_monomers.jsonl"
CONTEXT_BUDGET = EXP245_DIR / "data" / "context_budget.csv"
RESIDUAL_IDENTITY = EXP245_DIR / "data" / "residual_identity.csv"

#: #226's identity table — per-protein homology to the *pre*-decontamination
#: corpora, per arm, which is the "training support" axis H1 is about.
IDENTITY_TABLE = EXP226_DIR / "data" / "eval_train_identity_expanded.csv"

#: #94's KNN run over the same proteins (#245 rebuilt it for all 333 units).
KNN_SUMMARY = Path("/data/exp245_knn_all/knn_hit_summary.csv")
KNN_ALIGNMENTS = Path("/data/exp245_knn_all/aln.m8")

#: The RCSB assembly mmCIFs #245 already downloaded — secondary structure and
#: shape features are read straight off these.
CIF_CACHE = Path("/data/exp245/cif")

#: The colabfold MSAs Protenix's +MSA arm was run with. Depth is a property of
#: the protein, so it is a feature here even though only two predictors use it.
MSA_VOLUME = "protenix-foldbench-msa"
MSA_PATH = "{stem}/msa/0/0/non_pairing.a3m"

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"

#: The predictors #245 scored, in the order every exp247 table uses.
PREDICTORS = (
    "#232 m2-p06 (decontaminated)",
    "#232 m1-p02 (decontaminated)",
    "#199 cooldown (contaminated)",
    "Protenix-v2 single-seq",
    "ESMFold",
    "ESMFold2",
    "Protenix-v2 + MSA",
    "seq-KNN (unfiltered corpus)",
    "seq-KNN (decontaminated corpus)",
)
MARINFOLD = "#232 m2-p06 (decontaminated)"
NATURAL_SETS = ("eval-val", "eval-test")
