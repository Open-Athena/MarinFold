# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream inputs for exp260.

Everything this experiment reads is either a published, digest-pinned file on
the public HF bucket or a Modal volume of ColabFold MSAs that already exists.
Nothing here rebuilds ground truth, and nothing needs credentials to read.

The two MSA volumes were written by the same Protenix pipeline call —
``runner.msa_search.update_seq_msa(infer_data, target_dir, mode="colabfold")``
in ``exp12_data_protenix_foldbench_monomers/modal_app.py`` and
``exp74_evals_protenix_pyconfind_contacts/modal_app.py`` — so a depth measured
in one is comparable with a depth measured in the other. They were populated
about three weeks apart (2026-05 and 2026-06), and ColabFold's databases grow,
so :mod:`msa_depth_modal` re-measures the 11 stems both volumes hold to bound
how much that matters.
"""

from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
DATA = EXPERIMENT / "data"
PLOTS = EXPERIMENT / "plots"
EXPERIMENTS = EXPERIMENT.parent
EXP74 = EXPERIMENTS / "exp74_evals_protenix_pyconfind_contacts"

BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"

#: The legacy 554-unit target table (#169's copy of the #89 universe). Carries
#: the ``cameo_hard`` / ``casp_fm`` / ``denovo_pdb`` / ``foldbench100`` dataset
#: labels this experiment splits FoldBench from non-FoldBench on.
LEGACY_TARGETS_URL = f"{BUCKET}/data/contacts-v1-model-eval-exp169/eval_targets.parquet"
LEGACY_TARGETS_SHA256 = (
    "9de9bc1b99b7e7ab6d2b17a985f9e22bc7decd2b25e1b16be30dea921431c111"
)

#: #245's manifest of all 334 FoldBench monomers: eval set, designed flag,
#: viral flag, kingdom, and scorability.
FOLDBENCH_PREFIX = "data/contacts-v1-foldbench-monomers-exp245"
#: The scorable-unit target table: ``dataset``, ``stem``, ``L``, ``input_seq``.
FOLDBENCH_TARGETS_URL = (
    f"{BUCKET}/{FOLDBENCH_PREFIX}/eval_targets_foldbench_monomers.parquet"
)
FOLDBENCH_TARGETS_SHA256 = (
    "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8"
)

FOLDBENCH_SETS_URL = f"{BUCKET}/{FOLDBENCH_PREFIX}/eval_sets.csv"
FOLDBENCH_SETS_SHA256 = (
    "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861"
)

#: The two published ground-truth universes, in #89's schema: per-unit ``L``,
#: ``resolved`` indices, and ``contacts`` as ``[i, j, degree]``.
LEGACY_GROUND_TRUTH_URL = (
    f"{BUCKET}/data/contacts-v1-model-eval-exp89/gt_universe.jsonl"
)
FOLDBENCH_GROUND_TRUTH_URL = f"{BUCKET}/{FOLDBENCH_PREFIX}/gt_universe_scored.jsonl"

#: #245's per-protein baseline scores over the 333 scorable FoldBench units:
#: Protenix-v2 single-seq and +MSA, ESMFold, ESMFold2, and both KNN nulls.
FOLDBENCH_PER_PROTEIN_URL = f"{BUCKET}/{FOLDBENCH_PREFIX}/per_protein.csv.gz"

#: This experiment's own published results (written by the CoreWeave driver).
PUBLISH_PREFIX = "data/contacts-v1-msa-depth-exp260"
RUN_ID = "v1-01"
RESULTS_URL = f"{BUCKET}/{PUBLISH_PREFIX}/{RUN_ID}"

#: Modal volumes of ColabFold MSAs, and the a3m path inside each.
MSA_VOLUMES = {
    "foldbench": "protenix-foldbench-msa",
    "exp74": "protenix-exp74-msa",
}
MSA_PATH = "{stem}/msa/0/0/non_pairing.a3m"

#: Legacy datasets that are natural proteins. ``denovo_pdb`` is the PDB
#: ``DE NOVO PROTEIN`` class and ``foldbench100`` is scored through the
#: FoldBench universe instead, where the eval-set labels live.
NONFOLDBENCH_NATURAL_DATASETS = ("cameo_hard", "casp_fm")

#: MSA-depth tiers, as requested: shallow, thin, moderate, deep. Bins are
#: [lo, hi) on the raw ColabFold sequence count.
DEPTH_TIERS = (
    ("<10", 0, 10),
    ("10-100", 10, 100),
    ("100-1000", 100, 1000),
    ("1000+", 1000, None),
)
