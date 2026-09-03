# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned inputs for the exp260 MSA-depth evaluation.

Derived from ``experiments/exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2``
(PR #257) with two changes: ``eval-test`` joins the scored universe, taking it
from 670 to 887 units, and the E8 reference checkpoint is no longer scored
alongside. PR #257's own published aggregates take over as the reproduction
gate — the same weights and the same worker have to land on the same legacy-554,
``eval-val``, and ``eval-denovo`` numbers before the new ``eval-test`` slice is
believable. The checkpoint itself is read in place from the CoreWeave HF export
PR #257 created; nothing is re-exported or copied.
"""

from __future__ import annotations

from dataclasses import dataclass

MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"
MARIN_PREFIX = "s3://marin-us-east-02a/marin"
S3_ROOT = (
    f"{MARIN_PREFIX}/protein-structure/MarinFold/"
    "exp260_evals_msa_depth_stratified/rollout-v2/2026-08-31"
)

#: Where the published copies of the small result artifacts land. The CoreWeave
#: results prefix is not readable from a workstation, so the driver pushes the
#: aggregates, per-protein rows, timings, and manifest to the public bucket and
#: everything downstream reads them anonymously over HTTPS.
PUBLISH_BUCKET = "open-athena/MarinFold"
PUBLISH_PREFIX = "data/contacts-v1-msa-depth-exp260"

#: PR #257's run root. The checkpoint under test is the HF export that run
#: produced, read in place: this evaluation writes nowhere near it.
EXP232_RUN_ROOT = (
    f"{MARIN_PREFIX}/protein-structure/MarinFold/"
    "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01"
)

# Published evaluation inputs. The driver mirrors these small immutable files
# into the run prefix, verifies their bytes, and builds an 887-unit union without
# deduplicating either source: 554 legacy units plus all 333 scorable FoldBench
# monomers (eval-val 97, eval-test 217, eval-denovo 19). 112 stems appear in both
# universes and stay separate evaluation units.
BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
LEGACY_TARGETS_URL = f"{BUCKET}/data/contacts-v1-model-eval-exp169/eval_targets.parquet"
LEGACY_TARGETS_SIZE = 43_077
LEGACY_TARGETS_SHA256 = (
    "9de9bc1b99b7e7ab6d2b17a985f9e22bc7decd2b25e1b16be30dea921431c111"
)
LEGACY_GROUND_TRUTH_URL = (
    f"{BUCKET}/data/contacts-v1-model-eval-exp89/gt_universe.jsonl"
)
LEGACY_GROUND_TRUTH_SIZE = 7_956_102
LEGACY_GROUND_TRUTH_SHA256 = (
    "3ff6eb4e383582595ad6f9811c77e2839ebcc0030a050b9c1f15d020163331c9"
)
FOLDBENCH_PREFIX = "data/contacts-v1-foldbench-monomers-exp245"
FOLDBENCH_TARGETS_URL = (
    f"{BUCKET}/{FOLDBENCH_PREFIX}/eval_targets_foldbench_monomers.parquet"
)
FOLDBENCH_TARGETS_SIZE = 97_519
FOLDBENCH_TARGETS_SHA256 = (
    "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8"
)
FOLDBENCH_GROUND_TRUTH_URL = f"{BUCKET}/{FOLDBENCH_PREFIX}/gt_universe_scored.jsonl"
FOLDBENCH_GROUND_TRUTH_SIZE = 6_938_887
FOLDBENCH_GROUND_TRUTH_SHA256 = (
    "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5"
)
FOLDBENCH_SETS_URL = f"{BUCKET}/{FOLDBENCH_PREFIX}/eval_sets.csv"
FOLDBENCH_SETS_SIZE = 199_548
FOLDBENCH_SETS_SHA256 = (
    "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861"
)

EVAL_SETS = ("eval-val", "eval-test", "eval-denovo")
EXPECTED_SET_SIZES = {
    "legacy_554": 554,
    "eval-val": 97,
    "eval-test": 217,
    "eval-denovo": 19,
}
EXPECTED_UNITS = 887
EXPECTED_UNIQUE_STEMS = 773
EXPECTED_OVERLAPPING_STEMS = 112
#: Viral / non-viral partitions of each FoldBench set, from ``eval_sets.csv``.
#: #241 found the viral penalty is the sharpest homology-dependence signal we
#: have, so the split is reported, not just counted.
EXPECTED_VIRAL_SPLIT = {
    "eval-val-nonviral": 91,
    "eval-val-viral": 6,
    "eval-test-nonviral": 204,
    "eval-test-viral": 13,
    "eval-denovo-nonviral": 19,
}

#: PR #257's published aggregates for the checkpoint under test, keyed by
#: (subset, range, cut). This run re-scores those three subsets with the same
#: weights and the same worker, so they are a reproduction gate rather than a
#: result: a disagreement means the execution path changed, not the model.
#: Source: experiments/exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2/
#: data/coreweave_results/subset_aggregate_metrics.csv
PUBLISHED_REFERENCE_METRICS = {
    ("legacy_554", "all", "R"): 0.605059,
    ("legacy_554", "all", "AUC"): 0.945625,
    ("legacy_554", "long", "R"): 0.555022,
    ("legacy_554", "long", "AUC"): 0.931401,
    ("eval-val", "all", "R"): 0.551707,
    ("eval-val", "all", "AUC"): 0.936367,
    ("eval-val", "long", "R"): 0.535909,
    ("eval-val", "long", "AUC"): 0.923458,
    ("eval-denovo", "all", "R"): 0.609832,
    ("eval-denovo", "all", "AUC"): 0.964735,
    ("eval-denovo", "long", "R"): 0.572282,
    ("eval-denovo", "long", "AUC"): 0.954809,
}
#: #204 measured a 0.0023 span across four evaluations of one unchanged
#: checkpoint; 0.005 is the tolerance the eval-checkpoint skill fixes for a
#: rollout reproduction.
PUBLISHED_REFERENCE_TOLERANCE = 0.005
PUBLISHED_REFERENCE_SOURCE = (
    "experiments/exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2/"
    "data/coreweave_results/subset_aggregate_metrics.csv"
    ":marinfold-exp232-decontam-train-m2-p06-step363000 (PR #257)"
)

E8_REFERENCE_METRICS = {
    ("all", "R"): 0.4245291213628376,
    ("long", "R"): 0.3656151868856005,
    ("all", "AUC"): 0.9009633507400637,
    ("long", "AUC"): 0.8737803746877988,
}
E8_REFERENCE_TOLERANCE = 0.005


@dataclass(frozen=True)
class HfFile:
    """One immutable file in a Hugging Face checkpoint export."""

    name: str
    size: int
    digest: str
    digest_kind: str


@dataclass(frozen=True)
class Checkpoint:
    """One checkpoint evaluated from an existing CoreWeave HF directory."""

    label: str
    job_label: str
    run_name: str
    step: int
    checkpoint_files: tuple[HfFile, ...]
    weight_shard_digests: tuple[str, str]
    source_dtype: str
    coreweave_uri: str
    levanter_source_uri: str | None = None
    levanter_source_objects: int | None = None
    levanter_source_bytes: int | None = None
    levanter_source_manifest_sha256: str | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    eval_loss_step: int | None = None
    accepted_unfinished_rollouts: int = 0

    @property
    def files(self) -> tuple[HfFile, ...]:
        return self.checkpoint_files


E8_REFERENCE_CHECKPOINT = Checkpoint(
    label="e8_reference_step35679",
    job_label="e8ref",
    run_name="prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084",
    step=35_679,
    checkpoint_files=(
        HfFile(
            "config.json", 2_498, "2c84e510f70f1e1265ade774eda02f0e6da21be0", "git-sha1"
        ),
        HfFile(
            "model-00001-of-00002.safetensors",
            4_979_485_528,
            "0be51806a5ecbcbd4a7e2824c2c687a56e4bf0d5861db40a6432714270ccf50a",
            "sha256",
        ),
        HfFile(
            "model-00002-of-00002.safetensors",
            906_042_048,
            "67cf32f6959292aaea53de2082d83f39af87a829237660fdbc74ce9af960451e",
            "sha256",
        ),
        HfFile(
            "model.safetensors.index.json",
            20_882,
            "9880be895e6d9c514b62ed263640d46f67d01a29",
            "git-sha1",
        ),
        HfFile(
            "special_tokens_map.json",
            417,
            "d6a4b943a77aba8a2f7e51e6e174a6cf241cdd52",
            "git-sha1",
        ),
        HfFile(
            "tokenizer.json",
            64_026,
            "6e696c75de792ee564e93bfdc30871022a07ad75",
            "git-sha1",
        ),
        HfFile(
            "tokenizer_config.json",
            785,
            "d2eb7b23f57945904b33a95341ba86122efea48f",
            "git-sha1",
        ),
    ),
    weight_shard_digests=(
        "0be51806a5ecbcbd4a7e2824c2c687a56e4bf0d5861db40a6432714270ccf50a",
        "67cf32f6959292aaea53de2082d83f39af87a829237660fdbc74ce9af960451e",
    ),
    source_dtype="float32",
    coreweave_uri="s3://marin-us-east-02a/MarinFold/exp163/model/step-35679",
)

# The source identity and six-file HF manifest were pinned after the successful
# in-CoreWeave export job ``/eczech/exp232-export-train-step363000-v2-01-r1``.
TRAIN_CHECKPOINT = Checkpoint(
    label="exp232_decontam_train_m2_p06_step363000",
    job_label="m2p06tr",
    run_name=(
        "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1"
    ),
    step=363_000,
    checkpoint_files=(
        HfFile("config.json", 1_557, "d8e904f8170ddf00d74c864f31d258a4", "s3-etag"),
        HfFile(
            "model-00001-of-00002.safetensors",
            4_979_485_528,
            "9a11736b507565aa2be00a5753f51b12-95",
            "s3-etag",
        ),
        HfFile(
            "model-00002-of-00002.safetensors",
            906_042_048,
            "1a042bf9b4acde490f0c0ffee76306dd-18",
            "s3-etag",
        ),
        HfFile(
            "model.safetensors.index.json",
            20_882,
            "bc0a5fd2c9aae096abae4caf9040c79c",
            "s3-etag",
        ),
        HfFile(
            "tokenizer.json",
            64_407,
            "c4b3a16978e30eb150cca4fd8934b6ae",
            "s3-etag",
        ),
        HfFile(
            "tokenizer_config.json",
            290,
            "336f4e2ca951fa13a20cb1c4b68b2040",
            "s3-etag",
        ),
    ),
    weight_shard_digests=(
        "9a11736b507565aa2be00a5753f51b12-95",
        "1a042bf9b4acde490f0c0ffee76306dd-18",
    ),
    source_dtype="float32",
    coreweave_uri=(
        f"{EXP232_RUN_ROOT}/models/exp232-decontam-train-m2-p06-step363000"
        "/hf/step-363000"
    ),
    levanter_source_uri=(
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
        "checkpoints/protein/"
        "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-"
        "lr005-us-east1/2026.08.21.1/checkpoints/step-363000"
    ),
    levanter_source_objects=175,
    levanter_source_bytes=17_659_722_031,
    levanter_source_manifest_sha256=(
        "7e0c6f650fe6c76a5570695c24e447ddad4ae0def6662371c17b3ad1fd656b37"
    ),
    eval_loss=2.9680745601654053,
    eval_loss_step=361_494,
)

CHECKPOINTS = (TRAIN_CHECKPOINT,)
#: ``training`` is what this experiment runs. ``validation`` keeps the E8
#: reference reachable for a from-scratch path check; PR #257 already passed it
#: on this cluster with this worker, so it is not repeated here.
CHECKPOINT_SUITES = {
    "training": CHECKPOINTS,
    "validation": (E8_REFERENCE_CHECKPOINT,),
    "both": (E8_REFERENCE_CHECKPOINT, TRAIN_CHECKPOINT),
}


def run_root(run_id: str) -> str:
    """Return the isolated S3 prefix for one execution attempt."""

    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    ):
        raise ValueError(f"invalid run id: {run_id!r}")
    return f"{S3_ROOT}/{run_id}"


def checkpoint_model_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return the pre-existing CoreWeave HF directory used by workers."""

    del run_id
    return checkpoint.coreweave_uri
