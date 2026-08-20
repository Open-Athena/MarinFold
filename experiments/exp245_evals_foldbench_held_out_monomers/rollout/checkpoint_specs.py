# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable identities for the exp245 FoldBench-monomer evaluation.

Three checkpoints are scored over the same 334 units:

* the two decontaminated #232 finals PR #244 selected, whose training corpora
  are verified free of every one of these proteins at #225's 30 % rule, and
* the current default model -- #199's CoreWeave cooldown -- whose corpora were
  never filtered against FoldBench. It is the contamination contrast: the same
  recipe, the same architecture, trained on data that contains these proteins'
  homologs.

The checkpoint identities (paths, file sizes, S3 etags, source dtype, losses)
are copied unchanged from the two evaluations that pinned them -- PR #244's
``exp232_sweep_cv1_decontam/evals/rollout_v2/checkpoint_specs.py`` for the #232
pair and ``exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2`` for the
cooldown -- so all three are verified in place against the same manifests those
runs verified them against. ``test_rollout.py`` asserts the copies still match.

No checkpoint is copied anywhere: every one is read from its existing CoreWeave
S3 location by jobs running in the same region.
"""

from dataclasses import dataclass

MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"
MARIN_PREFIX = "s3://marin-us-east-02a/marin"
S3_ROOT = (
    f"{MARIN_PREFIX}/protein-structure/MarinFold/"
    "exp245_foldbench_held_out_monomers/evals/rollout"
)

# --- evaluation inputs, published by this experiment ------------------------
#
# The 334-unit FoldBench monomer universe and its ground truth. Both are public
# on the bucket and are mirrored into CoreWeave S3 by the driver, checked
# against the size and digest pinned here so the job cannot silently score a
# different eval set. Regenerate with `publish_eval_inputs.py --print-pins`.

BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
BUCKET_PREFIX = "data/contacts-v1-foldbench-monomers-exp245"

TARGETS_URL = f"{BUCKET}/{BUCKET_PREFIX}/eval_targets_foldbench_monomers.parquet"
TARGETS_SIZE = 97_519
TARGETS_SHA256 = "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8"
GROUND_TRUTH_URL = f"{BUCKET}/{BUCKET_PREFIX}/gt_universe_scored.jsonl"
GROUND_TRUTH_SIZE = 6_938_887
GROUND_TRUTH_SHA256 = "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5"
SETS_MANIFEST_URL = f"{BUCKET}/{BUCKET_PREFIX}/eval_sets.csv"
SETS_MANIFEST_SIZE = 199_548
SETS_MANIFEST_SHA256 = "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861"

#: The eval sets, and the unit counts a valid run must produce for each. These
#: are the *scored* sizes: eval-test holds 218 natural monomers, one of which
#: (``8uxt_A``) has no representable contacts-v1 document at an 8,192-token
#: context and is excluded from scoring by ``publish_eval_inputs.py``.
EVAL_SETS = ("eval-val", "eval-test", "eval-denovo")
EXPECTED_SET_SIZES = {"eval-val": 97, "eval-test": 217, "eval-denovo": 19}
EXPECTED_UNITS = 333


@dataclass(frozen=True)
class HfFile:
    """One immutable file in a Hugging Face checkpoint export."""

    name: str
    size: int
    digest: str
    digest_kind: str


@dataclass(frozen=True)
class Checkpoint:
    """One selected checkpoint and its pinned model export."""

    label: str
    job_label: str
    run_name: str
    step: int
    hf_repo_id: str | None
    hf_revision: str | None
    checkpoint_files: tuple[HfFile, ...]
    weight_shard_digests: tuple[str, str]
    source_dtype: str
    coreweave_uri: str
    train_loss: float | None = None
    eval_loss: float | None = None
    accepted_unfinished_rollouts: int = 0

    @property
    def hf_subfolder(self) -> str:
        if self.hf_repo_id is None or self.hf_revision is None:
            raise ValueError(f"{self.label} is not sourced from Hugging Face")
        return f"{self.run_name}/hf/step-{self.step}"

    @property
    def files(self) -> tuple[HfFile, ...]:
        return self.checkpoint_files


def exp232_files(weight_etags: tuple[str, str]) -> tuple[HfFile, ...]:
    """Return the six-file exp232 final HF-export manifest."""

    return (
        HfFile("config.json", 1_557, "d8e904f8170ddf00d74c864f31d258a4", "s3-etag"),
        HfFile(
            "model-00001-of-00002.safetensors",
            4_979_485_528,
            weight_etags[0],
            "s3-etag",
        ),
        HfFile(
            "model-00002-of-00002.safetensors", 906_042_048, weight_etags[1], "s3-etag"
        ),
        HfFile(
            "model.safetensors.index.json",
            20_882,
            "bc0a5fd2c9aae096abae4caf9040c79c",
            "s3-etag",
        ),
        HfFile("tokenizer.json", 64_407, "c4b3a16978e30eb150cca4fd8934b6ae", "s3-etag"),
        HfFile(
            "tokenizer_config.json", 290, "336f4e2ca951fa13a20cb1c4b68b2040", "s3-etag"
        ),
    )


M2_P06_CHECKPOINT = Checkpoint(
    label="exp232_decontam_m2_p06_step145199",
    job_label="m2p06d",
    run_name="prot-exp232-cw-cv1-decontam-s02-m2-p06-aug",
    step=145_199,
    hf_repo_id=None,
    hf_revision=None,
    checkpoint_files=exp232_files(
        ("2e38a75033f4df3a73a4be9bc2ceeefe-95", "f444bd62152329ef71c7c46e7ee1c3cd-18")
    ),
    weight_shard_digests=(
        "2e38a75033f4df3a73a4be9bc2ceeefe-95",
        "f444bd62152329ef71c7c46e7ee1c3cd-18",
    ),
    source_dtype="float32",
    coreweave_uri=(
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
        "checkpoints/protein/prot-exp232-cw-cv1-decontam-s02-m2-p06-aug/"
        "2026.08.14.2/hf/step-145199"
    ),
    train_loss=2.853921175003052,
    eval_loss=2.9918437004089355,
)

M1_P02_CHECKPOINT = Checkpoint(
    label="exp232_decontam_m1_p02_step145199",
    job_label="m1p02d",
    run_name="prot-exp232-cw-cv1-decontam-s02-m1-p02-aug",
    step=145_199,
    hf_repo_id=None,
    hf_revision=None,
    checkpoint_files=exp232_files(
        ("ef77fdbd78983ce83059be5c8f200a50-95", "7235c44426f1c3e0489710e4c31ff5a8-18")
    ),
    weight_shard_digests=(
        "ef77fdbd78983ce83059be5c8f200a50-95",
        "7235c44426f1c3e0489710e4c31ff5a8-18",
    ),
    source_dtype="float32",
    coreweave_uri=(
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/"
        "checkpoints/protein/prot-exp232-cw-cv1-decontam-s02-m1-p02-aug/"
        "2026.08.14.2/hf/step-145199"
    ),
    train_loss=2.9559922218322754,
    eval_loss=3.0068159103393555,
    # PR #244 accepted 7 unfinished rollouts for this checkpoint on the 577-unit
    # universe; that allowance was specific to two proteins in that eval set and
    # does not carry over. This run starts at zero and raises if any rollout
    # hits the token cap -- `check_context_budget.py` says none should.
)


#: #199's CoreWeave cooldown -- the current default model, and the contaminated
#: reference here. Identity copied from
#: ``exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/checkpoint_specs.py``.
COOLDOWN_CHECKPOINT = Checkpoint(
    label="exp199_cw_p06_cool_step290400",
    job_label="p06cool",
    run_name="prot-exp199-cw-cv1-p06-cool-s01",
    step=290_400,
    hf_repo_id=None,
    hf_revision=None,
    checkpoint_files=exp232_files(
        ("c4685b3b45694c66418a6f1ff779af91-95", "3788cd21299125acfe3e2d04e91e84e0-18")
    ),
    weight_shard_digests=(
        "c4685b3b45694c66418a6f1ff779af91-95",
        "3788cd21299125acfe3e2d04e91e84e0-18",
    ),
    source_dtype="float32",
    coreweave_uri=(
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp199_continue_contacts_v1_cw/checkpoints/protein/"
        "prot-exp199-cw-cv1-p06-cool-s01/2026.08.14.1/hf/step-290400"
    ),
    train_loss=2.86245059967041,
    eval_loss=2.9396727085113525,
)

CHECKPOINTS = (M2_P06_CHECKPOINT, M1_P02_CHECKPOINT, COOLDOWN_CHECKPOINT)
CHECKPOINT_SUITES = {
    "exp245": CHECKPOINTS,
    # The two decontaminated checkpoints alone, for a rerun that does not need
    # the contaminated reference scored again.
    "decontam": (M2_P06_CHECKPOINT, M1_P02_CHECKPOINT),
}


def run_root(run_id: str) -> str:
    """Return the isolated S3 prefix for one execution attempt."""

    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    ):
        raise ValueError(f"invalid run id: {run_id!r}")
    return f"{S3_ROOT}/{run_id}"


def model_s3_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return an eval-local S3 mirror path for a pinned HF checkpoint.

    Unused by this experiment -- all three checkpoints already live in CoreWeave
    S3 and are read in place -- but ``hf_to_s3.py`` is a verbatim copy of PR
    #244's module and imports it.
    """

    return f"{run_root(run_id)}/models/{checkpoint.label}"


def checkpoint_model_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return the pre-existing CoreWeave checkpoint used by this evaluation."""

    del run_id
    return checkpoint.coreweave_uri
