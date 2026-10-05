# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned inputs for the exp277 full-corpus rollout-v2 evaluation."""

from dataclasses import dataclass

MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"
MARIN_PREFIX = "s3://marin-us-east-02a/MarinFold"
S3_ROOT = (
    f"{MARIN_PREFIX}/exp277_models_single_mpnn_pilot/evals/rollout-v2/2026-09-13"
)

# Published evaluation inputs. The driver mirrors these small immutable files
# into the run prefix, verifies their bytes, and builds a 670-unit union without
# deduplicating either source: 554 legacy units plus the 97 eval-val and 19
# eval-denovo FoldBench units specified by the current eval-checkpoint skill.
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

EVAL_SETS = ("eval-val", "eval-denovo")
EXPECTED_SET_SIZES = {"legacy_554": 554, "eval-val": 97, "eval-denovo": 19}
EXPECTED_UNITS = 670
EXPECTED_UNIQUE_STEMS = 556

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

# The final training and HF artifacts were verified after the run completed.
EXP277_CHECKPOINT = Checkpoint(
    label="exp277_full_epoch_m2_p06_step266344",
    job_label="exp277",
    run_name="contacts-v1-exp277-m2-p06-full-epoch-1.5B",
    step=266_344,
    checkpoint_files=(
        HfFile("config.json", 1_726, "09ffdb6012707caf6b8d9c07583ef014", "s3-etag"),
        HfFile(
            "model-00001-of-00002.safetensors",
            4_979_485_528,
            "c54c3d763a013dfb6fe5f6ea10b1142e-95",
            "s3-etag",
        ),
        HfFile(
            "model-00002-of-00002.safetensors",
            906_042_048,
            "cc2e3173c80dbc8d080dbfd6ca7b23df-18",
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
            296,
            "5acd13b50d727187034880bd78bcb928",
            "s3-etag",
        ),
    ),
    weight_shard_digests=(
        "c54c3d763a013dfb6fe5f6ea10b1142e-95",
        "cc2e3173c80dbc8d080dbfd6ca7b23df-18",
    ),
    source_dtype="float32",
    coreweave_uri=(
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
        "runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344"
    ),
    train_loss=3.1852176189422607,
    eval_loss=2.984407424926758,
    eval_loss_step=266_344,
)

# The reshuffled second epoch, continued from the first run's last pre-cooldown
# checkpoint (step-213072) and cooled down over its own final 20%. Its config and
# tokenizer bytes are identical to the first epoch's; only the weight shards
# differ, which is what the pinned ETags below assert.
EXP277_EPOCH2_CHECKPOINT = Checkpoint(
    label="exp277_full_epoch2_from213072_step479417",
    job_label="ep2",
    run_name="contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B",
    step=479_417,
    checkpoint_files=(
        HfFile("config.json", 1_726, "09ffdb6012707caf6b8d9c07583ef014", "s3-etag"),
        HfFile(
            "model-00001-of-00002.safetensors",
            4_979_485_528,
            "73410d564fa17fc29185e8f41eca6e7c-95",
            "s3-etag",
        ),
        HfFile(
            "model-00002-of-00002.safetensors",
            906_042_048,
            "06da6a1d8f28a78ecdf1ef416cdaa032-18",
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
            296,
            "5acd13b50d727187034880bd78bcb928",
            "s3-etag",
        ),
    ),
    weight_shard_digests=(
        "73410d564fa17fc29185e8f41eca6e7c-95",
        "06da6a1d8f28a78ecdf1ef416cdaa032-18",
    ),
    source_dtype="float32",
    coreweave_uri=(
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
        "runs/contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B/hf/step-479417"
    ),
    train_loss=3.183701753616333,
    eval_loss=2.980989456176758,
    eval_loss_step=479_417,
)

CHECKPOINTS = (EXP277_CHECKPOINT,)
# Score both epochs in one job. Rollout scoring is stochastic (#204 spans 0.0023
# across four evaluations of one checkpoint) and the expected effect here is
# small, so the first epoch is re-scored alongside rather than compared against
# its committed v2-01 numbers. That also re-derives the epoch-1 result on this
# cluster, which is a tighter reproducibility check for this specific comparison
# than re-running the E8 gate on an execution path that has not changed.
EPOCH_COMPARISON = (EXP277_CHECKPOINT, EXP277_EPOCH2_CHECKPOINT)
CHECKPOINT_SUITES = {
    "exp277": CHECKPOINTS,
    "exp277-epochs": EPOCH_COMPARISON,
}


def run_root(run_id: str) -> str:
    """Return the isolated S3 prefix for one execution attempt."""

    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    ):
        raise ValueError(f"invalid run id: {run_id!r}")
    return f"{S3_ROOT}/{run_id}"


def checkpoint_model_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return the verified CoreWeave HF directory used by workers."""

    del run_id
    return checkpoint.coreweave_uri
