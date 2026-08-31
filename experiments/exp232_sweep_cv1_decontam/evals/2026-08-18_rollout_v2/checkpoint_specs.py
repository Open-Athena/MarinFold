# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable identities for the exp232 decontaminated-checkpoint evaluation."""

from dataclasses import dataclass

E8_HF_REPO_ID = "open-athena/marinfold-exp75"
E8_HF_REVISION = "4c9e7779635b585730180823e0ab4b3319b82f67"
MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"
MARIN_PREFIX = "s3://marin-us-east-02a/marin"
S3_ROOT = (
    f"{MARIN_PREFIX}/protein-structure/MarinFold/"
    "exp232_sweep_cv1_decontam/evals/rollout_v2"
)
LEGACY_TARGETS_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/eval_targets.parquet"
)
LEGACY_TARGETS_SIZE = 43_077
LEGACY_TARGETS_SHA256 = (
    "9de9bc1b99b7e7ab6d2b17a985f9e22bc7decd2b25e1b16be30dea921431c111"
)
GROUND_TRUTH_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-eval2-exp226/gt_universe_eval2.jsonl"
)
GROUND_TRUTH_SIZE = 8_362_085
GROUND_TRUTH_SHA256 = "86116d7961e77d2948bc17f938c076a264992a7bbae8c173989e64b5d03cd1fc"
EVAL2_MANIFEST_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-eval2-exp226/eval2_manifest.csv"
)
EVAL2_MANIFEST_SIZE = 81_591
EVAL2_MANIFEST_SHA256 = (
    "7c8b144d79153d87f10ad095d879003b4d8972d61f315b746e53889658bcdd6d"
)


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


E8_REFERENCE_CHECKPOINT = Checkpoint(
    label="e8_reference_step35679",
    job_label="e8ref",
    run_name="prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084",
    step=35_679,
    hf_repo_id=E8_HF_REPO_ID,
    hf_revision=E8_HF_REVISION,
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
    accepted_unfinished_rollouts=7,
)

CHECKPOINTS = (E8_REFERENCE_CHECKPOINT, M2_P06_CHECKPOINT, M1_P02_CHECKPOINT)
CHECKPOINT_SUITES = {"exp232": CHECKPOINTS}

E8_REFERENCE_METRICS = {
    ("all", "R"): 0.4245291213628376,
    ("long", "R"): 0.3656151868856005,
    ("all", "AUC"): 0.9009633507400637,
    ("long", "AUC"): 0.8737803746877988,
}
E8_REFERENCE_TOLERANCE = 0.005


def run_root(run_id: str) -> str:
    """Return the isolated S3 prefix for one execution attempt."""

    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    ):
        raise ValueError(f"invalid run id: {run_id!r}")
    return f"{S3_ROOT}/{run_id}"


def model_s3_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return an eval-local S3 mirror for a pinned HF checkpoint."""

    return f"{run_root(run_id)}/models/{checkpoint.hf_subfolder}"


def checkpoint_model_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return the pre-existing CoreWeave checkpoint used by this evaluation."""

    del run_id
    return checkpoint.coreweave_uri
