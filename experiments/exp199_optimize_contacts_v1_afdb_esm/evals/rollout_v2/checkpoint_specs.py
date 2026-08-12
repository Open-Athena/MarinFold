# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable checkpoint and artifact identities for the exp199 rollout-v2 eval."""

from dataclasses import dataclass

EXP199_HF_REPO_ID = "open-athena/marinfold-exp199"
EXP199_HF_REVISION = "ed7103bfd7dac3f75ba759e5ec827da3d75ff0ed"
E8_HF_REPO_ID = "open-athena/marinfold-exp75"
E8_HF_REVISION = "4c9e7779635b585730180823e0ab4b3319b82f67"
MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"
MARIN_PREFIX = "s3://marin-us-east-02a/marin"
S3_ROOT = (
    f"{MARIN_PREFIX}/protein-structure/MarinFold/"
    "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2"
)
TARGETS_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/eval_targets.parquet"
)
TARGETS_SIZE = 43_077
TARGETS_SHA256 = "9de9bc1b99b7e7ab6d2b17a985f9e22bc7decd2b25e1b16be30dea921431c111"
GROUND_TRUTH_URL = (
    "https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
    "data/contacts-v1-model-eval-exp169/gt_universe.jsonl"
)
GROUND_TRUTH_SIZE = 7_956_102
GROUND_TRUTH_SHA256 = "3ff6eb4e383582595ad6f9811c77e2839ebcc0030a050b9c1f15d020163331c9"


@dataclass(frozen=True)
class HfFile:
    """One immutable file in a Hugging Face checkpoint export."""

    name: str
    size: int
    digest: str
    digest_kind: str


@dataclass(frozen=True)
class Checkpoint:
    """One selected checkpoint and its pinned Hugging Face export."""

    label: str
    job_label: str
    run_name: str
    step: int
    hf_repo_id: str
    hf_revision: str
    checkpoint_files: tuple[HfFile, ...]
    weight_shard_digests: tuple[str, str]
    source_dtype: str
    coreweave_uri: str | None = None
    train_loss: float | None = None
    eval_loss: float | None = None

    @property
    def hf_subfolder(self) -> str:
        return f"{self.run_name}/hf/step-{self.step}"

    @property
    def files(self) -> tuple[HfFile, ...]:
        return self.checkpoint_files


def exp199_files(weight_shard_digests: tuple[str, str]) -> tuple[HfFile, ...]:
    """Return the shared exp199 HF export file manifest."""

    return (
        HfFile(
            "config.json",
            1_557,
            "e17a7f2ae8b396a707784940570a4908c359e6fa",
            "git-sha1",
        ),
        HfFile(
            "model-00001-of-00002.safetensors",
            4_979_485_528,
            weight_shard_digests[0],
            "sha256",
        ),
        HfFile(
            "model-00002-of-00002.safetensors",
            906_042_048,
            weight_shard_digests[1],
            "sha256",
        ),
        HfFile(
            "model.safetensors.index.json",
            20_882,
            "9880be895e6d9c514b62ed263640d46f67d01a29",
            "git-sha1",
        ),
        HfFile(
            "tokenizer.json",
            64_407,
            "8b40b35c6dca9a4d0090b975a007599eabf72eff",
            "git-sha1",
        ),
        HfFile(
            "tokenizer_config.json",
            290,
            "e242116d9a12a666749ec722845b6d012250ea94",
            "git-sha1",
        ),
    )


CHECKPOINTS = (
    Checkpoint(
        label="trc_p03_aug_step72599",
        job_label="p03aug",
        run_name="prot-exp199-cv1-s01-m1-p03-aug-us-east1",
        step=72_599,
        hf_repo_id=EXP199_HF_REPO_ID,
        hf_revision=EXP199_HF_REVISION,
        checkpoint_files=exp199_files(
            (
                "52ff4bdf5ac1fb4b212a5e7d95dd48cb1f0abc2ae31882e27f70accd81eef169",
                "2983578865f8d0092f5f2bb6f50fbc9c2bae345e5384140f7602228c3d3c9c8e",
            )
        ),
        weight_shard_digests=(
            "52ff4bdf5ac1fb4b212a5e7d95dd48cb1f0abc2ae31882e27f70accd81eef169",
            "2983578865f8d0092f5f2bb6f50fbc9c2bae345e5384140f7602228c3d3c9c8e",
        ),
        source_dtype="float32",
        train_loss=2.9426701068878174,
        eval_loss=3.011530637741089,
    ),
    Checkpoint(
        label="trc_p03_base_step72599",
        job_label="p03base",
        run_name="prot-exp199-cv1-s01-m1-p03-base-us-east5",
        step=72_599,
        hf_repo_id=EXP199_HF_REPO_ID,
        hf_revision=EXP199_HF_REVISION,
        checkpoint_files=exp199_files(
            (
                "4bb5da27c9732f62b06113d786f037b12ce1bd5c1284fb8c51a8bf6d6ede9c3c",
                "45d4fcfd790f18913677d7d77d27e8508a2f9d59dff08fc82a728f010d6cd045",
            )
        ),
        weight_shard_digests=(
            "4bb5da27c9732f62b06113d786f037b12ce1bd5c1284fb8c51a8bf6d6ede9c3c",
            "45d4fcfd790f18913677d7d77d27e8508a2f9d59dff08fc82a728f010d6cd045",
        ),
        source_dtype="float32",
        train_loss=2.936706781387329,
        eval_loss=3.00742244720459,
    ),
    Checkpoint(
        label="cw_p06_aug_step145199",
        job_label="p06aug",
        run_name="prot-exp199-cw-cv1-s02-m1-p06-aug",
        step=145_199,
        hf_repo_id=EXP199_HF_REPO_ID,
        hf_revision=EXP199_HF_REVISION,
        checkpoint_files=exp199_files(
            (
                "e8db3b664752ab2fb50c72b98c90b63a9c8414f23c26e3e348371187b686ea99",
                "a7a38503f20053d5fa42cca9c86a137eb768ec8f16f2d38074d01672b49cf1a6",
            )
        ),
        weight_shard_digests=(
            "e8db3b664752ab2fb50c72b98c90b63a9c8414f23c26e3e348371187b686ea99",
            "a7a38503f20053d5fa42cca9c86a137eb768ec8f16f2d38074d01672b49cf1a6",
        ),
        source_dtype="float32",
        train_loss=2.8773207664489746,
        eval_loss=2.971200942993164,
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

CHECKPOINT_SUITES = {
    "exp199": CHECKPOINTS,
    "e8-reference": (E8_REFERENCE_CHECKPOINT,),
}

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
    """Return the eval-local S3 mirror for one pinned HF checkpoint."""

    return f"{run_root(run_id)}/models/{checkpoint.hf_subfolder}"


def checkpoint_model_uri(run_id: str, checkpoint: Checkpoint) -> str:
    """Return an existing CoreWeave artifact or this eval's verified HF mirror."""

    return checkpoint.coreweave_uri or model_s3_uri(run_id, checkpoint)
