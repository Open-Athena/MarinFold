"""Evaluation locality follows the selected region, including reference models."""

import pytest

from common import validate_eval_locality


@pytest.mark.parametrize(
    ("cluster", "bucket"),
    [("cw-rno2a", "marin-rno2a"), ("cw-us-east-02a", "marin-us-east-02a")],
)
def test_local_evaluation_accepts_production_and_reference_exports(
    cluster: str, bucket: str
) -> None:
    root = f"s3://{bucket}/MarinFold/exp347_qwen_base_contacts"
    for checkpoint in [
        root + "/checkpoints/run/hf/step-100",
        f"s3://{bucket}/MarinFold/exp163/model/step-35679",
    ]:
        validate_eval_locality(checkpoint, root + "/eval-val/run/step-100", cluster)


@pytest.mark.parametrize(
    ("checkpoint", "output"),
    [
        (
            "s3://marin-us-east-02a/MarinFold/model",
            "s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts/eval-val/run",
        ),
        (
            "s3://marin-rno2a/MarinFold/model",
            "s3://marin-us-east-02a/MarinFold/exp347_qwen_base_contacts/eval-val/run",
        ),
        (
            "s3://marin-rno2a-other/MarinFold/model",
            "s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts/eval-val/run",
        ),
        (
            "s3://marin-rno2a/MarinFold/model",
            "s3://marin-rno2a/MarinFold/exp347_qwen_base_contacts-other/eval-val/run",
        ),
    ],
)
def test_local_evaluation_rejects_remote_or_unrelated_paths(
    checkpoint: str, output: str
) -> None:
    with pytest.raises(ValueError, match="local to its compute cluster"):
        validate_eval_locality(checkpoint, output, "cw-rno2a")
