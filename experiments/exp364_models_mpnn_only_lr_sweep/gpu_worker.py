"""Run training as a normal Python entry point, preserving unhandled failures."""

import sys
import traceback

import cloudpickle
import fsspec
from iris.cluster.client.job_info import get_job_info
from marin.training.training import run_levanter_train_lm


def main(config_path: str) -> None:
    """Load the trusted driver config and retain a durable failure traceback."""
    with fsspec.open(config_path, "rb") as stream:
        config = cloudpickle.load(stream)
    try:
        run_levanter_train_lm(config)
    except Exception:
        info = get_job_info()
        if info is None:
            raise
        error_path = (
            f"{config_path}.task-{info.task_index}.attempt-{info.attempt_id}.error.txt"
        )
        with fsspec.open(error_path, "wt") as stream:
            stream.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main(sys.argv[1])
