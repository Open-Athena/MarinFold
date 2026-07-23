# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""HF export of an exp108 sweep run — WIP against modern marin (issue #108).

NOT YET PORTED. marin 0.2.38 refactored its execution framework: the old export
path this used (``marin.execution.executor_main`` +
``marin.export.convert_checkpoint_to_hf_step``) is gone/renamed. HF export is
only needed AFTER a run produces a checkpoint, so it's deferred rather than
block the training scaffold.

TODO to port:
  * find the modern marin HF-export API (levanter's checkpoint→HF converter, or
    marin's replacement for ``convert_checkpoint_to_hf_step``);
  * run it at iris **batch** priority (like training — via a direct
    ``fray.JobRequest(priority=3)``, not the executor);
  * co-locate the contacts-v1 tokenizer with the exported weights (hard rule);
  * point it at the run's checkpoint under
    ``s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/checkpoints/<run_name>/``
    (confirm the exact ``step-{N}`` / run-id subdir from the S3 listing).

Choose the run to export with ``EXP108_EXPORT_LR`` (default 1e-3, #75's winner).
"""

import os

from train_qwen_3b_contacts_v1_sweep import run_name_for


def main() -> None:
    lr = float(os.environ.get("EXP108_EXPORT_LR", "1e-3"))
    run_name = run_name_for(lr)
    raise NotImplementedError(
        "exp108 HF export is not yet ported to modern marin (0.2.38). "
        f"Target run: {run_name}. See this module's docstring for the TODO. "
        "Checkpoints are at "
        f"s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/checkpoints/{run_name}/."
    )


if __name__ == "__main__":
    main()
