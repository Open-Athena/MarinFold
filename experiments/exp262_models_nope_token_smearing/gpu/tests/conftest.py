# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Put the repository root and this project on the path.

``exp262_train_cw`` imports the exp232 contract as
``experiments.exp232_sweep_cv1_decontam.training_contract``, which resolves only
with the repository root importable — the same way exp232's own scripts run.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[4]

for path in (str(PROJECT), str(REPOSITORY)):
    if path not in sys.path:
        sys.path.insert(0, path)
