# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Driver for exp78: ESMFold / ESMFold2 contact-prediction eval vs pyconfind GT.

Subcommands:
  - ``contact-eval``  Run pyconfind on the predicted ESM structures and score
                      contacts (structure predictor) against pyconfind ground
                      truth, identically to exp74's Protenix structure config.

The structure predictions themselves come from the Modal apps, invoked
directly::

    modal run esmfold_app.py::setup_weights
    python esmfold_app.py  --manifest data/eval_manifest_foldbench.csv data/eval_manifest_exp65.csv
    modal run esmfold2_app.py::setup_weights
    python esmfold2_app.py --manifest data/eval_manifest_foldbench.csv data/eval_manifest_exp65.csv

Combine + plot live in ``combine_scores.py`` / ``plot.py`` (run directly).
"""

import argparse

import contact_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    contact_eval.add_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
