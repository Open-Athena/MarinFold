# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Driver for the exp12 Protenix-on-FoldBench experiment.

Subcommands:

- ``prepare-inputs``  Build per-protein Protenix JSON jobs + cache the GT
  mmCIFs from RCSB. Reads FoldBench's ``targets/monomer_protein.csv`` and
  emits one job JSON + one GT CIF per protein. Local-only, no GPU.

- ``run``  Fan out across (protein × {single_seq, msa}) on Modal. Calls
  into ``modal_app.py`` which loads Protenix once per container, attaches
  the distogram-capture forward hook, and persists outputs to a Modal
  Volume.

- ``select-best``  Per (protein, mode), pick the top-1 sample by
  Protenix's ``ranking_score`` and link the matching structure +
  confidence + distogram into a clean ``best/`` tree.

- ``score``  Per (protein, mode), compute MAE of the expected distance
  (``Σ p_bin · midpoint``) against the GT CA-CA distance, and dRMSD
  of the predicted CA-CA pairwise distance matrix vs GT. Emit
  ``data/scores.csv``.

- ``plot``  Render comparison PNGs to ``plots/`` from
  ``data/scores.csv``.

See ``README.md`` in this directory for the full plan.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Each subcommand binding lands here as we implement it. Stubbed
    # for now so the CLI surface is locked in and we can iterate
    # phase-by-phase without churning users.
    subparsers.add_parser("prepare-inputs", help="FoldBench CSV → Protenix JSON + GT CIF cache")
    subparsers.add_parser("run", help="Run Protenix on Modal across proteins × modes")
    subparsers.add_parser("select-best", help="Pick top-1 sample per (protein, mode) by ranking_score")
    subparsers.add_parser("score", help="MAE on expected distances + dRMSD on CA-CA")
    subparsers.add_parser("plot", help="Render comparison PNGs from data/scores.csv")

    args = parser.parse_args()
    raise NotImplementedError(
        f"Subcommand {args.cmd!r} not yet implemented — see README.md for the phased plan."
    )


if __name__ == "__main__":
    main()
