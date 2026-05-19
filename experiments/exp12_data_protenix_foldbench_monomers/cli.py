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
  (``Σ p_bin · midpoint``) against the GT CB-CB (CA-for-GLY) distance,
  and dRMSD of the predicted CA-CA pairwise distance matrix vs GT
  CA-CA. Emit ``data/scores.csv``.

- ``plot``  Render comparison PNGs to ``plots/`` from
  ``data/scores.csv``.

See ``README.md`` in this directory for the full plan.
"""

import argparse

import plot as plot_mod
import prepare_inputs
import score as score_mod
import select_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    prepare_inputs.add_subparser(subparsers)
    _add_run_subparser(subparsers)
    select_best.add_subparser(subparsers)
    score_mod.add_subparser(subparsers)
    plot_mod.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Wire the ``run`` subcommand. Delegates to ``modal_app.py`` lazily.

    Modal client import is deferred so ``prepare-inputs`` / ``score`` /
    ``plot`` work without needing the modal SDK installed.
    """
    p = subparsers.add_parser("run", help="Run Protenix on Modal across proteins × modes.")
    p.add_argument("--inputs", required=True, help="Dir produced by prepare-inputs.")
    p.add_argument("--modes", default="single_seq,msa", help="Comma-separated mode list.")
    p.add_argument("--output-volume", default="foldbench-protenix-runs",
                   help="Modal Volume name for outputs (created if missing).")
    p.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds.")
    p.add_argument("--n-sample", type=int, default=8, help="Diffusion samples per seed.")
    p.add_argument("--n-cycle", type=int, default=10, help="Trunk cycles (recycles).")
    p.add_argument("--gpu", default="H100", help="Modal GPU class.")

    def _dispatch(args: argparse.Namespace) -> None:
        # Lazy import so the CLI loads without the modal SDK.
        import modal_app
        modal_app.run_cli(args)
    p.set_defaults(func=_dispatch)


if __name__ == "__main__":
    main()
