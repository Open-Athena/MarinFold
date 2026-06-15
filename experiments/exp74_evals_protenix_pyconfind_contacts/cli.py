# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Driver for exp74: Protenix contact-prediction eval vs pyconfind ground truth.

Subcommands:
  - ``run``           Fan out Protenix across (protein x {single_seq, msa}) on
                      Modal (delegates to ``modal_app.py``). Inputs come from
                      ``prepare_exp65.py`` (exp65) or the exp12 HF bucket
                      (FoldBench-100).
  - ``select-best``   Pick the top-1 sample per (protein, mode) by Protenix's
                      ranking_score into a clean ``best/`` tree (reused verbatim
                      from exp12).
  - ``contact-eval``  Score the four configs vs pyconfind ground truth.

Input prep and eval-manifest building live in ``prepare_exp65.py`` /
``eval_manifest.py`` (run those directly).
"""

import argparse

import contact_eval
import select_best


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Run Protenix on Modal across proteins x modes.")
    p.add_argument("--inputs", required=True, help="Dir with manifest.csv + jobs/ (from prepare_exp65.py).")
    p.add_argument("--modes", default="single_seq,msa", help="Comma-separated mode list.")
    p.add_argument("--output-volume", default="exp74-protenix-runs", help="Modal output Volume name.")
    p.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds.")
    p.add_argument("--n-sample", type=int, default=8, help="Diffusion samples per seed.")
    p.add_argument("--n-cycle", type=int, default=10, help="Trunk cycles (recycles).")
    p.add_argument("--gpu", default="H100", help="Modal GPU class.")
    p.add_argument("--stems-file", default=None, help="Optional one-stem-per-line file restricting the run.")

    def _dispatch(args: argparse.Namespace) -> None:
        import modal_app
        modal_app.run_cli(args)

    p.set_defaults(func=_dispatch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _add_run_subparser(subparsers)
    select_best.add_subparser(subparsers)
    contact_eval.add_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
