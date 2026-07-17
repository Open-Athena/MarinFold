# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored pure-Python subset of bio2token (flagshippioneering/bio2token).

Source: https://github.com/flagshippioneering/bio2token (main, 2026-07-07).
Only the CUDA-free pieces are vendored here so this experiment can run the
bio2token tokenizer on CPU/MPS/XLA *without* the upstream package's
``mamba-ssm`` + ``torch==2.4.1+cu121`` pins (see the experiment README):

- ``layers/fsq.py``            — Finite Scalar Quantizer (parameter-free codebook).
- ``data/utils/utils.py``      — pdb_2_dict / uniform_dataframe / compute_masks
                                 (the reference input pipeline; used as the
                                 correctness oracle for our own adapter).
- ``data/utils/tokens.py``,
  ``data/utils/molecule_conventions.py`` — atom/residue conventions.

The Mamba encoder itself is NOT vendored; it is reimplemented in pure
PyTorch in ``../mamba.py`` (the upstream encoder needs CUDA kernels).

NOTE (license): the upstream repo did not ship a top-level LICENSE at the
vendored commit. Confirm licensing before publishing/redistributing this
vendored code. See the experiment README "Open questions".
"""
