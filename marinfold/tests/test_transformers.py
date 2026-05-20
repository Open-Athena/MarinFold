# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from marinfold.inference._transformers import _resolve_dtype  # noqa: E402


def test_bfloat16_resolves_to_torch_bfloat16() -> None:
    assert _resolve_dtype("bfloat16") is torch.bfloat16


def test_shared_backend_default_dtype_stays_safe_for_mps_models() -> None:
    from marinfold.inference._transformers import TransformersBackend

    assert TransformersBackend.__init__.__kwdefaults__ == {
        "dtype": "bfloat16",
        "device": None,
        "tail_batch_size": 64,
    }
