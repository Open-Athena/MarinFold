# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from materialize_removal_manifest import row_fields


def test_row_fields_preserves_entry_id_suffix() -> None:
    assert row_fields("afdb|02035_1677_AF-A0A3R7JMW2-F1") == (
        "afdb",
        2035,
        1677,
        "AF-A0A3R7JMW2-F1",
    )
    assert row_fields("esm_atlas|02008_3262_9a06_e8ce") == (
        "esm_atlas",
        2008,
        3262,
        "9a06_e8ce",
    )


def test_row_fields_rejects_malformed_id() -> None:
    with pytest.raises(ValueError, match="malformed"):
        row_fields("not-a-row-id")
