"""Checks for fold-mode contact screening."""

from analyze_foldswitch import fold_scores

TARGET = {
    "common_positions": list(range(40)),
    "contacts_fold1": [[5, 20], [6, 21], [7, 22], [8, 23]],
    "contacts_fold2": [[5, 30], [6, 31], [7, 32], [8, 33]],
    "fs_lo": 5,
    "fs_hi": 10,
}


def test_fold_screen_distinguishes_opposing_contact_modes() -> None:
    fold1 = fold_scores([[5, 20], [6, 21]], TARGET)
    fold2 = fold_scores([[5, 30], [6, 31]], TARGET)
    assert fold1["fold1_hit"] and not fold1["fold2_hit"]
    assert fold2["fold2_hit"] and not fold2["fold1_hit"]
    assert fold1["enrichment"] > 0
    assert fold2["enrichment"] < 0
