"""Shared contact-count grid for the exp311 runner and analysis."""


def cuts(length: int) -> list[int]:
    """Include zero, each multiple of ten, and exact L once."""
    return sorted({*range(0, length + 1, 10), length})
