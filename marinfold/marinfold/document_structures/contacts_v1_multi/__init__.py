"""Multi-hypothesis contacts with an explicit final synthesis boundary."""

from .format import (
    BEGIN,
    CONTACT,
    END,
    FINAL,
    MULTI,
    ParsedHistory,
    parse_history,
    truncate_history,
)

__all__ = ["BEGIN", "CONTACT", "END", "FINAL", "MULTI", "ParsedHistory", "parse_history", "truncate_history"]
