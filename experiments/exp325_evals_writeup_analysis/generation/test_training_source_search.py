"""Check that local domain matches cannot inflate whole-query coverage."""

from search_training_sources import domain_alignment


def test_gaps_insertions_overlaps_and_weak_domains() -> None:
    domains = [
        {"bitscore": 20, "ievalue": 1e-8, "is_reported": True,
         "alignment_display": {"hmmfrom": 2, "hmmto": 5, "m": 10,
                               "model": "AB.CD", "aseq": "K-lMN"}},
        {"bitscore": 10, "ievalue": 1e-6, "is_reported": True,
         "alignment_display": {"hmmfrom": 5, "hmmto": 7, "m": 10,
                               "model": "EFG", "aseq": "PQR"}},
        {"bitscore": 5, "ievalue": 0.1, "is_reported": True,
         "alignment_display": {"hmmfrom": 1, "hmmto": 10, "m": 10,
                               "model": "ABCDEFGHIJ", "aseq": "ABCDEFGHIJ"}},
    ]
    # The insertion 'l' contributes no query position, deletion 3 stays empty,
    # overlap 5 is counted once, and the weak full-length domain adds nothing.
    assert domain_alignment({"domains": domains}, 10) == "-K-MNQR---"
