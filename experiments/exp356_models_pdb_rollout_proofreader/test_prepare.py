"""Protect the coordinate-completeness contract used for negative labels."""

from prepare import canonical_span


def test_terminal_trimming_does_not_accept_internal_missing_residues():
    canonical = 'ACDEFGHIKLMNPQRSTVWY'
    assert canonical_span(canonical,canonical,2,.8) == (0,20)
    assert canonical_span(canonical[1:-1],canonical,2,.8) == (1,19)
    assert canonical_span(canonical[:8]+canonical[9:],canonical,2,.8) is None
    assert canonical_span(canonical[3:],canonical,2,.8) is None
    assert canonical_span('ACDEFGHIK','ACDEFGHIKACDEFGHIK',20,.4) is None
