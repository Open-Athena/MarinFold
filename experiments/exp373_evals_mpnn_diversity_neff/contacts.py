"""Canonical contact sets and diagnostics for the exp373 paired samples."""

import hashlib
import re

STATEMENT = re.compile(r'<(contact|retract)>\s+<p(\d+)>\s+<p(\d+)>')


def sample_seed(stem: str, mode: str, block: int, rollout: int) -> int:
    """Create a sampling stream independent of model, temperature and shard."""
    key = f'exp373:{stem}:{mode}:{block}:{rollout}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63 - 1)


def parse_contacts(text: str, offset: int, length: int) -> dict:
    """Fold edits in order, undo ring offsets, and count every anomaly."""
    text = text.split('<end>', 1)[0]
    live: set[tuple[int, int]] = set()
    counts = dict(statements=0, duplicate_contacts=0, invalid_positions=0,
                  near_diagonal=0, orphan_retractions=0, retractions=0)
    for match in STATEMENT.finditer(text):
        kind, a, b = match.groups()
        counts['statements'] += 1
        a, b = int(a), int(b)
        if a >= 2000 or b >= 2000:
            counts['invalid_positions'] += 1
            continue
        i, j = sorted(((a-offset) % 2000, (b-offset) % 2000))
        if j >= length:
            counts['invalid_positions'] += 1
            continue
        if j-i < 6:
            counts['near_diagonal'] += 1
            continue
        pair = (i, j)
        if kind == 'retract':
            counts['retractions'] += 1
            counts['orphan_retractions'] += int(pair not in live)
            live.discard(pair)
        else:
            counts['duplicate_contacts'] += int(pair in live)
            live.add(pair)
    counts['malformed_statements'] = text.count('<contact>') + text.count('<retract>') - counts['statements']
    counts['unexpected_text'] = bool(STATEMENT.sub('', text).strip())
    return {'contacts': sorted(live), **counts}
