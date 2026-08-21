# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the issue #211 Phase 0 result through the real library code.

Phase 0 was run in a throwaway numpy script before the experiment existed. This
re-runs it against ``consistency.py`` + ``arms.py`` on the GPU-batched path, so
the numbers quoted in the issue are backed by the code that will actually produce
the results. Run::

    uv run python validate_phase0.py

Expected (1QYS, L=92, 76 contacts at sep>=6, degree>=0.001):

* the ground-truth set embeds to ~0 contact excess;
* separation-matched random does not;
* the triangle tier reports 0 violations for **both** — the documented null.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms import separation_matched_random  # noqa: E402
from consistency import (  # noqa: E402
    MIN_SEP,
    Bounds,
    contact_matrix,
    embed_residual,
    packing_score,
    triangle_violations,
)

# The one structure vendored in the repo; enough to validate the machinery.
# calibrate_bounds.py does this properly across all 554 eval proteins.
CIF = Path(__file__).resolve().parents[2] / "marinfold/tests/data/1QYS.cif"
MIN_DEGREE = 0.001


def load_1qys():
    from marinfold.document_structures.contacts_v1.parse import analyze_structure

    an = analyze_structure(str(CIF))
    length = len(an.residues)
    pairs = [
        (c.seq_i, c.seq_j)
        for c in an.contacts
        if c.degree >= MIN_DEGREE and (c.seq_j - c.seq_i) >= MIN_SEP
    ]
    return an.entry_id, length, pairs


def main() -> int:
    entry, length, gt = load_1qys()
    bounds = Bounds()
    rng = np.random.default_rng(0)

    sets = {"GT": gt}
    for k in range(3):
        sets[f"random-{k}"] = separation_matched_random(
            gt, length, np.random.default_rng(100 + k)
        )

    names = list(sets)
    masks = np.stack([contact_matrix(sets[n], length) for n in names])

    print(f"{entry}: L={length}, {len(gt)} GT contacts (sep>={MIN_SEP}, deg>={MIN_DEGREE})")
    print(f"bounds: bond={bounds.bond} U_contact={bounds.u_contact} "
          f"L_noncontact={bounds.l_noncontact} d_min={bounds.d_min}\n")

    tri = [triangle_violations(m, bounds) for m in masks]
    pack = [packing_score(m) for m in masks]
    emb = embed_residual(masks, bounds, n_restarts=4, seed=0)

    print(f"{'arm':<12} {'n_c':>4} {'maxdeg':>7} {'tri_viol':>9} "
          f"{'excess':>8} {'per_c':>7} {'unsat':>7} {'nc_viol':>8} {'Rg':>6}")
    for n, p, t, e in zip(names, pack, tri, emb):
        print(f"{n:<12} {p['n_contacts']:>4.0f} {p['max_degree']:>7.0f} "
              f"{t['n_triangle_violations']:>9.0f} {e['contact_excess']:>8.2f} "
              f"{e['contact_excess_per_contact']:>7.3f} {e['unsat_frac']:>7.3f} "
              f"{e['noncontact_violation']:>8.1f} {e['rg']:>6.1f}")

    gt_excess = emb[0]["contact_excess"]
    rnd_excess = [e["contact_excess"] for e in emb[1:]]
    ok = gt_excess < min(rnd_excess)
    print(f"\nGT excess {gt_excess:.2f} vs random min {min(rnd_excess):.2f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"triangle tier fired on {sum(t['n_triangle_violations'] > 0 for t in tri)}/"
          f"{len(tri)} arms (Phase 0 expects 0 — the documented null)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
