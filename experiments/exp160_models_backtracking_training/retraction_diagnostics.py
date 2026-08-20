# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does the model's retraction actually track wrongness? (issue #160)

Aggregate accuracy metrics cannot say *why* backtracking helped or didn't.
These diagnostics answer that, and one of them is the experiment's real
pass/fail:

**Is retraction discriminative?** Among the contacts a model emits, are the
ones it later retracts enriched for **false positives** relative to the ones it
keeps? If retracted contacts are no likelier to be wrong than kept ones, then
retraction is noise and the mechanism has failed — regardless of what the
headline R-precision does. We report it as a 2x2 contingency plus:

    precision  = P(false positive | retracted)      # of what it took back, how much was wrong
    recall     = P(retracted | false positive)      # of its mistakes, how many it caught
    enrichment = precision / P(false positive)      # 1.0 = no signal; higher = discriminative

Enrichment is the headline: it is scale-free, so it is comparable across
proteins with wildly different false-positive base rates.

Also reported:

- **retract rate** — a model that never emits ``<retract>`` has simply ignored
  the mechanism (the trivial failure mode).
- **retraction distance** — statements between a contact and its retraction. If
  the model only ever retracts the statement it just emitted, the long-delay
  training signal did not transfer (corpus mean was ~6.4).
- **recovery** — after retracting a pair, does the model go on to emit a *true*
  contact involving either of those residues? Retracting is only half the job.

Everything is computed from a document's edit list + the ground-truth pair set,
so the same code serves corpus QA and trained-model rollouts. Inputs are in
whatever pair space the caller uses (position or sequence index) as long as
both arguments agree.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

Pair = tuple[int, int]


def _canon(a: int, b: int) -> Pair:
    return (a, b) if a <= b else (b, a)


@dataclass
class DocDiagnostics:
    """Per-document retraction diagnostics."""

    n_statements: int = 0
    n_contact_statements: int = 0
    n_retract_statements: int = 0
    # Unique pairs, by whether they were ever retracted and whether they are GT.
    retracted_fp: int = 0
    retracted_tp: int = 0
    kept_fp: int = 0
    kept_tp: int = 0
    # Statements between each contact emission and its retraction.
    distances: list[int] = field(default_factory=list)
    # Retractions followed by a true contact on one of the freed residues.
    n_recovered: int = 0
    n_retracted_pairs: int = 0
    # Edit-list anomalies (should be 0 in authored corpora).
    n_retract_absent: int = 0

    @property
    def n_emitted(self) -> int:
        return self.retracted_fp + self.retracted_tp + self.kept_fp + self.kept_tp

    @property
    def n_fp(self) -> int:
        return self.retracted_fp + self.kept_fp


def diagnose_document(
    statements: Iterable[tuple[str, int, int]],
    gt: Iterable[Pair],
) -> DocDiagnostics:
    """Diagnose one document's edit list against its ground-truth pairs.

    ``statements`` is the ordered ``(kind, a, b)`` stream (e.g. from
    ``read.iter_structure_statements``); ``gt`` the true contact pairs.
    """
    gt_set = {_canon(*p) for p in gt}
    diag = DocDiagnostics()

    emitted_at: dict[Pair, int] = {}      # last emission index per live pair
    ever_retracted: set[Pair] = set()
    ever_emitted: set[Pair] = set()
    live: set[Pair] = set()
    # Residues freed by a retraction, awaiting a later true contact.
    pending_recovery: list[tuple[Pair, bool]] = []

    for index, (kind, a, b) in enumerate(statements):
        pair = _canon(a, b)
        diag.n_statements += 1
        if kind == "contact":
            diag.n_contact_statements += 1
            ever_emitted.add(pair)
            emitted_at[pair] = index
            live.add(pair)
            # Recovery: a TRUE contact touching a residue freed by an earlier
            # retraction counts as recovering from that mistake (once).
            if pair in gt_set:
                for slot, (retracted_pair, done) in enumerate(pending_recovery):
                    if done:
                        continue
                    if pair[0] in retracted_pair or pair[1] in retracted_pair:
                        pending_recovery[slot] = (retracted_pair, True)
                        diag.n_recovered += 1
                        break
        else:  # retract
            diag.n_retract_statements += 1
            if pair not in live:
                diag.n_retract_absent += 1
                continue
            live.discard(pair)
            ever_retracted.add(pair)
            diag.n_retracted_pairs += 1
            if pair in emitted_at:
                diag.distances.append(index - emitted_at[pair])
            pending_recovery.append((pair, False))

    for pair in ever_emitted:
        is_gt = pair in gt_set
        if pair in ever_retracted:
            if is_gt:
                diag.retracted_tp += 1
            else:
                diag.retracted_fp += 1
        else:
            if is_gt:
                diag.kept_tp += 1
            else:
                diag.kept_fp += 1
    return diag


def aggregate(diags: Sequence[DocDiagnostics]) -> dict[str, float]:
    """Corpus-level summary. ``enrichment`` is the headline discrimination number."""
    if not diags:
        return {}
    retracted_fp = sum(d.retracted_fp for d in diags)
    retracted_tp = sum(d.retracted_tp for d in diags)
    kept_fp = sum(d.kept_fp for d in diags)
    kept_tp = sum(d.kept_tp for d in diags)
    n_emitted = retracted_fp + retracted_tp + kept_fp + kept_tp
    n_retracted = retracted_fp + retracted_tp
    n_fp = retracted_fp + kept_fp
    distances = [d for diag in diags for d in diag.distances]

    base_rate = n_fp / n_emitted if n_emitted else float("nan")
    precision = retracted_fp / n_retracted if n_retracted else float("nan")
    recall = retracted_fp / n_fp if n_fp else float("nan")
    enrichment = (precision / base_rate) if (n_retracted and base_rate) else float("nan")

    n_docs = len(diags)
    n_with_retract = sum(1 for d in diags if d.n_retract_statements > 0)
    total_retracted_pairs = sum(d.n_retracted_pairs for d in diags)
    return {
        "n_documents": n_docs,
        "docs_with_any_retraction": n_with_retract,
        "frac_docs_with_retraction": n_with_retract / n_docs,
        "mean_retracts_per_doc": sum(d.n_retract_statements for d in diags) / n_docs,
        "mean_contacts_per_doc": sum(d.n_contact_statements for d in diags) / n_docs,
        "n_emitted_pairs": n_emitted,
        "fp_base_rate": base_rate,
        "retract_precision_fp": precision,
        "retract_recall_fp": recall,
        "retract_enrichment": enrichment,
        "retracted_tp_count": retracted_tp,
        "mean_retract_distance": (sum(distances) / len(distances)) if distances else float("nan"),
        "median_retract_distance": (sorted(distances)[len(distances) // 2] if distances else float("nan")),
        "frac_immediate_retractions": (
            sum(1 for d in distances if d <= 1) / len(distances) if distances else float("nan")
        ),
        "recovery_rate": (
            sum(d.n_recovered for d in diags) / total_retracted_pairs
            if total_retracted_pairs else float("nan")
        ),
        "n_retract_absent": sum(d.n_retract_absent for d in diags),
    }


def format_report(summary: dict[str, float]) -> str:
    """Human-readable summary, with the pass/fail line called out."""
    if not summary:
        return "(no documents)"
    lines = [
        f"documents:            {summary['n_documents']:.0f} "
        f"({summary['frac_docs_with_retraction']:.1%} contain a retraction)",
        f"mean contacts/doc:    {summary['mean_contacts_per_doc']:.1f}",
        f"mean retracts/doc:    {summary['mean_retracts_per_doc']:.1f}",
        "",
        f"FP base rate:         {summary['fp_base_rate']:.3f}  "
        f"(of {summary['n_emitted_pairs']:.0f} emitted pairs)",
        f"retract precision:    {summary['retract_precision_fp']:.3f}  "
        "P(false positive | retracted)",
        f"retract recall:       {summary['retract_recall_fp']:.3f}  "
        "P(retracted | false positive)",
        f"ENRICHMENT:           {summary['retract_enrichment']:.2f}x  "
        "<- 1.0 = no signal (the pass/fail)",
        f"true contacts retracted: {summary['retracted_tp_count']:.0f}",
        "",
        f"retract distance:     mean {summary['mean_retract_distance']:.1f} / "
        f"median {summary['median_retract_distance']:.0f} statements "
        f"({summary['frac_immediate_retractions']:.1%} immediate)",
        f"recovery rate:        {summary['recovery_rate']:.3f}  "
        "(true contact on a freed residue after retracting)",
    ]
    if summary.get("n_retract_absent"):
        lines.append(f"malformed retracts:   {summary['n_retract_absent']:.0f}")
    return "\n".join(lines)
