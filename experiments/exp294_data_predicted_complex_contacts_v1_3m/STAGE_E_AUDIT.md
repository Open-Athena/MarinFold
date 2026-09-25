# Stage E: how many additional heterodimers can external sources supply?

AFCDB's heterodimer arm is exhausted at **183,809** against the issue's 500,000
target (see README). This audits the four sources the issue names. Measured
2026-09-25.

## Answer

**Yes, 500k is reachable — but the binding constraint is our deduplication
policy, not the sources.**

| | heterodimers |
| --- | --- |
| AFCDB (extracted, unique sequence pairs) | 183,809 |
| \+ PINDER, deduplicated to unique pairs | **218,499** |
| \+ PINDER, redundancy preserved | **590,304** |

PINDER alone clears 500k *if documents are kept per structure rather than per
sequence pair*. AFCDB forced the dedup question to be invisible — it stores
essentially one model per accession pair (mean 1.000) — so collapsing to unique
pairs cost nothing there. PINDER is the opposite: 461,710 heterodimer systems
over 38,428 unique pairs, ~12 structures per pair, because the PDB holds many
crystal forms of the same complex. Those are genuinely different structures,
and the issue explicitly asks to "preserve redundancy in the stored corpus but
control it in sampling", which argues for keeping them.

## PINDER — the clear winner

Apache-2.0, PDB-derived (experimental), index and metadata are two public
parquet files totalling 228 MB, no structures needed to audit.

| | |
| --- | --- |
| heterodimer systems (all splits) | 760,177 over 34,609 clusters, 41,248 PDB entries |
| heterodimers excluding `invalid` split | 461,710 |
| fit the contacts-v1 1,998-residue ring | 455,634 |
| eligible unique pairs (ring + >=1 interface contact) | 37,644 |
| **overlapping with our AFCDB heterodimers** | **2,954 (7.8%)** |
| **additional unique pairs** | **34,690** |
| **additional systems, redundancy preserved** | **406,495** |
| **new interface clusters** | **23,639** |

The 7.8% overlap is the important number: PINDER is experimental and AFCDB is
predicted, so they are nearly disjoint at the sequence-pair level. This is
additive data, not a re-run of what we have.

## Predictomes — additive, and cheaply licensed

CC BY 4.0, ~1.6M predicted human protein pairs (AlphaFold), plus ~180k pairs of
associated screens.

Our AFCDB heterodimers are only **7.4% human** (13,644 of 183,809); the largest
taxon is *Glycine max* at 20%, with human fourth at 7.2%. So a human-only
source is close to orthogonal to what we hold, rather than redundant with it.
Needs a confidence gate comparable to our ipSAE/pDockQ2 floor before any count
is trustworthy, and the top-16k subset alone is 53 GB, so this is a real
project rather than an index read.

## Protein Complex Atlas — blocked on licence

1.1M predicted complexes, 181,671 high-confidence, ColabFold v1.5.5, and
genuinely broad taxonomically (36 pathogenic bacteria, 167 representative
bacterial/archaeal species, human, mouse, *Arabidopsis*, and 81,235 human-virus
candidates). That breadth is exactly what our narrow heterodimer taxonomy
lacks.

**But the paper is CC BY-NC-ND 4.0** — NonCommercial **and** NoDerivatives.
Generating contacts-v1 documents from those structures is squarely a derivative
work. The data deposit's own terms need checking against the article licence
before this source is counted at all; on the article licence alone it is
unusable.

## PPIRef — subsumed

~300,000 PDB interfaces, ~50,000 non-redundant (PPIRef50K at iDist 0.04).
Same provenance as PINDER and roughly an order of magnitude smaller, so its
incremental contribution over PINDER is expected to be small. Not worth
auditing further unless PINDER is rejected.

## Recommendation

1. **Take PINDER.** Apache-2.0, disjoint from AFCDB, +34,690 pairs or +406,495
   structures, +23,639 interface clusters, and the whole audit needed only a
   228 MB index read.
2. **Decide the dedup policy first**, because it decides whether we reach 500k.
   The issue's own redundancy-preserving stance favours per-structure documents
   with interface-aware sampling weights.
3. **Predictomes second**, if human coverage matters — it is close to
   orthogonal to our taxonomy, but needs a confidence gate and a 53 GB download.
4. **Check Protein Complex Atlas's data licence** before spending anything on
   it; the article licence forbids derivatives.
5. **Skip PPIRef** unless PINDER is rejected.

Each would enter as a separately labelled arm, as the issue requires, so the
predicted/experimental mixture stays tunable at training time.
