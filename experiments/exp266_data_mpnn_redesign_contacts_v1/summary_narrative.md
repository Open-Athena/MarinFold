## exp266 — ProteinMPNN redesign of the decontaminated contacts-v1 set

Take all 3,963,003 backbones of the decontaminated AFDB corpus (#225), redesign
each into 8 new sequences with ProteinMPNN, re-run pyconfind, and publish the
~31.7 M-document corpus.

This is the cheap null version of "generate novel structures with BoltzGen /
Proteina and inverse-fold them". That version costs GPU-weeks and changes two
things at once — novel folds *and* MPNN-flavoured sequences. This changes only
the second, so it isolates the part that would have to pay off anyway.

## Two things verified before any code was written

**pyconfind ignores input side chains.** confind rebuilds them from the Dunbrack
rotamer library, so a structure stripped to N/CA/C/O gives bit-identical contact
degrees (4/4 structures, max Δ = 0.000). A backbone plus a residue-name
assignment is therefore a complete pyconfind input — the redesigned corpus is
computed under exactly the same contact operator as contacts-v1, and all-atom
generators would buy us nothing here.

**The contact label is strongly sequence-dependent at fixed geometry.**
Shuffling the sequence on an identical backbone keeps only ~43–54 % of the
native contacts (Jaccard 0.31–0.43). So the 8 redesigns are not 8
near-duplicates, and they supply a contrast the corpus cannot: today every fold
appears with exactly one sequence.

## Where the work runs, and why

CoreWeave's rno2a H100 fleet is prepaid and pinned fully warm, and a live check
found **224 of 512 GPUs and 6,734 vCPUs idle** — free at the margin. rno2a has
no CPU fan-out pool (its single CPU node hosts the controller); cw-us-east-02a
has one, 735 of 768 vCPUs idle.

So Stage B runs as independent 1×H100 tasks on rno2a. The two halves of the
per-backbone work land on different hardware and roughly balance: ~0.1 s of GPU
for 8 ProteinMPNN designs against ~4.2 s of CPU for 8 pyconfind runs, i.e.
~0.3 s across a task's 15 cores. Neither device idles.

The catch: CoreWeave task pods carry only CoreWeave S3 credentials, so they
cannot read AFDB's requester-pays GCS bucket. Backbones are therefore staged
once from GCP as a compact artifact — a measured 11.2 KB/protein, ~46 GB, versus
~700 GB of all-atom mmCIF — encoded as **int32 milli-ångströms** so the hop is
provably lossless. A test asserts the round-trip reproduces the document, its
`sha1` and its `global_plddt` exactly; float32 would have perturbed coordinates
at the scale where a marginal pyconfind contact flips.

## Two upstream traps, and the composition risk

`temperature` broadcasts, so all 8 designs fit in one `sample()` call
(10.38 s → 1.30 s at L=154). `tied_featurize` mis-pads `omit_AA_mask` for
mixed-length batches, so batches must be exact-length — which also means zero
padding waste.

The worry was that ProteinMPNN's Ala/Glu/Lys/Leu bias would shorten the
documents, since contact count collapses for small side chains (poly-ALA gives
~0 contacts). Measured over two 48-protein samples: the largest AA shift is 2.2 percentage
points, and **contacts per residue lands at a ratio of 0.968 and 0.942** (native
0.930/0.902 vs designed 0.900/0.850). The two runs disagree by 0.026, which is
the honest precision of n=48 — real and small, call it 3-6 %, and a number the
pilot has to settle at scale on AFDB rather than PDB structures.

## The smoke ran, and the risk resolved

Stage A2 on GCP and Stage B on CoreWeave both succeeded. The check that
matters: staged rows carry `native_sha1` from the *published* corpus, and
rebuilding from the staged coordinates reproduced **200 / 200** documents
byte-for-byte.

On 400 length-representative backbones, **contacts per residue came out at
1.016 of native** — the feared "Ala-rich designs shorten the documents"
artifact does not happen on AFDB. That overturns the workstation estimates
(0.968 and 0.942), which were measured on better-packed PDB crystal
structures; and 1.040, which came from the 200 *shortest* AFDB entries.
Sampling the right distribution changed the answer.

Measured throughput on a realistic length band is 5.3 backbones/s per 1xH100
task, CPU-bound. The full run is **3.7-7.4 h on 28-56 tasks** (to ~22 h with
slack), on a prepaid fleet that had 224 GPUs idle.

Six cluster-only failures were found and fixed along the way, none reachable
from a local test — including a gcsfs 416 that exposes every marin pipeline
reading parquet from GCS, and a rotamer-library race across the forked
document pool.

Awaiting the go/no-go on the full run.
