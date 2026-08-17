# Summary slides — exp241: why does eval2-natural exist?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Prose only — the renderer reflows each paragraph with textwrap, so
     markdown tables come out as raw pipes. Keep headings under ~45 chars
     so they do not run off the slide. -->

## The question

eval2 (#226) keeps eval proteins whose sequence identity to our training data is
under 40%. 307 survive and 78 are labelled natural — the subset every
novel-protein claim now rests on.

But almost every natural protein deposited to the PDB recently has a sequence
that was determined years earlier, is in UniProt, and therefore has an AlphaFold
model. On that reasoning eval2-natural should be close to empty. It is 78, and
53 of the 78 have zero significant hits anywhere in 70.9M training sequences, at
a median length of 148 residues. A 400-residue natural protein with no
detectable homolog in 70M sequences should not happen.

## The filter, exactly

40%, not 30%. A protein is in eval2 if the best MMseqs2 identity over hits with
evalue <= 1e-3 and query coverage >= 0.50 is below 0.40, against the union of
both training arms — or if it has no such hit at all. 30% is a retrospective
column, not the definition.

The comparison target is not "everything known". It is 70,889,604 sequences:
4.1M from AFDB, which is 1.9% of AFDB v4, plus 66.8M ESM Atlas cluster
representatives. And the AFDB arm was filtered twice against structurally
singular proteins — afdb-24M requires cluster membership and drops singletons,
then exp53 keeps five members per structural cluster and drops any cluster with
fewer than three.

## 15 of the 78 are not natural

exp226 resolved source organisms only for the FoldBench rows, so all 24 CAMEO
and 19 CASP rows carried "natural" as a default — nothing had ever looked.

Looking finds 14 entities whose source organism is "synthetic construct" and 10
entries keyworded DE NOVO PROTEIN; 15 distinct proteins, all from CAMEO-hard,
every one unambiguous from its own title: "De novo Design of Near Infrared
Fluorescent Proteins", "The designed serine hydrolase known as win1". CAMEO
draws from weekly PDB pre-releases, which are full of design-lab depositions.

The correction runs one way only — 0 of the 396 de novo rows look natural. So
eval2-natural is 63, not 78, and eval2 is 77% designed.

## The sequences are old, and we did not train on them

Of the 63 audited-natural proteins: 60 have a UniProt sequence entry, AlphaFold
has a model for 45 of them, and 0 are in our AFDB training arm.

UniProt first published these sequences a median of 15 years ago — median 2011,
range 1987 to 2026. P12255 from Bordetella pertussis has been public since
October 1989. These are not new sequences.

What is wrong is the step "in AFDB, therefore in our training set". The arm is
1.9% of AFDB. So 45 of the 63 are proteins AlphaFold folded and we simply did
not sample.

## Viral proteins are the hole in both corpora

Viral eval proteins have an AFDB-arm hit 22% of the time and an ESM-Atlas hit
41% of the time, against 88% and 84% for bacteria. 66% of them survive into
eval2, versus 15% of bacterial and 13% of eukaryotic proteins.

27 of the 63 — 43% of eval2-natural — are viral. AFDB's pLDDT and
cluster-membership filters strip poorly-modelled and structurally singular viral
proteins; the ESM Atlas is metagenomic, so it carries phage but not eukaryotic
viruses.

## The base rate: 7%

Everything above conditions on eval2 membership, and eval2 is the filter, so it
is partly circular. The uncircular measurement: 585 protein chains sampled at
random from the 183,327 RCSB entities deposited since 2022, put through eval2's
own filter and target database.

7.1% of random recent natural PDB chains have no 40%-identity relative in our
training set — 95% CI 5.2% to 9.5%. One in fourteen, before any eval curation.
The kingdom effect reproduces independently: viral 31.4% versus bacterial 1.8%,
Fisher odds ratio 25.6, p = 4e-09.

## The correction helps MarinFold

Moving the 15 designs to the designed side and re-aggregating exp226's own
per-protein scores — same bootstrap, same seed, only membership changes — moves
the baselines further than it moves MarinFold. Protenix-v2 single-seq drops from
0.326 to 0.230; MarinFold drops from 0.337 to 0.313.

So exp226's finding gets stronger, not weaker. It reported parity with Protenix
single-seq on the natural half: +0.011, a tie. On the audited natural half
MarinFold wins by +0.083, CI +0.031 to +0.136. The 15 designs were where the
baseline was strong.

Everything else holds: MarinFold still loses to ESMFold, ESMFold2 and
Protenix+MSA on eval2-natural, all significant, and still beats the seq-KNN null.

## Viral and non-viral rank differently

27 of the 63 are viral, so the stratification is not cosmetic. On viral proteins
MarinFold scores 0.253 and ties ESMFold — paired delta -0.004, not significant.
On non-viral it scores 0.359 and loses to ESMFold by 0.145.

The mirror holds against Protenix single-seq: MarinFold is +0.113 on non-viral
and +0.043 on viral. A single pooled eval2-natural number averages two regimes
with different rankings.

Caveat on the checkpoint: the bars are exp199 p06, the model every baseline was
scored beside. The current default is the p06 cooldown, which scores 0.358
against p06's 0.337 on the published n=78. Its per-protein eval2 rows are on
CoreWeave S3 and need one in-cluster job to re-cut to n=63.

## Conclusion

eval2-natural exists because our training corpus is not everything known — it is
a 1.9% sample of AFDB plus 67M metagenomic cluster representatives — and because
a fifth of the set was never natural.

Three things change for how it is used. The n is 63, not 78, and the 15 designs
belong on the designed side of the split eval2 exists to make. "No homolog in the
training set" means unsampled, not novel — a generalisation claim needs the
fold-novelty axis, not this filter. And eval2-natural is 43% viral, so a headline
on it is substantially a statement about viral protein structure.

The 7% base rate over 183k recent PDB entities is roughly 13,000 candidate
natural chains. Sampling that population directly is the route to an
eval2-natural of several hundred, and unlike the FoldBench expansion it does not
run out.
