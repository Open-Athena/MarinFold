# exp343 — the exp277 corpus plus predicted complexes

MarinFold's default model, `contacts-v1-exp277-m2-p06-full-epoch-1.5B`, has
never seen a protein complex. Every document it trained on is one chain.

[#294](https://github.com/Open-Athena/MarinFold/issues/294) built 3,410,738
multi-chain contacts-v1 documents — 11.15 B tokens, 262 M interface contacts,
611,656 heterodimers — and published them. This experiment trains the same 1.5B
model, on the same settings, over exp277's entire corpus **plus** those
complexes, for one epoch.

## What "matched settings" means here

Everything that defines the baseline is imported, not restated: exp232's
m2-p06 model and optimizer contract, exp277's finite one-epoch data config, and
exp277's pinned dependency lock. Scratch init, seq len 8192, global batch 128,
LR 1e-3, weight decay 0.2, WSD 10%/20%/0.1, packed with blocked cross-document
attention, scheduled amino-acid order augmentation, seeds 0 and 0.

Two things differ, and only two: a fifth training corpus, and a second
validation set.

## Why a shard is held out

The monomer eval sets contain no complexes. A model that learned the multi-chain
format and one that ignored it entirely would score the same on eval-val. So
shard 00170 — 10,738 documents, 0.31% of the complex corpus — never enters
training and becomes a complex validation set. Its loss, against exp277's on the
same documents, is the measurement this experiment exists for.

## What the complex corpus is, in the mixture

11.15 B tokens against exp277's 248.58 B: **4.3%**. That is the honest framing
of the expected monomer result. A 4.3% mixture change spent on a document type
the benchmark does not contain should leave eval-val where it was; the
predeclared threshold is 0.005 and the expectation is a tie. The interesting
outcomes are a complex loss that drops a long way, and any movement at all on
designed proteins.

## The tokenizer trap, checked rather than assumed

The corpus ships a 2,848-token tokenizer. The model has 2,845 embedding rows.
The three extra ids belong to other document structures and are unused here, and
below 2,845 the two vocabularies are identical — so the corpus is tokenized with
the model's tokenizer and the embedding table is untouched. Every record is
validated against that vocabulary as the caches are built, so a document
reaching an out-of-contract id fails the build instead of indexing past the
table.

## The baseline, measured before the experiment exists

exp277 scored on all 10,738 held-out complex documents: **3.697 nats/token**
overall. Split by what each token encodes, it says where a complex-blind model
actually fails.

It pays **2.957** on intra-chain contacts and **3.452** on inter-chain — it
handles the contacts that look like monomer contacts, and is half a nat worse on
the ones that cross a chain boundary. Inter-chain is 12.5% of contact tokens, so
a single fused loss would be dominated by the part it already knows. That is why
the split exists.

The chain-terminus statements cost **11.81 nats**. Uniform over the whole
2,845-token vocabulary is 7.95, so exp277 is not uncertain there — it is
confidently wrong. A monomer model has only ever seen one `<n-term>`/`<c-term>`
pair per document; a complex has *k* of them at unpredictable ring positions.

Homodimers are much easier than heterodimers (3.363 against 4.329 inter-chain): a
homodimer interface is two copies of one sequence, so intra-chain knowledge partly
transfers. PINDER's experimental crystallised fragments are hardest at 4.651.

## The answer: a trade, not a free win

**Monomers got worse.** eval-val R-precision 0.52739 against exp277's 0.55374 —
**−0.0264**, with a paired protein-bootstrap interval of [−0.0392, −0.0148]. The
predeclared hypothesis was a tie within 0.005; this is five times that. Five of
six subset/range intervals exclude zero. For scale, the drop is larger than the
entire gain exp277 made over exp232 on legacy 554.

The evaluation was run twice end to end, 67,000 freshly sampled rollouts each
time; the draws agree to 5.3e-4 on eval-val, so the effect is ~50x the sampling
noise. The complex evaluation, a deterministic forward pass, reproduces to
3e-11.

**Complexes got much better.** Held-out complex loss 3.0369 against 3.6970 —
**−0.660 nats** — and the split by token role says exactly where it came from.

Chain termini: 11.81 → **3.48**. exp277 was paying more than a uniform
distribution over the vocabulary, i.e. it was confidently wrong about how many
chains a document has. That is gone.

Inter-chain contacts: 3.452 → **2.541**. exp277 was 0.495 nats worse on contacts
crossing a chain boundary than on contacts within one; exp343 is **0.060** worse.
The interface penalty essentially closed — which is the specific thing the #294
corpus was built to teach, and it transferred.

## What a reader should take from this

Spending 4.3% of the training mixture on complex documents buys a large,
well-localised gain on complexes and costs about 0.027 R-precision on natural
monomer contact prediction. Anyone making this the default mixture is choosing
between those two, and this experiment prices the choice rather than making it.

One confound limits the causal claim: exp343 saw 11.15 B tokens exp277 did not,
and this is one seed per arm. Nothing here separates "complex documents hurt
monomers" from "this much extra data, at this budget, hurts monomers". A
token-matched control is the follow-up that would settle it.

## Status

Complete. Training finished at step 280,154; both evaluations are in and
committed. Final numbers and the figure below.
