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

## Status

Corpus staged, tokenized and audited. The scorer is validated on both exp277's
and exp343's own export formats, and agrees with levanter's own complex
validation loss to three significant figures. The monomer eval is ported with a
byte-identical worker.

Production training is submitted and waiting on a 128-H100 gang;
`cw-us-east-02a` went to zero free GPUs shortly after submission. Results and the
exp343 series on the figure below land here as they arrive.
