## exp262: does contacts-v1 need RoPE?

Issue [#262](https://github.com/Open-Athena/MarinFold/issues/262). contacts-v1
documents are self-describing and order-randomized — every statement carries its
own `<pXXX>` coordinate and statement order is uniform noise — so the proposal is
to delete RoPE and replace it with a width-3 causal *token smear*: mix the
previous two tokens' embeddings into the current one, as in the nanogpt
speedrun's smear module.

Phase 0 asks the trained default checkpoint whether there is anything to win,
before any training is bought. Phase 1 is a 3-arm ablation at the exp232 1.5B
architecture on a reduced token budget.

## Two is the tight window

Enumerating the contacts-v1 grammar: a token's (statement form, slot) role is
NOT a function of the previous 1 token class — `(POS, POS)` is ambiguous between
a contact's second argument and a residue statement's head — and IS a
deterministic function of the previous 2. Lookback 3 adds nothing. Width-3 smear
is the tight bound.

## What Phase 0 found

Three probes on `contacts-v1-exp199-cooldown-1.5B`, teacher-forced over
ground-truth documents from the exp245 monomers, on a local A5000.

**The smear half is directly motivated.** Layer 1 holds two nearly pure
previous-token heads (0.999 and 0.996 of their attention at offset 1) and a third
splitting 0.75/0.16 across offsets 1 and 2 — a width-3 smear implemented in
attention, at the cost of three heads.

**The NoPE half loses its mechanism but keeps its premise.** Co-referent
retrieval is already distance-uniform out to 2048 tokens, so RoPE is not costing
us long-range reach and the long-protein story is dead. But randomizing every
exact cross-statement distance costs *less* than deterministically rescaling them
at matched stretch, so the model genuinely is not reading those distances.

**What position is actually for: counting.** Keeping the position range fixed but
letting two tokens share an id costs +1.23 nats, ten times any
distance-randomizing intervention. The model uses position as an index, which is
exactly what NoPE would take away — and exactly what the pre-registered
stopping-behaviour guardrail was written to catch.

## The pilot: the two changes only work together

15M-parameter twins on 150M tokens of the real decontaminated corpus, three seeds
per arm at each arm's own best learning rate: smear alone buys 0.034 nats, NoPE
alone COSTS 0.085, and together they buy 0.157 — an interaction of 0.208. The
gain is entirely in the structure section, the shuffled bag where the theory said
it should land. Note this overturns the Phase 0 reading, which had the smear as
the safe half and NoPE as the speculative one.

## The answer: no, and by half a model generation

Both full-budget 1.5B runs completed the full 145,200-step schedule. Final
validation loss: control 2.9745, NoPE + width-3 smear 3.0021. The proposal is
0.0275 nats WORSE — about half the 0.053 nats the whole #75 to #117 generation
was worth, in the wrong direction. Seventeen matched post-transition evals
average +0.0166 and the final five agree to within 0.003, so this is not a
marginal call.

The control finished 0.0173 better than exp232's run of the same recipe, which
both confirms the newer marin pin is loss-neutral and bounds the run-to-run
spread of the setup at about the size of the effect we were chasing.

## Two things that outlast the negative result

The cheap proxies inverted the sign. A 15M pilot said -0.157 and a 10%-budget
screen at the correct model size and data said -0.174; the answer was +0.0275.
The screen is the worse offender: it ran the production model on production data
and still got the sign wrong, because 14,520 steps ends before the transition
below.

Every 1.5B run has one ~0.09-nat learning transition mid-training whose timing is
NOT reproducible — the control's is near 15.5k and exp232's near 22k, and those
are the same architecture, data and seeds. That 6,500-step jitter is three times
the effect under test. Any future mid-run architecture comparison at this scale
needs seed replicates, and nobody had characterised this before.
